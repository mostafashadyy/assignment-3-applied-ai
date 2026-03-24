import os
import json
import csv
import re
from time import perf_counter
from openai import OpenAI
from openai import BadRequestError
from dotenv import load_dotenv

from tools import get_current_weather, get_weather_forecast, calculator
from advanced_tools import (
    execute_tools_parallel,
    compare_parallel_vs_sequential,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

LLM_MODEL = "openai/gpt-oss-120b"

weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get the weather forecast for a location for a specific "
                "number of days"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days to forecast (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": ["location"],
            },
        },
    },
]

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, "
                        "e.g., '2 + 2' or '5 * (3 + 2)'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

cot_tools = weather_tools + [calculator_tool]
advanced_tools = cot_tools

weather_available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
}

available_functions = {
    **weather_available_functions,
    "calculator": calculator,
}

cot_system_message = """You are a helpful assistant that can answer questions about weather and perform calculations.
Use only these function tools when needed: get_current_weather, get_weather_forecast, calculator.
Never call or mention any other tools (for example: web.run, browser.search, web_search, news tools).
When responding to complex questions, please follow these steps:
1. Think step-by-step about what information you need.
2. Break down the problem into smaller parts.
3. Use the appropriate tools to gather information.
4. Explain your reasoning clearly.
5. Provide a clear final answer.
For example, if someone asks about temperature conversions or comparisons between cities, first get the weather data, then use the calculator if needed, showing your work.
"""

advanced_system_message = """You are a helpful weather assistant that can use weather tools and a calculator to solve multi-step problems.
Use only these function tools when needed: get_current_weather, get_weather_forecast, calculator.
Never call or mention any other tools (for example: web.run, browser.search, web_search, news tools).
Guidelines:
1. If the user asks about several independent locations, use multiple weather tool calls in parallel when appropriate.
2. If a question requires several steps, continue using tools until the task is completed.
3. If a tool fails, explain the issue clearly and continue safely when possible.
4. For complex comparison or calculation queries, prepare a structured final response.
"""

required_output_keys = [
    "query_type",
    "locations",
    "summary",
    "tool_calls_used",
    "final_answer",
]

structured_output_prompt = """For complex comparison or calculation queries,
return the final answer as a valid JSON object with exactly these keys:
- query_type
- locations
- summary
- tool_calls_used
- final_answer
Do not include markdown fences.
"""


def to_assistant_message_dict(message):
    assistant_message = {
        "role": message.role,
        "content": message.content if message.content is not None else "",
    }

    if message.tool_calls:
        assistant_message["tool_calls"] = [
            tool_call.model_dump() if hasattr(tool_call, "model_dump") else tool_call
            for tool_call in message.tool_calls
        ]

    return assistant_message


def get_last_assistant_text(messages):
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content") or "(No response generated)"
    return "(No response generated)"


def call_tool_direct(tool_call, function_map):
    function_name = tool_call.function.name
    if function_name not in function_map:
        return f"Error: Unknown function {function_name}"

    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as error:
        return f"Error: Invalid JSON arguments: {str(error)}"

    try:
        return function_map[function_name](**function_args)
    except Exception as error:
        return f"Error: {str(error)}"


def process_messages(client, messages, tools=None, available_functions=None, max_iterations=5):
    tools = tools or []
    available_functions = available_functions or {}

    try:
        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else "none",
            )

            response_message = response.choices[0].message
            messages.append(to_assistant_message_dict(response_message))

            if not response_message.tool_calls:
                return messages

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_response = call_tool_direct(tool_call, available_functions)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

        messages.append(
            {
                "role": "assistant",
                "content": "I stopped after reaching the maximum number of tool iterations.",
            }
        )
        return messages
    except BadRequestError as error:
        if "Tool choice is none, but model called a tool" not in str(error):
            raise

        fallback_messages = messages + [
            {
                "role": "system",
                "content": (
                    "Do not call any tools. Provide a direct best-effort response "
                    "based only on the existing conversation and tool outputs."
                ),
            }
        ]
        fallback_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=fallback_messages,
            tool_choice="none",
        )
        fallback_message = fallback_response.choices[0].message
        messages.append(to_assistant_message_dict(fallback_message))
        return messages


def process_messages_advanced(client, messages, tools=None, available_functions=None):
    tools = tools or []
    available_functions = available_functions or {}

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else "none",
    )

    response_message = response.choices[0].message
    messages.append(to_assistant_message_dict(response_message))

    if response_message.tool_calls:
        tool_results = execute_tools_parallel(
            response_message.tool_calls,
            available_functions,
        )
        messages.extend(tool_results)

    return messages, response_message


def validate_structured_output(response_text):
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON output: {str(error)}")

    for key in required_output_keys:
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(parsed["locations"], list):
        raise ValueError("'locations' must be a list")
    if not isinstance(parsed["tool_calls_used"], list):
        raise ValueError("'tool_calls_used' must be a list")

    return parsed


def get_structured_final_response(client, messages):
    structured_messages = messages + [
        {
            "role": "system",
            "content": structured_output_prompt,
        }
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=structured_messages,
        response_format={"type": "json_object"},
        tool_choice="none",
    )

    content = response.choices[0].message.content or "{}"
    return validate_structured_output(content)


def should_use_structured_output(user_input):
    keywords = [
        "compare",
        "difference",
        "average",
        "warmer",
        "colder",
        "higher",
        "lower",
        "temperature difference",
    ]
    lower_input = user_input.lower()
    return any(keyword in lower_input for keyword in keywords)


def run_conversation(
    client,
    system_message="You are a helpful weather assistant.",
    tools=None,
    available_functions=None,
    assistant_name="Weather Assistant",
):
    tools = tools or weather_tools
    available_functions = available_functions or weather_available_functions

    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    print(f"{assistant_name}: Hello! I can help you with weather information.")
    print("Ask me about the weather anywhere!")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"\n{assistant_name}: Goodbye! Have a great day!")
            break
        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            messages = process_messages(
                client,
                messages,
                tools,
                available_functions,
            )
            print(f"\n{assistant_name}: {get_last_assistant_text(messages)}\n")
        except BadRequestError as error:
            print(f"\n{assistant_name}: Request failed: {error}\n")
            messages.pop()
        except Exception as error:
            print(f"\n{assistant_name}: Unexpected error: {error}\n")
            messages.pop()

    return messages


def run_conversation_advanced(
    client,
    system_message=advanced_system_message,
    max_iterations=5,
):
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    print("Advanced Weather Assistant: Hello! Ask me complex weather questions.")
    print("I can compare cities, perform calculations, and return structured outputs.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAdvanced Weather Assistant: Goodbye! Have a great day!")
            break
        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            for _ in range(max_iterations):
                messages, response_message = process_messages_advanced(
                    client,
                    messages,
                    advanced_tools,
                    available_functions,
                )

                if not response_message.tool_calls:
                    if should_use_structured_output(user_input):
                        try:
                            structured_result = get_structured_final_response(client, messages)
                            print("\nAdvanced Weather Assistant (Structured JSON):")
                            print(json.dumps(structured_result, indent=2))
                            print()
                        except Exception:
                            print(
                                "\nAdvanced Weather Assistant:"
                                f" {get_last_assistant_text(messages)}\n"
                            )
                    else:
                        print(
                            "\nAdvanced Weather Assistant:"
                            f" {get_last_assistant_text(messages)}\n"
                        )
                    break
            else:
                print(
                    "\nAdvanced Weather Assistant: I stopped after reaching the"
                    " maximum number of tool iterations.\n"
                )
        except BadRequestError as error:
            print(f"\nAdvanced Weather Assistant: Request failed: {error}\n")
            messages.pop()
        except Exception as error:
            print(f"\nAdvanced Weather Assistant: Unexpected error: {error}\n")
            messages.pop()

    return messages


def run_single_query_mode(client, query, system_message, tools, available_functions, max_iterations=1):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    start = perf_counter()

    if max_iterations == 1:
        messages = process_messages(client, messages, tools, available_functions)
    else:
        for _ in range(max_iterations):
            messages, response_message = process_messages_advanced(
                client,
                messages,
                tools,
                available_functions,
            )
            if not response_message.tool_calls:
                break

    elapsed = perf_counter() - start
    return get_last_assistant_text(messages), elapsed


def extract_locations_for_bonus(query):
    matches = re.findall(
        r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s]+?)(?=\s*(?:,|and|\?|$))",
        query,
        flags=re.IGNORECASE,
    )
    locations = []
    for item in matches:
        location = " ".join(item.split()).strip(" ,.?")
        if location and location.lower() not in [value.lower() for value in locations]:
            locations.append(location)
    return locations


class _FunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, tool_id, function_name, function_arguments):
        self.id = tool_id
        self.function = _FunctionCall(function_name, function_arguments)


def create_multilocation_tool_calls(locations):
    return [
        _ToolCall(
            f"benchmark_call_{index}",
            "get_current_weather",
            json.dumps({"location": location}),
        )
        for index, location in enumerate(locations)
    ]


def get_rating(prompt_text):
    while True:
        value = input(prompt_text).strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        print("Please enter a number from 1 to 5.")


def save_bonus_results_to_csv(row, file_path="evaluation_results.csv"):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_bonus_evaluation(client):
    print("Bonus Evaluation Mode")
    print("Enter one query and I will run Basic, Chain of Thought, and Advanced agents.")
    query = input("Query: ").strip()
    if not query:
        print("No query entered.")
        return

    basic_response, basic_time = run_single_query_mode(
        client,
        query,
        "You are a helpful weather assistant.",
        weather_tools,
        weather_available_functions,
        max_iterations=1,
    )
    cot_response, cot_time = run_single_query_mode(
        client,
        query,
        cot_system_message,
        cot_tools,
        available_functions,
        max_iterations=1,
    )
    advanced_response, advanced_time = run_single_query_mode(
        client,
        query,
        advanced_system_message,
        advanced_tools,
        available_functions,
        max_iterations=5,
    )

    locations = extract_locations_for_bonus(query)
    used_fallback_locations = False
    if len(locations) < 2:
        locations = ["Cairo", "Riyadh", "London"]
        used_fallback_locations = True

    benchmark_tool_calls = create_multilocation_tool_calls(locations)
    benchmark = compare_parallel_vs_sequential(benchmark_tool_calls, available_functions)

    print("\n=== Responses (Side by Side) ===")
    print(f"Basic ({basic_time:.2f}s):\n{basic_response}\n")
    print(f"Chain of Thought ({cot_time:.2f}s):\n{cot_response}\n")
    print(f"Advanced ({advanced_time:.2f}s):\n{advanced_response}\n")

    print("=== Timing Comparison ===")
    print(f"Locations benchmarked: {', '.join(locations)}")
    print(f"Sequential tool time: {benchmark['sequential_time']:.4f}s")
    print(f"Parallel tool time:   {benchmark['parallel_time']:.4f}s")
    if benchmark["speedup"] is None:
        print("Observed speedup: N/A")
    else:
        print(f"Observed speedup: {benchmark['speedup']:.2f}x")

    print("\nRate each response from 1 to 5")
    basic_rating = get_rating("Basic rating (1-5): ")
    cot_rating = get_rating("Chain of Thought rating (1-5): ")
    advanced_rating = get_rating("Advanced rating (1-5): ")

    row = {
        "query": query,
        "basic_response": basic_response,
        "basic_time_seconds": f"{basic_time:.6f}",
        "basic_rating": basic_rating,
        "cot_response": cot_response,
        "cot_time_seconds": f"{cot_time:.6f}",
        "cot_rating": cot_rating,
        "advanced_response": advanced_response,
        "advanced_time_seconds": f"{advanced_time:.6f}",
        "advanced_rating": advanced_rating,
        "locations_benchmarked": ", ".join(locations),
        "used_fallback_locations": used_fallback_locations,
        "sequential_tool_time_seconds": f"{benchmark['sequential_time']:.6f}",
        "parallel_tool_time_seconds": f"{benchmark['parallel_time']:.6f}",
        "observed_speedup_x": (
            f"{benchmark['speedup']:.6f}" if benchmark["speedup"] is not None else "N/A"
        ),
    }

    save_bonus_results_to_csv(row)
    print("\nSaved ratings and timing results to evaluation_results.csv\n")

if __name__ == "__main__":
    choice = input(
        "Choose mode (1: Basic, 2: Chain of Thought, 3: Advanced, 4: Bonus Evaluation): "
    ).strip()

    if choice == "1":
        run_conversation(
            client,
            "You are a helpful weather assistant.",
            weather_tools,
            weather_available_functions,
            "Weather Assistant",
        )
    elif choice == "2":
        run_conversation(
            client,
            cot_system_message,
            cot_tools,
            available_functions,
            "Chain of Thought Assistant",
        )
    elif choice == "3":
        run_conversation_advanced(client, advanced_system_message)
    elif choice == "4":
        run_bonus_evaluation(client)
    else:
        print("Invalid choice. Defaulting to Basic agent.")
        run_conversation(
            client,
            "You are a helpful weather assistant.",
            weather_tools,
            weather_available_functions,
            "Weather Assistant",
        )