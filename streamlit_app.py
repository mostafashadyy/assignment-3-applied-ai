import csv
import json
import os
import re
from datetime import datetime
from time import perf_counter

import streamlit as st

from advanced_tools import execute_tools_parallel, execute_tools_sequential
from conversational_agent import (
    LLM_MODEL,
    advanced_system_message,
    advanced_tools as agent_advanced_tools,
    available_functions,
    client,
    cot_system_message,
    cot_tools,
    get_last_assistant_text,
    process_messages,
    process_messages_advanced,
    to_assistant_message_dict,
    weather_available_functions,
    weather_tools,
)

SYSTEM_BASIC = "You are a helpful weather assistant. Respond directly and clearly."
SYSTEM_COT = cot_system_message
SYSTEM_ADVANCED = advanced_system_message
CSV_PATH = "evaluation_results.csv"


def apply_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f8fc 0%, #ffffff 35%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .app-card {
            background: #ffffff;
            border: 1px solid #e8ebf3;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
            margin-bottom: 0.8rem;
        }
        .app-kicker {
            color: #69738a;
            font-size: 0.92rem;
            margin-top: -0.2rem;
            margin-bottom: 0.2rem;
        }
        .title-row {
            background: #ffffff;
            border: 1px solid #e8ebf3;
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header():
    st.markdown(
        """
        <div class="title-row">
            <h2 style="margin:0;">🌤️ Weather Assignment Studio</h2>
            <div class="app-kicker">Normal assignment mode + bonus evaluation in one polished interface.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.caption("Use mode selector in main view to switch workflows.")
        if st.button("Clear all app state", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("### 📁 Saved Results")
        if os.path.exists(CSV_PATH):
            st.success(f"Found {CSV_PATH}")
        else:
            st.info("No saved CSV yet")


def extract_locations(text):
    matches = re.findall(r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s]+?)(?=\s*(?:,|and|\?|$))", text, flags=re.IGNORECASE)
    clean = []
    for item in matches:
        location = " ".join(item.split()).strip(" ,.?")
        if location and location.lower() not in {value.lower() for value in clean}:
            clean.append(location)
    return clean


def run_basic_agent(query):
    messages = [
        {"role": "system", "content": SYSTEM_BASIC},
        {"role": "user", "content": query},
    ]
    start = perf_counter()
    messages = process_messages(
        client,
        messages,
        weather_tools,
        weather_available_functions,
    )
    elapsed = perf_counter() - start
    content = get_last_assistant_text(messages)
    return content, elapsed


def run_cot_agent(query):
    messages = [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": query},
    ]
    start = perf_counter()
    messages = process_messages(
        client,
        messages,
        cot_tools,
        available_functions,
    )
    elapsed = perf_counter() - start
    content = get_last_assistant_text(messages)
    return content, elapsed


def run_advanced_agent(query):
    messages = [
        {"role": "system", "content": SYSTEM_ADVANCED},
        {"role": "user", "content": query},
    ]
    start = perf_counter()

    final_message = None
    for _ in range(5):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=agent_advanced_tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        messages.append(to_assistant_message_dict(response_message))

        if response_message.tool_calls:
            tool_results = execute_tools_parallel(response_message.tool_calls, available_functions)
            messages.extend(tool_results)
            continue

        final_message = response_message
        break

    if final_message is None:
        content = get_last_assistant_text(messages)
    else:
        content = final_message.content or "(No response generated)"

    elapsed = perf_counter() - start
    return content, elapsed


def benchmark_sequential_vs_parallel(locations):
    tool_calls = []

    for index, location in enumerate(locations):
        tool_calls.append(
            {
                "id": f"manual_call_{index}",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": json.dumps({"location": location}),
                },
            }
        )

    class FunctionCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ToolCall:
        def __init__(self, call_id, function_name, function_arguments):
            self.id = call_id
            self.function = FunctionCall(function_name, function_arguments)

    parsed_calls = [
        ToolCall(item["id"], item["function"]["name"], item["function"]["arguments"])
        for item in tool_calls
    ]

    seq_start = perf_counter()
    execute_tools_sequential(parsed_calls, available_functions)
    sequential_time = perf_counter() - seq_start

    par_start = perf_counter()
    execute_tools_parallel(parsed_calls, available_functions)
    parallel_time = perf_counter() - par_start

    if parallel_time == 0:
        speedup = 0.0
    else:
        speedup = sequential_time / parallel_time

    return sequential_time, parallel_time, speedup


def save_results_to_csv(data):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def render_chat_messages(messages):
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = message.get("content") or ""
        if role == "assistant" and not content.strip():
            continue

        with st.chat_message(role):
            st.markdown(content)


def get_mode_config(agent_choice):
    if agent_choice == "Basic":
        return {
            "system": SYSTEM_BASIC,
            "tools": weather_tools,
            "functions": weather_available_functions,
            "advanced": False,
            "title": "Basic Agent",
        }

    if agent_choice == "Chain of Thought":
        return {
            "system": SYSTEM_COT,
            "tools": cot_tools,
            "functions": available_functions,
            "advanced": False,
            "title": "Chain of Thought Agent",
        }

    return {
        "system": SYSTEM_ADVANCED,
        "tools": agent_advanced_tools,
        "functions": available_functions,
        "advanced": True,
        "title": "Advanced Agent",
    }


def ensure_normal_chat_state(agent_choice):
    config = get_mode_config(agent_choice)

    if "normal_agent" not in st.session_state:
        st.session_state.normal_agent = agent_choice

    if "normal_messages" not in st.session_state:
        st.session_state.normal_messages = [{"role": "system", "content": config["system"]}]

    if st.session_state.normal_agent != agent_choice:
        st.session_state.normal_agent = agent_choice
        st.session_state.normal_messages = [{"role": "system", "content": config["system"]}]


def run_normal_assignment_mode():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Normal Assignment")
    st.caption("Choose one agent and chat normally. Same logic, cleaner interface.")

    agent_choice = st.selectbox(
        "Agent type",
        ["Basic", "Chain of Thought", "Advanced"],
        index=0,
    )
    config = get_mode_config(agent_choice)
    ensure_normal_chat_state(agent_choice)

    top_cols = st.columns([1, 2, 2])
    with top_cols[0]:
        if st.button("Clear normal chat"):
            st.session_state.normal_messages = [{"role": "system", "content": config["system"]}]
            st.rerun()
    with top_cols[1]:
        st.success(f"Running: {config['title']}")
    with top_cols[2]:
        st.info(f"Messages: {len(st.session_state.normal_messages) - 1}")

    render_chat_messages(st.session_state.normal_messages)

    prompt = st.chat_input("Ask about weather, forecast, or calculations...", key="normal_chat_input")

    if prompt:
        st.session_state.normal_messages.append({"role": "user", "content": prompt})
        try:
            if config["advanced"]:
                for _ in range(5):
                    st.session_state.normal_messages, response_message = process_messages_advanced(
                        client,
                        st.session_state.normal_messages,
                        config["tools"],
                        config["functions"],
                    )
                    if not response_message.tool_calls:
                        break
            else:
                st.session_state.normal_messages = process_messages(
                    client,
                    st.session_state.normal_messages,
                    config["tools"],
                    config["functions"],
                )
        except Exception as error:
            st.session_state.normal_messages.pop()
            st.error(f"Request failed: {error}")

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def run_bonus_mode():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Bonus Evaluation")
    st.caption("One query → three agents → speedup metrics → ratings → CSV.")

    if "latest_run" not in st.session_state:
        st.session_state.latest_run = None

    query = st.chat_input("Enter one query to test all agents...", key="bonus_chat_input")

    if query:
        with st.spinner("Running Basic, Chain of Thought, and Advanced agents..."):
            basic_response, basic_time = run_basic_agent(query)
            cot_response, cot_time = run_cot_agent(query)
            advanced_response, advanced_time = run_advanced_agent(query)

        locations = extract_locations(query)
        used_fallback_locations = False
        if len(locations) < 2:
            locations = ["Cairo", "Giza", "Alexandria"]
            used_fallback_locations = True

        with st.spinner("Measuring sequential vs parallel tool execution..."):
            sequential_time, parallel_time, speedup = benchmark_sequential_vs_parallel(locations)

        st.session_state.latest_run = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "basic_response": basic_response,
            "basic_time": basic_time,
            "cot_response": cot_response,
            "cot_time": cot_time,
            "advanced_response": advanced_response,
            "advanced_time": advanced_time,
            "locations_benchmarked": ", ".join(locations),
            "used_fallback_locations": used_fallback_locations,
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
        }

    if st.session_state.latest_run:
        run = st.session_state.latest_run

        response_cols = st.columns(3)
        with response_cols[0]:
            st.subheader("Basic")
            st.metric("Latency (s)", f"{run['basic_time']:.2f}")
            st.write(run["basic_response"])

        with response_cols[1]:
            st.subheader("Chain of Thought")
            st.metric("Latency (s)", f"{run['cot_time']:.2f}")
            st.write(run["cot_response"])

        with response_cols[2]:
            st.subheader("Advanced")
            st.metric("Latency (s)", f"{run['advanced_time']:.2f}")
            st.write(run["advanced_response"])

        st.divider()

        timing_cols = st.columns(3)
        with timing_cols[0]:
            st.metric("Sequential Tool Time (s)", f"{run['sequential_time']:.2f}")
        with timing_cols[1]:
            st.metric("Parallel Tool Time (s)", f"{run['parallel_time']:.2f}")
        with timing_cols[2]:
            st.metric("Observed Speedup", f"{run['speedup']:.2f}x")

        if run["used_fallback_locations"]:
            st.info(f"Query was not clearly multi-location, so benchmark used: {run['locations_benchmarked']}")
        else:
            st.info(f"Benchmark locations extracted from query: {run['locations_benchmarked']}")

        st.divider()
        st.subheader("Rate Response Quality (1-5)")

        rating_cols = st.columns(3)
        with rating_cols[0]:
            basic_rating = st.selectbox("Basic rating", [1, 2, 3, 4, 5], key="basic_rating")
        with rating_cols[1]:
            cot_rating = st.selectbox("Chain of Thought rating", [1, 2, 3, 4, 5], key="cot_rating")
        with rating_cols[2]:
            advanced_rating = st.selectbox("Advanced rating", [1, 2, 3, 4, 5], key="advanced_rating")

        if st.button("Save ratings and timings"):
            row = {
                "timestamp": run["timestamp"],
                "query": run["query"],
                "basic_response": run["basic_response"],
                "basic_time_seconds": f"{run['basic_time']:.6f}",
                "basic_rating": basic_rating,
                "cot_response": run["cot_response"],
                "cot_time_seconds": f"{run['cot_time']:.6f}",
                "cot_rating": cot_rating,
                "advanced_response": run["advanced_response"],
                "advanced_time_seconds": f"{run['advanced_time']:.6f}",
                "advanced_rating": advanced_rating,
                "locations_benchmarked": run["locations_benchmarked"],
                "sequential_tool_time_seconds": f"{run['sequential_time']:.6f}",
                "parallel_tool_time_seconds": f"{run['parallel_time']:.6f}",
                "observed_speedup_x": f"{run['speedup']:.6f}",
                "used_fallback_locations": run["used_fallback_locations"],
            }
            save_results_to_csv(row)
            st.success(f"Saved to {CSV_PATH}")

    st.markdown('</div>', unsafe_allow_html=True)


st.set_page_config(page_title="Weather Assignment UI", page_icon="🌤️", layout="wide")
apply_custom_style()
render_top_header()
render_sidebar()

mode = st.radio(
    "Choose mode",
    ["Normal Assignment", "Bonus Evaluation"],
    horizontal=True,
)

if mode == "Normal Assignment":
    run_normal_assignment_mode()
else:
    run_bonus_mode()
