import json
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

def execute_tool_safely(tool_call, available_functions):
    function_name = tool_call.function.name

    if function_name not in available_functions:
        return json.dumps({
            "success": False,
            "error": f"Unknown function: {function_name}",
        })

    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as error:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON arguments: {str(error)}",
        })

    try:
        result = available_functions[function_name](**function_args)
        return json.dumps({
            "success": True,
            "function_name": function_name,
            "result": result
        })
    except TypeError as error:
        return json.dumps({
            "success": False,
            "error": f"Invalid arguments: {str(error)}",
        })
    except Exception as error:
        return json.dumps({
            "success": False,
            "error": f"Tool execution failed: {str(error)}",
        })


def execute_tools_parallel(tool_calls, available_functions):
    if not tool_calls:
        return []

    def run_tool(tool_call):
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": execute_tool_safely(tool_call, available_functions),
        }

    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        return list(executor.map(run_tool, tool_calls))


def execute_tools_sequential(tool_calls, available_functions):
    tool_results = []

    for tool_call in tool_calls:
        tool_results.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": execute_tool_safely(tool_call, available_functions),
        })

    return tool_results


def compare_parallel_vs_sequential(tool_calls, available_functions):
    seq_start = perf_counter()
    sequential_results = execute_tools_sequential(tool_calls, available_functions)
    sequential_time = perf_counter() - seq_start

    par_start = perf_counter()
    parallel_results = execute_tools_parallel(tool_calls, available_functions)
    parallel_time = perf_counter() - par_start

    speedup = sequential_time / parallel_time if parallel_time > 0 else None

    return {
        "sequential_results": sequential_results,
        "parallel_results": parallel_results,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }