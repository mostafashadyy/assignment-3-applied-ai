# Weather Conversational Agent (Assignment 3)

This project implements a weather-focused conversational system with **three agent types**:
1. **Basic Agent**
2. **Chain of Thought Agent**
3. **Advanced Agent**

It also includes a bonus evaluation mode that compares quality and latency across the three strategies and stores results in `evaluation_results.csv`.

## 1) Setup Instructions

### Prerequisites
- Python 3.10+
- A Groq API key (for OpenAI-compatible chat completion endpoint)
- A WeatherAPI key

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment variables
Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key_here
WEATHER_API_KEY=your_weatherapi_key_here
```

### Run the CLI app
```bash
python conversational_agent.py
```

Then choose one mode:
- `1` = Basic Agent
- `2` = Chain of Thought Agent
- `3` = Advanced Agent
- `4` = Bonus Evaluation (runs all 3, asks for ratings, saves CSV)

### Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

The Streamlit app supports:
- **Normal Assignment mode** (chat with one selected agent)
- **Bonus Evaluation mode** (single query tested across all agents + timing + ratings)

---

## 2) Brief Documentation of the Implementation

### Core files
- `tools.py`
   - `get_current_weather(location)`
   - `get_weather_forecast(location, days)`
   - `calculator(expression)`

- `conversational_agent.py`
   - Defines tool schemas for function calling.
   - Implements:
      - `process_messages(...)` for standard tool loop.
      - `process_messages_advanced(...)` for advanced orchestration.
      - `run_conversation(...)` for Basic/CoT CLI chat.
      - `run_conversation_advanced(...)` for multi-step advanced chat.
   - Adds optional structured JSON output for complex comparison queries.
   - Includes bonus evaluation flow and CSV logging.

- `advanced_tools.py`
   - Safe tool execution wrapper.
   - Parallel execution with `ThreadPoolExecutor`.
   - Sequential vs parallel benchmark helper.

- `streamlit_app.py`
   - UI for both normal chat and bonus evaluation.
   - Displays side-by-side responses, latency, and speedup metrics.
   - Saves user ratings + timings to CSV.

### Agent behavior summary
- **Basic Agent**
   - Uses weather tools only.
   - Best for simple weather questions.

- **Chain of Thought Agent**
   - Uses weather tools + calculator.
   - Prompt encourages step-by-step reasoning and explanation.

- **Advanced Agent**
   - Uses weather tools + calculator with multi-iteration orchestration.
   - Executes multiple tool calls in parallel when possible.
   - Returns structured JSON for complex comparison-style queries.

---

## 3) Example Conversations

> Note: exact responses may vary by model output and live API data.

### A) Basic Agent (simple weather lookup)
**User:** What is the weather in Giza?

**Assistant (example):**
The current weather in Giza is:
- Temperature: 20.3°C (68.5°F)
- Condition: Partly cloudy
- Humidity: 40%
- Wind: 23.4 km/h

### B) Chain of Thought Agent (weather + calculation reasoning)
**User:** Compare today’s temperature in Cairo and Riyadh, then compute the difference in Celsius.

**Assistant (example):**
1. I fetched current weather for Cairo.
2. I fetched current weather for Riyadh.
3. I computed the absolute difference using the calculator tool.

Result: Riyadh is warmer than Cairo by approximately X°C.

### C) Advanced Agent (multi-step + orchestration)
**User:** Compare Cairo, Giza, and Alexandria weather and tell me which is warmest and by how much compared to the coolest city.

**Assistant (example):**
I retrieved weather for all three locations, then compared temperatures.
Warmest city: Cairo (example)
Coolest city: Alexandria (example)
Difference: Y°C

For comparison-style queries, the agent may also produce structured JSON with keys:
- `query_type`
- `locations`
- `summary`
- `tool_calls_used`
- `final_answer`

---

## 4) Analysis: Impact of Reasoning and Orchestration on Quality & Performance

Based on `evaluation_results.csv` in this project:

- In one weather query test (`what is the weateh in giza`):
   - Basic latency: **2.13s**, rating: **3/5**
   - Chain of Thought latency: **0.46s**, rating: **1/5**
   - Advanced latency: **1.17s**, rating: **5/5**

- Parallel vs sequential benchmark in the same run:
   - Sequential tool time: **0.618s**
   - Parallel tool time: **0.199s**
   - Observed speedup: **3.11x**

### Interpretation
- **Quality:**
   - Advanced generally produced the most complete and grounded response because it continued tool usage across multiple steps and integrated results clearly.
   - Basic quality was inconsistent; in some runs it gave useful outputs, in others it produced generic fallback text.
   - Chain of Thought prompting alone did not guarantee better quality; when tool use is weak or skipped, reasoning quality drops.

- **Performance:**
   - For multi-location queries, parallel tool orchestration can significantly reduce total tool-execution latency.
   - Advanced mode may still have higher end-to-end latency than the fastest single-pass mode because it performs extra reasoning/tool iterations, but this often improves answer quality.

Overall, the results show a practical tradeoff: **more orchestration tends to improve response usefulness**, while **parallel execution helps offset latency cost**.

---

## 5) Challenges Encountered and How They Were Addressed

1. **Inconsistent tool invocation by model responses**
    - **Challenge:** Sometimes the model attempted to answer directly without using tools.
    - **Fix:** Tightened system prompts to explicitly restrict allowed tools and set clear behavior guidelines.

2. **Tool-call / tool-choice mismatch edge cases**
    - **Challenge:** API can raise errors when tool-choice settings and model behavior conflict.
    - **Fix:** Added guarded handling in `process_messages(...)` and fallback no-tool response flow.

3. **Complex query handling across multiple steps**
    - **Challenge:** Single tool-call loops were not always enough for comparisons/calculations.
    - **Fix:** Added iterative advanced orchestration with up to 5 rounds and structured-output validation.

4. **Latency for multi-location queries**
    - **Challenge:** Sequential tool calls increase wait time.
    - **Fix:** Implemented parallel execution in `advanced_tools.py` and measured speedup against sequential execution.

5. **Evaluation repeatability**
    - **Challenge:** Hard to compare modes objectively without shared logs.
    - **Fix:** Implemented CSV logging of query, responses, timings, ratings, and speedup metrics.

---

## 6) Output Artifact

- Evaluation logs are stored in: `evaluation_results.csv`
- This file captures quality ratings and timing metrics for comparison between Basic, CoT, and Advanced agents.
