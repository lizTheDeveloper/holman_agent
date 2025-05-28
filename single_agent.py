from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled
import os
import asyncio
from typing_extensions import TypedDict
from openai import AsyncOpenAI
import gradio as gr


client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

class Line(TypedDict):
    y1: int
    y2: int
    x1: int
    x2: int



@function_tool
async def calculate_slope_intercept(line: Line):
    """Calculate the slope and y-intercept of a line given two points."""
    print(f"Calculating slope and y-intercept for points ({line['x1']}, {line['y1']}) and ({line['x2']}, {line['y2']})")
    if line['x1'] == line['x2']:
        raise ValueError("x1 and x2 cannot be the same value (vertical line).")

    slope = (line['y2'] - line['y1']) / (line['x2'] - line['x1'])
    y_intercept = line['y1'] - slope * line['x1']
    return slope, y_intercept


homework_agent = Agent(
    name="Homework Agent",
    instructions="You help the user with homework questions.",
    tools=[calculate_slope_intercept],
    model="qwen-qwq-32b"
)

async def gradio_agent_interface(user_input: str) -> str:
    """
    Gradio interface handler for the homework agent.
    Accepts a user question, runs the agent, and returns the output.
    Handles exceptions gracefully.
    """
    try:
        result = await Runner.run(homework_agent, user_input)
        return str(result.final_output)
    except Exception as e:
        return f"Error: {e}"


def launch_gradio():
    """
    Launches a simple Gradio interface for the homework agent.
    """
    iface = gr.Interface(
        fn=gradio_agent_interface,
        inputs=gr.Textbox(label="Ask a homework question:"),
        outputs=gr.Textbox(label="Agent's answer:"),
        title="Homework Agent",
        description="Ask the agent any homework question!",
        allow_flagging="never"
    )
    iface.launch(share=True)

# To launch the Gradio UI, uncomment the following line:
launch_gradio()
# Note: Gradio does not support async directly in fn, so you may need to use gr.Interface(fn=lambda x: asyncio.run(gradio_agent_interface(x)), ...) if you encounter issues.

