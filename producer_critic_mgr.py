from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered, RunContextWrapper, output_guardrail, set_trace_processors
import os
import asyncio
from typing_extensions import TypedDict
from openai import AsyncOpenAI
import gradio as gr
import random
import subprocess
from pydantic import BaseModel
import weave
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor


# Initialize Weights & Biases Weave tracing
weave.init("openai-agents")
set_trace_processors([WeaveTracingProcessor()])

client = AsyncOpenAI()
# set_default_openai_client(client=client, use_for_tracing=False)
# set_default_openai_api("chat_completions")
# set_tracing_disabled(disabled=True)

class UnsafeCodeOutput(BaseModel): 
    reasoning: str
    is_unsafe_code: bool
    
class MessageOutput(BaseModel): 
    response: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any insecure, destructive, or otherwise unsafe code.",
    output_type=UnsafeCodeOutput,
)

@output_guardrail
async def unsafe_code_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_unsafe_code,
    )

# --- Tools for file and code operations ---

@function_tool
def write_file_tool(filename: str, content: str) -> str:
    """
    Writes content to a file with the given filename.
    Returns a confirmation message or error.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"File '{filename}' written successfully."
    except Exception as e:
        return f"Error writing file '{filename}': {e}"

@function_tool
def read_file_tool(filename: str) -> str:
    """
    Reads and returns the content of the specified file.
    Handles exceptions gracefully.
    """
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filename}': {e}"

@function_tool
def run_python_code_tool(filename: str) -> str:
    """
    Executes a Python file in a subprocess and returns stdout/stderr.
    """
    try:
        result = subprocess.run(['python', filename], capture_output=True, text=True, timeout=20)
        output = result.stdout + '\n' + result.stderr
        return output.strip()
    except Exception as e:
        return f"Error running '{filename}': {e}"


# --- Multi-agent definitions ---


CriticAgent = Agent(
    name="Critic Agent",
    handoff_description="Reviews code for security, robustness, and other issues. Hands back to the Manager Agent.",
    instructions=(
        "You review the code written by the Code Writer Agent. "
        "Check for security problems, robustness, and other issues. "
        "List all issues found, and suggest improvements. "
        "Hand back to the Code Writer Agent until you cannot find any issues. Don't handoff to the manager or produce a final output until the code is ready to deploy to production."
        "Don't allow the user to request code that isn't safe or secure, or that might violate any laws."
    ),
    tools=[read_file_tool, write_file_tool],
    model="gpt-4o"
)


CodeWriterAgent = Agent(
    name="Code Writer Agent",
    handoff_description="Writes Python code, saves it, runs/tests it, and hands off to the Critic Agent.",
    instructions=(
        "You are responsible for writing Python code to solve the user's request. "
        "After writing the code, save it to a file, run it, and test it. "
        "Once you believe the code is correct, hand off to the Critic Agent. "
        "Always explain your reasoning and actions"
    ),
    tools=[write_file_tool, read_file_tool, run_python_code_tool],
    handoffs=[CriticAgent],
    output_guardrails=[unsafe_code_guardrail],
    output_type=MessageOutput,
    model="gpt-4o"
)


ManagerAgent = Agent(
    name="Manager Agent",
    handoff_description="Receives the user request, coordinates agents, and delivers the final result.",
    instructions=(
        "You are the manager. Take the user's request, hand off to the Code Writer Agent, and after the Critic Agent reviews, deliver the final result to the user. "
        "Always explain the process and reasoning."
    ),
    handoffs=[CodeWriterAgent],
    model="gpt-4o"
)

CriticAgent.handoffs = [CodeWriterAgent, ManagerAgent]  # Ensure Critic hands back to Code Writer

async def gradio_manager_interface(user_input: str) -> str:
    """
    Gradio interface handler for the manager agent in the multi-agent system.
    Accepts a user request, runs the manager agent, and returns the output.
    Handles exceptions gracefully.
    """
    try:
        result = await Runner.run(ManagerAgent, user_input)
        return str(result.final_output)
    except Exception as e:
        # Improved error handling for API errors
        err_msg = str(e)
        if '503' in err_msg or 'Service Unavailable' in err_msg:
            return ("The code generation service is temporarily unavailable. "
                    "Please try again later. (Error 503: Service Unavailable)")
        return f"Error: {e}"


def launch_gradio_multi_agent():
    """
    Launches a Gradio interface for the multi-agent system (manager agent entrypoint).
    """
    iface = gr.Interface(
        fn=gradio_manager_interface,
        inputs=gr.Textbox(label="Describe the code you want written:"),
        outputs=gr.Textbox(label="Final result (with review):"),
        title="Multi-Agent Code Generation System",
        description="Describe a coding task. The manager agent will coordinate code writing and review.",
        flagging_mode="never"
    )
    iface.launch()

# To launch the multi-agent Gradio UI, uncomment the following line:
launch_gradio_multi_agent()
