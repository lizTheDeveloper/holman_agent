from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled
import os
import asyncio
from typing_extensions import TypedDict
from openai import AsyncOpenAI
import gradio as gr
import importlib.util
import sys
from types import ModuleType
from typing import Callable, List
import glob
import uuid
import inspect


client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
if not os.path.exists(TOOLS_DIR):
    os.makedirs(TOOLS_DIR)

def extract_original_prompt_from_code(code: str) -> str:
    """
    Extract the original prompt from a docstring with the marker 'ORIGINAL_PROMPT:'.
    Returns the prompt string or None if not found.
    """
    import re
    docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if docstring_match:
        docstring = docstring_match.group(1)
        marker = 'ORIGINAL_PROMPT:'
        if marker in docstring:
            # Extract everything after the marker
            return docstring.split(marker, 1)[1].strip()
    return None

def patch_function_tool_import(code: str) -> str:
    """
    Patch the import of function_tool in the tool code to use the correct import statement.
    """
    import re
    # Remove any import of function_tool from function_tools or tools
    code = re.sub(r'from\s+function_tools\s+import\s+function_tool', '', code)
    code = re.sub(r'from\s+tools\s+import\s+function_tool', '', code)
    # Ensure correct import at the top
    if 'from agents import function_tool' not in code:
        code = 'from agents import function_tool\n' + code
    return code

def load_tools_from_directory(directory: str) -> List[Callable]:
    """
    Load all @function_tool-decorated callables from .py files in the given directory.
    If a file fails to load due to a syntax error, try to regenerate it from the original prompt in the docstring.
    If regeneration fails, move it to .broken and log the error.
    When regenerating, provide the agent with the error message and the original code for debugging.
    Patch broken imports before retrying.
    Returns a list of tool callables.
    """
    tools = []
    for py_file in glob.glob(os.path.join(directory, "*.py")):
        try:
            module_name = f"tool_{os.path.splitext(os.path.basename(py_file))[0]}"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            module.__dict__["function_tool"] = function_tool
            with open(py_file, "r") as f:
                code = f.read()
            exec(code, module.__dict__)
            for v in module.__dict__.values():
                if callable(v) and hasattr(v, "_function_tool"):
                    tools.append(v)
        except Exception as e:
            # Try to regenerate from docstring prompt
            try:
                with open(py_file, "r") as f:
                    code = f.read()
                prompt = extract_original_prompt_from_code(code)
                if prompt:
                    # Patch broken imports before debugging
                    patched_prompt = patch_function_tool_import(prompt)
                    # Use the tool_fixer tool to fix the code
                    error_message = str(e)
                    fixed_code = tool_fixer(error_message, patched_prompt)
                    fixed_code = patch_function_tool_import(fixed_code)
                    with open(py_file, "w") as f:
                        f.write(f'"""ORIGINAL_PROMPT:\n{patched_prompt}\n"""\n' + fixed_code)
                    print(f"Regenerated tool {py_file} using tool_fixer. Retrying load...")
                    # Try loading again
                    try:
                        module_name = f"tool_{os.path.splitext(os.path.basename(py_file))[0]}"
                        spec = importlib.util.spec_from_file_location(module_name, py_file)
                        module = importlib.util.module_from_spec(spec)
                        module.__dict__["function_tool"] = function_tool
                        with open(py_file, "r") as f2:
                            code2 = f2.read()
                        exec(code2, module.__dict__)
                        for v in module.__dict__.values():
                            if callable(v) and hasattr(v, "_function_tool"):
                                tools.append(v)
                        continue  # Success, skip to next file
                    except Exception as regen_e:
                        print(f"Regeneration failed for {py_file}: {regen_e}")
                # If no prompt or regeneration failed, move to .broken
                broken_path = py_file + ".broken"
                os.rename(py_file, broken_path)
                print(f"Error loading tool from {py_file}: {e}. Moved to {broken_path}")
            except Exception as move_err:
                print(f"Error loading tool from {py_file}: {e}. Also failed to move/regenerate: {move_err}")
    return tools

# Load all tools from the tools directory at startup
persisted_tools = load_tools_from_directory(TOOLS_DIR)



class SelfModifyingAgent:
    """
    An agent that can equip itself with new tools and restart itself.
    """
    def __init__(self, name: str, instructions: str, tools: List[Callable], model: str):
        self.name = name
        self.instructions = instructions
        self.tools = tools.copy() + persisted_tools
        self.model = model
        self._restart_agent()

    def _restart_agent(self):
        """Re-instantiate the underlying Agent with the current tools."""
        self.agent = Agent(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model
        )

    def add_tool(self, tool_func: Callable):
        """Add a new tool and restart the agent."""
        self.tools.append(tool_func)
        self._restart_agent()

    def get_agent(self):
        return self.agent




@function_tool
def add_tool_and_restart(tool_code: str, tool_name: str = None):
    """
    Dynamically add a new tool to the agent, save it to the tools directory, and restart the agent.
    tool_code: Python code defining an async function decorated with @function_tool.
    tool_name: Optional filename for the tool (without .py). If not provided, a unique name is generated.
    The original tool_code is stored in the docstring with the marker 'ORIGINAL_PROMPT:' for future regeneration.
    Returns a success or error message.
    """
    try:
        # Save the tool code to a file for persistence, with the original code in the docstring
        if tool_name is None:
            tool_name = f"tool_{uuid.uuid4().hex[:8]}"
        filename = os.path.join(TOOLS_DIR, f"{tool_name}.py")
        # Insert the original code as a docstring at the top
        tool_code_with_prompt = f'"""ORIGINAL_PROMPT:\n{tool_code}\n"""\n' + tool_code
        with open(filename, "w") as f:
            f.write(tool_code_with_prompt)
        # Load the tool from the file
        new_tools = load_tools_from_directory(TOOLS_DIR)
        # Only add the new tool(s) that are not already present
        added = 0
        for tool in new_tools:
            if tool not in self_mod_agent.tools:
                self_mod_agent.add_tool(tool)
                added += 1
        if added == 0:
            return "No new @function_tool found or tool already exists."
        return f"Added {added} tool(s), saved to {filename}, and restarted the agent."
    except Exception as e:
        return f"Error adding tool: {e}"

# Global instance of the self-modifying agent
self_mod_agent = SelfModifyingAgent(
    name="Self-Modifying Agent",
    instructions="You help the user and can equip yourself with new tools.",
    model="qwen-qwq-32b",
    tools=[
        add_tool_and_restart,  # Initial tool to add new tools
        *persisted_tools
    ]
)

# Add the tool-adder to the agent's toolset
self_mod_agent.add_tool(add_tool_and_restart)

@function_tool
def pip_install_and_save(requirements: str) -> dict:
    """
    Install one or more pip packages and add them to requirements.txt.
    Args:
        requirements: A string of space- or comma-separated package names (optionally with versions).
    Returns:
        dict: status and message.
    """
    import subprocess
    import shlex
    req_list = [r.strip() for r in requirements.replace(',', ' ').split() if r.strip()]
    if not req_list:
        return {'status': 'error', 'message': 'No valid requirements provided.'}
    try:
        # Install each requirement
        for req in req_list:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', req], capture_output=True, text=True)
            if result.returncode != 0:
                return {'status': 'error', 'message': f'Failed to install {req}: {result.stderr}'}
        # Add to requirements.txt
        req_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        req_file = os.path.abspath(req_file)
        # Read existing requirements
        existing = set()
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                for line in f:
                    existing.add(line.strip())
        # Append new ones
        with open(req_file, 'a') as f:
            for req in req_list:
                if req not in existing:
                    f.write(req + '\n')
        return {'status': 'success', 'message': f"Installed and saved: {', '.join(req_list)}"}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

self_mod_agent.add_tool(pip_install_and_save)

@function_tool
def tool_fixer(error_message: str, original_code: str) -> str:
    """
    Receives a Python error message and the original code, and returns a fixed version of the code as a string.
    The returned code must be a valid Python async function decorated with @function_tool.
    """
    # For now, this is a stub. In production, this would call an LLM or use a code-fixing agent.
    # Here, just return the original code for demonstration.
    # Replace this with a call to an LLM or a more advanced fixer if available.
    return original_code

self_mod_agent.add_tool(tool_fixer)

async def gradio_agent_interface(user_input: str) -> str:
    """
    Gradio interface handler for the self-modifying agent.
    Accepts a user question, runs the agent, and returns the output.
    Handles exceptions gracefully.
    """
    try:
        result = await Runner.run(self_mod_agent.get_agent(), user_input)
        return str(result.final_output)
    except Exception as e:
        return f"Error: {e}"


def launch_gradio():
    """
    Launches a simple Gradio interface for the self-modifying agent.
    """
    iface = gr.Interface(
        fn=gradio_agent_interface,
        inputs=gr.Textbox(label="Ask a question:"),
        outputs=gr.Textbox(label="Agent's answer:"),
        title="Self-Modifying Agent",
        description="Ask the agent any question!",
        allow_flagging="never"
    )
    iface.launch(share=True)

# Ensure nest_asyncio is imported at the top if available
try:
    import nest_asyncio
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

# To launch the Gradio UI, uncomment the following line:
launch_gradio()
# Note: Gradio does not support async directly in fn, so you may need to use gr.Interface(fn=lambda x: asyncio.run(gradio_agent_interface(x)), ...) if you encounter issues.

