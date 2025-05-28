from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled
import os
import asyncio
from typing_extensions import TypedDict
from openai import AsyncOpenAI
import gradio as gr
import random


client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

class CustomerRequest(TypedDict):
    name: str
    request: str
    

class Customer(TypedDict):
    name: str
    request: str



@function_tool
def issue_refund_tool(customer: Customer) -> str:
    """
    Simulates issuing a refund for a customer. Prints a confirmation message.
    Accepts extra arguments for compatibility with agent framework.
    """
    print(f"Refund issued to customer: {customer.get('name', 'Unknown')}")
    return "Refund has been issued."

@function_tool
def generate_coupon_code_tool(customer: Customer) -> str:
    """
    Generates a random promotional coupon code and prints it.
    Accepts a customer argument and extra args for compatibility, but does not use them.
    """
    code = f"PROMO-{random.randint(10000, 99999)}"
    print(f"Generated coupon code: {code}")
    return code


RefundAgent = Agent(
    name="Refund Agent",
    handoff_description="Handles customer refund requests.",
    instructions="You handle customer refund requests. Always confirm the refund was issued.",
    tools=[issue_refund_tool],
    model="qwen-qwq-32b"
)

SalesAgent = Agent(
    name="Sales Agent",
    handoff_description="Handles sales and upgrades, offers promotional codes.",
    instructions="You try to convince the user to upgrade by offering a promotional code.",
    tools=[generate_coupon_code_tool],
    model="qwen-qwq-32b"
)

TriageAgent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the first point of contact for customer service. "
        "If the customer's request is about a refund, hand off to the Refund Agent. "
        "Otherwise, hand off to the Sales Agent. "
        "Always explain your reasoning for the handoff."
    ),
    handoffs=[RefundAgent, SalesAgent],
    model="qwen-qwq-32b"
)

async def gradio_agent_interface(user_input: str) -> str:
    """
    Gradio interface handler for the triage agent.
    Accepts a user question, runs the triage agent, and returns the output.
    Handles exceptions gracefully.
    """
    try:
        result = await Runner.run(TriageAgent, user_input)
        return str(result.final_output)
    except Exception as e:
        return f"Error: {e}"


def launch_gradio():
    """
    Launches a simple Gradio interface for the customer service agent.
    """
    iface = gr.Interface(
        fn=gradio_agent_interface,
        inputs=gr.Textbox(label="Ask a customer service question:"),
        outputs=gr.Textbox(label="Agent's answer:"),
        title="Customer Service Agent",
        description="Ask the agent any customer service question!",
        allow_flagging="never"
    )
    iface.launch()

# To launch the Gradio UI, uncomment the following line:
launch_gradio()
# Note: Gradio does not support async directly in fn, so you may need to use gr.Interface(fn=lambda x: asyncio.run(gradio_agent_interface(x)), ...) if you encounter issues.

# async def main():
#     runner = await Runner.run(TriageAgent, "I need a refund for $150. My name is Liz.")

# if __name__ == "__main__":
#     asyncio.run(main())
