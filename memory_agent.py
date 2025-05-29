from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled
import os
import asyncio
from typing_extensions import TypedDict
from openai import AsyncOpenAI
import gradio as gr
from pydantic import BaseModel
import logfire
import psycopg2
from sentence_transformers import SentenceTransformer
from agents.mcp import MCPServer, MCPServerStdio


client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
selected_model = "qwen-qwq-32b"

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")

logfire.configure()
logfire.instrument_openai_agents()

@function_tool
def query_documents(search_query: str):
    """
    Queries the database for documents related to the search query.
    Args:
        search_query (str): The query string to search for in the documents.
    """
    if not search_query:
        print("No search query provided.")
        return "No search query provided."
    print(f"Searching for documents related to: {search_query}")
    db_connection = psycopg2.connect(
        "postgresql://localhost:5432/holman_rag"
    )
    cursor = db_connection.cursor()

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    search_embeddings = model.encode(search_query)

    cursor.execute(
        "SELECT title, content FROM documents ORDER BY embeddings <=> cast(%s as vector(768)) LIMIT 5",
        (search_embeddings.tolist(),)
    )
    results = cursor.fetchall()
    
    cursor.close()
    db_connection.close()
    
    return "\n".join([f"Title: {title}\nContent: {content[0:500]}" for title, content in results]) if results else "No documents found."

## tool to write a raw sql query to the database
@function_tool
def execute_sql_query(query: str) -> str:
    """
    Executes a raw SQL query against the database.
    Args:
        query (str): The SQL query to execute.
    Returns:
        str: Result of the query execution or an error message.
    """
    if not query:
        return "No SQL query provided."
    
    try:
        db_connection = psycopg2.connect(
            "postgresql://localhost:5432/holman_rag"
        )
        cursor = db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        db_connection.commit()
        cursor.close()
        db_connection.close()
        
        return f"Query executed successfully. Results: {results}"
    except Exception as e:
        return f"Error executing query: {e}"





async def gradio_agent_interface(user_input: str) -> str:
    """
    Gradio interface handler for the memory agent.
    Accepts a user question, runs the agent, and returns the output.
    Handles exceptions gracefully.
    """
    try:
        async with MCPServerStdio(
            name="Graph RAG Server", 
            params={
                "command": "/Users/annhoward/src/holman_agent/.venv/bin/python3.10",
                "args": ["db_mcp.py"],
            },
        ) as server:
            memory_agent = Agent(
                name="Memory Agent",
                instructions="You help the user keep their hard drive organized. You can answer questions about files on the user's hard drive, and you can also write raw SQL queries to the database. You can save Entities, Relationships, and Observations to the database, and then search them. When searching, always use a fuzzy search, and re-search if you don't find anything. Prefer to find something, rather than nothing. Exact string matching is unlikely, so always use case-insensitive and fuzzy matching.",
                tools=[query_documents, execute_sql_query],
                model=selected_model,
                mcp_servers=[server]
            )
            result = await Runner.run(memory_agent, user_input)
            return str(result.final_output)
    except Exception as e:
        return f"Error: {e}"


def launch_gradio():
    """
    Launches a simple Gradio interface for the memory agent.
    """
    iface = gr.Interface(
        fn=gradio_agent_interface,
        inputs=gr.Textbox(label="Ask a question about files on your hard drive:"),
        outputs=gr.Textbox(label="Agent's answer:"),
        title="Memory Agent",
        description="Ask the agent any question about files on your hard drive!",
        allow_flagging="never"
    )
    iface.launch(share=True)

# To launch the Gradio UI, uncomment the following line:
launch_gradio()
# Note: Gradio does not support async directly in fn, so you may need to use gr.Interface(fn=lambda x: asyncio.run(gradio_agent_interface(x)), ...) if you encounter issues.

