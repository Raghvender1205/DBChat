
import os
import sqlalchemy
from dotenv import load_dotenv, find_dotenv
from typing import Any, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

load_dotenv(find_dotenv())


def get_engine():
    DB_URI = "postgresql+psycopg2://postgres:1234@localhost:5432/chatdb" # TODO: 
    engine = sqlalchemy.create_engine(DB_URI)
    db = SQLDatabase(engine)

    return db

def get_llm():
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model="llama3.1:70b",
        temperature=0,
        cache=False
    )

    return llm

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """Create a ToolNode with a fallback to handle errors and surface them to agent"""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n Please fix your mistakes",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def get_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    return tools

##########
# Tools
##########
@tool
def db_query_tool(query: str) -> str:
    """
    Execute the SQL query against the db and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."

    return result

def get_query_check():
    query_check_system = """You are a SQL expert with a strong attention to detail.
    Double check the PostgreSQL query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check."""
    
    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )
    query_check = query_check_system | llm.bind_tools(
        tools=[db_query_tool], tool_choice="required"
    )

    return query_check

########### 
# Workflow
###########
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# node for first tool cal
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

# workflow
def get_workflow():
    workflow = StateGraph(State)

    return workflow

if __name__ == '__main__':
    db = get_engine()
    llm = get_llm()
    tools = get_tools(db, llm)
    print(tools)