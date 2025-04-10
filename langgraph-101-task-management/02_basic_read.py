import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from typing_extensions import TypedDict
# import pytest

from utils.task import read_tasks

load_dotenv()


# Define our state
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

    # Confidential stuff
    userid: str

#define our tool 
@tool
def retrieve_tasks(userid: str) -> str:
    """Reads all tasks"""
    return read_tasks(userid)

tools = [retrieve_tasks]
tool_node = ToolNode(tools=tools) 

# define our agent node 
def agent(state: State) -> State:
    # create a langchain model 
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools) # pass in a list of defined functions with @tool decorator

    # Define a system message 
    system_message = SystemMessage(content="You are a helpful assistant. the user is {state['userid']}")
    
    # define the messages 
    messages = [system_message] + state["messages"]
    
    # invoke the model 
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# define graph 
def create_graph():
    graph_builder = StateGraph(State) 
    
    # add all the node  
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("executor", tool_node)
    
    # add the edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", "executor")
    graph_builder.add_edge("executor", "agent")
    
    return graph_builder.compile()


# def test_retrieve_tasks():
#     # Test with existing user
#     result = retrieve_tasks("YourTechBud")
#     assert "Buy groceries" in result
#     assert "Walk the dog" in result
    
#     # Test with non-existent user
#     result = retrieve_tasks("NonExistentUser")
#     assert result == "No tasks found"

# Our main function
def main():
    graph = create_graph()

    # Confidential stuff
    userid = "YourTechBud"

    query = input("Your query: ")

    initial_state = {"messages": [HumanMessage(content=query)], "userid": userid}

    for event in graph.stream(initial_state):
        for key in event:
            print("\n*******************************************\n")
            print(key + ":")
            print("---------------------\n")
            print(event[key]["messages"][-1].content)


if __name__ == "__main__":
    main()
