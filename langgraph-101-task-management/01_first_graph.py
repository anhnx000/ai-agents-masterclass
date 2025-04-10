import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


# Define our state
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# define first node


def first_node(state: State):
    # create a langchain chat object
    llm = ChatOpenAI(model="gpt-4o-mini")
    # define system message
    system_message = SystemMessage(content="You are a helpful assistant.")
    # define messages
    messages = [system_message] + state["messages"]

    return {"messages": [llm.invoke(messages)]}


# define graph
def create_graph():
    # create the graph
    graph_builder = StateGraph(State)

    # add all the nodes 
    graph_builder.add_node('first_node', first_node)
    
    # add the edges
    graph_builder.add_edge(START, 'first_node')
    graph_builder.add_edge('first_node', END)
    return graph_builder.compile()

def main():
    graph = create_graph()
    
    query = input("type a query: ")
    initial_state = {"messages": [HumanMessage(content=query)]}
    
    for event in graph.stream(initial_state):
        for key in event:
            print("\n")
            print("*" * 10)
            print(key + ":")
            print("+" * 10)
            print(event[key]['messages'][-1].content)
            
if __name__ == "__main__":
    main()
            
