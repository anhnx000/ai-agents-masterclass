import os
from dataclasses import dataclass
from typing import Annotated
import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing_extensions import TypedDict
from langsmith import Client

from utils.tasks import read_tasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify LangSmith configuration
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "task-management-agent")
tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

logger.info(f"LangSmith Configuration:")
logger.info(f"API Key Set: {langsmith_api_key is not None}")
logger.info(f"Project: {langsmith_project}")
logger.info(f"Tracing Enabled: {tracing_enabled}")

# Try to initialize the LangSmith client
try:
    langsmith_client = Client()
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LangSmith client: {e}")
    langsmith_client = None


# Define our state
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

    # Confidential stuff
    userid: str

# Define our PydanticAI agent
@dataclass
class MyDeps:
    userid: str


pydantic_agent = Agent(
    model=OpenAIModel('gpt-4o-mini'),
    system_prompt="You are a helpful AI assistant",
    deps_type=MyDeps
)


@pydantic_agent.tool
def get_tasks(ctx: RunContext[MyDeps]) -> str:
    """
    Get the user's tasks
    """
    logger.info(f"Retrieving tasks for user: {ctx.deps.userid}")
    return read_tasks(ctx.deps.userid)


# Define our agent node
def agent(state: State):
    query = state["messages"][-1].content
    logger.info(f"Processing query: {query}")

    # Invoke the model
    result = pydantic_agent.run_sync(query, deps=MyDeps(userid=state["userid"]))
    logger.info("Agent completed processing")
    return {"messages": [AIMessage(content=result.data)]}

# define our graph 
def create_graph():
    logger.info("Creating graph")
    graph_builder = StateGraph(State)

    # add all the nodes 
    graph_builder.add_node('agent', agent)
    
    # add the edges 
    graph_builder.add_edge(START, 'agent')    
    graph_builder.add_edge('agent', END)
    
    # Compile the graph
    graph = graph_builder.compile()
    logger.info("Graph compiled successfully")
    
    # visualization
    import nest_asyncio
    from IPython.display import Image, display
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

    # nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

    graph.get_graph().draw_mermaid_png(
        curve_style=CurveStyle.LINEAR,
        node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
        wrap_label_n_words=9,
        output_file_path="./graph_agents.png",
        draw_method=MermaidDrawMethod.API,
        background_color="white",
        padding=10,
    )
    
    
    
    return graph


def main():
    # Get project name from environment or use default
    project_name = os.getenv("LANGCHAIN_PROJECT", "task-management-agent")
    logger.info(f"Using LangSmith project: {project_name}")
    
    graph = create_graph()
    
    # Confidential stuff
    userid = "YourTechBud"

    query = input("Your query: ")
    logger.info(f"Received query: {query}")

    initial_state = {"messages": [HumanMessage(content=query)], "userid": userid}

    # Configure tracing with more metadata
    config = {
        "tags": ["demo", "task-management"],
        "metadata": {
            "user_id": userid,
            "app_version": "1.0",
            "environment": "development"
        }
    }
    
    logger.info("Starting graph execution with tracing")
    
    try:
        # Run with tracing enabled
        for event in graph.stream(initial_state, config=config):
            for key in event:
                print("\n*******************************************\n")
                print(key + ":")
                print("---------------------\n")
                print(event[key]["messages"][-1].content)
        
        # Check if tracing was successful
        if langsmith_client:
            try:
                # Get recent runs to verify tracing
                runs = langsmith_client.list_runs(
                    project_name=project_name,
                    limit=1
                )
                run_id = next(runs).id if runs else None
                
                if run_id:
                    trace_url = f"https://smith.langchain.com/projects/{project_name}/runs/{run_id}"
                    logger.info(f"Trace successfully created. View at: {trace_url}")
                    print(f"\nYou can view the trace in LangSmith at: {trace_url}")
                else:
                    logger.warning("No recent runs found in LangSmith")
                    print("\nNo traces found in LangSmith. Please check your configuration.")
            except Exception as e:
                logger.error(f"Error checking LangSmith traces: {e}")
                print(f"\nError checking LangSmith traces: {e}")
        else:
            print("\nLangSmith client initialization failed. Please check your API key and configuration.")
    
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        print(f"Error: {e}")
        print("Check if your OPENAI_API_KEY is set correctly.")
    
if __name__ == "__main__":
    main()



















