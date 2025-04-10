import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph

from basic_read import State, create_graph, retrieve_tasks, agent

# Test State class
def test_state_structure():
    state = State(messages=[], userid="test_user")
    assert "messages" in state
    assert "userid" in state
    assert state["userid"] == "test_user"
    assert state["messages"] == []

# Test retrieve_tasks tool
@patch('basic_read.read_tasks')
def test_retrieve_tasks(mock_read_tasks):
    # Setup mock
    mock_read_tasks.return_value = "Task 1, Task 2"
    
    # Test the function
    result = retrieve_tasks("test_user")
    
    # Verify
    assert result == "Task 1, Task 2"
    mock_read_tasks.assert_called_once_with("test_user")

# Test agent function
@patch('basic_read.ChatOpenAI')
def test_agent(mock_chat):
    # Setup mock
    mock_llm = Mock()
    mock_chat.return_value.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Test response")
    
    # Create test state
    test_state = {
        "messages": [HumanMessage(content="Test query")],
        "userid": "test_user"
    }
    
    # Test the function
    result = agent(test_state)
    
    # Verify
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Test response"

# Test graph creation
def test_create_graph():
    graph = create_graph()
    assert isinstance(graph, StateGraph)

# Test full workflow integration
@patch('basic_read.ChatOpenAI')
@patch('basic_read.read_tasks')
def test_workflow_integration(mock_read_tasks, mock_chat):
    # Setup mocks
    mock_read_tasks.return_value = "Task 1, Task 2"
    mock_llm = Mock()
    mock_chat.return_value.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Test response")
    
    # Create graph
    graph = create_graph()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Show my tasks")],
        "userid": "test_user"
    }
    
    # Test graph execution
    events = list(graph.stream(initial_state))
    
    # Verify events were generated
    assert len(events) > 0
    # Verify each event has messages
    for event in events:
        assert any("messages" in value for value in event.values())

# Test error handling in retrieve_tasks
@patch('basic_read.read_tasks')
def test_retrieve_tasks_error(mock_read_tasks):
    # Setup mock to raise an exception
    mock_read_tasks.side_effect = Exception("Database error")
    
    # Test the function
    with pytest.raises(Exception) as exc_info:
        retrieve_tasks("test_user")
    
    assert str(exc_info.value) == "Database error"

# Test agent with empty messages
@patch('basic_read.ChatOpenAI')
def test_agent_empty_messages(mock_chat):
    # Setup mock
    mock_llm = Mock()
    mock_chat.return_value.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Empty query response")
    
    # Create test state with empty messages
    test_state = {
        "messages": [],
        "userid": "test_user"
    }
    
    # Test the function
    result = agent(test_state)
    
    # Verify
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)

# Test graph with invalid state
def test_graph_invalid_state():
    graph = create_graph()
    
    # Create invalid state (missing required fields)
    invalid_state = {
        "messages": [HumanMessage(content="Test query")]
        # Missing userid
    }
    
    # Test graph execution with invalid state
    with pytest.raises(Exception):
        list(graph.stream(invalid_state))

# Test system message formatting
@patch('basic_read.ChatOpenAI')
def test_agent_system_message(mock_chat):
    # Setup mock
    mock_llm = Mock()
    mock_chat.return_value.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Test response")
    
    # Create test state
    test_state = {
        "messages": [HumanMessage(content="Test query")],
        "userid": "test_user"
    }
    
    # Test the function
    result = agent(test_state)
    
    # Verify system message was included
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    assert any(isinstance(msg, SystemMessage) for msg in messages)
    assert any("test_user" in msg.content for msg in messages if isinstance(msg, SystemMessage)) 
