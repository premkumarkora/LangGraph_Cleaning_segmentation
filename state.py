from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the data analytics agentic graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str
    df_path: str  # Path to the current active CSV file
