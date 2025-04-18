# ────────────── Imports & Environment Setup ────────────── #
import os
import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain.agents import Tool
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool

# ────────────────────────────────
# Tools Initialization
# ────────────────────────────────

# Tool 2: Tavily Web Search
from langchain_community.tools.tavily_search import TavilySearchResults
@tool(tavily_search)
def tavily_search(query: str) -> str:
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
    results = search_tool.run(query)
    return f"[Web Search Results]\n{results}"

tavily_tool = Tool(
    name="WebSearch",
    func=tavily_search,
    description="Search web if Pinecone fails, Performs a web search for healthcare information using Tavily API and scrapes the top websites."
)

@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

tavily_tool = TavilySearchResults()
tools = [
    Tool.from_function(name="web_search", func=tavily_tool.invoke, description="Useful for answering questions about current events or general knowledge."),
    Tool.from_function(name="final_answer", func=lambda input: input, description="Return the final answer to the user.")
]

# ────────────────────────────────
# Tool Mapping
# ────────────────────────────────
tool_str_to_func = {
    "web_search": tavily_tool,
    "final_answer": lambda input: input
}

# ────────────────────────────────
# Oracle Node (decides what to call)
# ────────────────────────────────
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
oracle = initialize_agent(tools, llm, agent="openai-functions", verbose=True)

def run_oracle(state: dict):
    print("🧠 Running Oracle")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="Oracle decided tool"
    )
    return {
        "input": state["input"],
        "chat_history": state["chat_history"],
        "intermediate_steps": state["intermediate_steps"] + [(action_out, "TBD")]
    }

# ────────────────────────────────
# Router Node
# ────────────────────────────────
def router(state: dict) -> str:
    try:
        last_action, _ = state["intermediate_steps"][-1]
        return last_action.tool
    except Exception as e:
        print(f"⚠️ Router fallback due to: {e}")
        return "final_answer"

# ────────────────────────────────
# Tool Runner
# ────────────────────────────────
def run_tool(state: dict):
    last_action, _ = state["intermediate_steps"][-1]
    tool_name = last_action.tool
    tool_args = last_action.tool_input
    print(f"🔧 Running Tool: {tool_name}({tool_args})")
    
    tool_func = tool_str_to_func[tool_name]
    tool_output = tool_func.invoke(tool_args)

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(tool_output)
    )
    return {
        "input": state["input"],
        "chat_history": state["chat_history"],
        "intermediate_steps": state["intermediate_steps"][:-1] + [(action_out, str(tool_output))]
    }

# ────────────────────────────────
# Report Builder
# ────────────────────────────────
def build_report(output: dict):
    data = output["intermediate_steps"][-1][0].tool_input
    return f"""
REPORT
------
{data}
"""

# ────────────────────────────────
# Graph Setup
# ────────────────────────────────
from typing import TypedDict, List, Tuple

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: List[Tuple[AgentAction, str]]

graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges("oracle", path=router)

graph.add_edge("web_search", "oracle")
graph.add_edge("final_answer", END)

runnable = graph.compile()

# ────────────────────────────────
# Run the Graph
# ────────────────────────────────
out = runnable.invoke({
    "input": "tell me something interesting about dogs",
    "chat_history": [],
    "intermediate_steps": [],
})


print(build_report(out))
