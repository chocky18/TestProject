# -----------------------------------------------
#          ENVIRONMENT VARIABLES SETUP
# -----------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -----------------------------------------------
#             EMBEDDING SETUP
# -----------------------------------------------
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model for sentence embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    """
    Generate embedding for the given text using MiniLM model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled.squeeze().tolist()

class CustomHFEmbedding(Embeddings):
    """
    Custom HuggingFace embedding class compatible with LangChain.
    """
    def embed_query(self, text: str) -> list:
        return get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in texts]


# -----------------------------------------------
#           PINECONE VECTOR STORE
# -----------------------------------------------
import pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

index_name = "medigraphai"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

vectorstore = PineconeVectorStore(index=index, embedding=CustomHFEmbedding())


# -----------------------------------------------
#             TOOL: PINECONE RETRIEVER
# -----------------------------------------------
from langchain_core.tools import tool
from langchain.agents import Tool

@tool("PineconeRetriever")
def pinecone_retriever(query: str) -> str:
    """
    Retrieve relevant documents from Pinecone for a given query.
    """
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found in Pinecone."
    return "\n\n".join([
        f"[Result {i+1}]\nText: {doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}"
        for i, doc in enumerate(docs)
    ])

pinecone_tool = Tool(
    name="PineconeRetriever",
    func=pinecone_retriever,
    description="Retrieves relevant healthcare documents from Pinecone DB."
)


# -----------------------------------------------
#               TOOL: WEB SEARCH (TAVILY)
# -----------------------------------------------
from langchain_community.tools.tavily_search import TavilySearchResults

@tool("WebSearch")
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily API for healthcare-related queries.
    """
    results = TavilySearchResults(api_key=TAVILY_API_KEY).run(query)
    return f"[Web Search Results]\n{results}"

tavily_tool = Tool(
    name="WebSearch",
    func=tavily_search,
    description="Search the web for healthcare information using Tavily API."
)


# -----------------------------------------------
#           LLM SETUP: GEMINI FLASH
# -----------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)


# -----------------------------------------------
#               PROMPT TEMPLATE
# -----------------------------------------------
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
You are a helpful medical assistant. Use the following tools:
{tools}

Question: {input}
Thought: your reasoning
Action: [tool name]
Action Input: tool input
Observation: tool result
... (repeat Thought/Action/Observation if needed)
Final Answer: [the final answer to the original question]

Begin!
{agent_scratchpad}
""")


# -----------------------------------------------
#             REACT AGENT + EXECUTOR
# -----------------------------------------------
from langchain.agents import AgentExecutor, create_react_agent

tools = [pinecone_tool, tavily_tool]
RAG_agent = create_react_agent(llm=llm, tools=tools, prompt=custom_prompt)
RAG_agent_executor = AgentExecutor(agent=RAG_agent, tools=tools, verbose=True)

def run_agentic_rag(query: str):
    """
    Run the RAG agent pipeline for a query.
    """
    return RAG_agent_executor.invoke({"input": query})


# -----------------------------------------------
#              TOOL: ECOMMERCE AGENT
# -----------------------------------------------
from browser_use import Agent
import asyncio

@tool("EcommerceAgent")
async def run_dynamic_agent(user_query: str):
    """
    Executes dynamic browser tasks via browser_use.Agent.
    """
    Ecommerce_agent = Agent(task=user_query, llm=llm)
    await Ecommerce_agent.run()

ecommerce_tool = Tool(
    name="EcommerceAgent",
    func=run_dynamic_agent,
    description="A dynamic web task runner (e.g. ecommerce operations)"
)


# -----------------------------------------------
#           TOOL: FINAL ANSWER STRUCTURE
# -----------------------------------------------
@tool("FinalAnswer")
def final_answer(introduction, research_steps, main_body, conclusion, sources):
    """
    Return structured final answer with all sections.
    """
    return {
        "introduction": introduction,
        "research_steps": research_steps if isinstance(research_steps, str) else "\n".join(research_steps),
        "main_body": main_body,
        "conclusion": conclusion,
        "sources": sources if isinstance(sources, str) else "\n".join(sources),
    }


# -----------------------------------------------
#               ORACLE AGENT LOGIC
# -----------------------------------------------
import json
import logging

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Handles query classification through two-stage agent reasoning.
    """
    def __init__(self):
        self.agents = {
            "Agent1": """
                You are Agent1. Classify the query: ["skincare"], ["nutrition"], ["general"], or multiple.
                ```json
                {"predicted_category": ["skincare", "nutrition"]}
                ```""",
            "Agent2": """
                You are Agent2. Validate the prediction and finalize it.
                ```json
                {"final_category": "skincare, nutrition"}
                ```"""
        }

    def make_api_call(self, prompt):
        response = llm(prompt)
        return response.strip() if response else ""

    def process_query(self, user_query):
        agent1_resp = self.make_api_call(f"{self.agents['Agent1']}\nQuery: {user_query}")
        agent2_resp = self.make_api_call(f"{self.agents['Agent2']}\nAgent1 Response: {agent1_resp}")
        try:
            return json.loads(agent2_resp.replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in response."}


# -----------------------------------------------
#              SCRATCHPAD FOR DEBUG
# -----------------------------------------------
from langchain_core.agents import AgentAction

def create_scratchpad(intermediate_steps: list[AgentAction]):
    """
    Create readable scratchpad from intermediate agent steps.
    """
    research_steps = []
    for action in intermediate_steps:
        research_steps.append(f"Tool: {action.tool}, input: {action.tool_input}\nOutput: {action.log}")
    return "\n---\n".join(research_steps)


# -----------------------------------------------
#             TOOL DISPATCH AND GRAPH NODES
# -----------------------------------------------
tool_str_to_func = {
    "PineconeRetriever": pinecone_retriever,
    "WebSearch": tavily_search,
    "EcommerceAgent": run_dynamic_agent,
    "FinalAnswer": final_answer
}

def run_oracle(state: dict):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
    return {"intermediate_steps": [action_out]}

def router(state: dict):
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    print("Router invalid format")
    return "FinalAnswer"

def run_tool(state: dict):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


# -----------------------------------------------
#                  LANGGRAPH SETUP
# -----------------------------------------------
from langgraph.graph import StateGraph, END
from langgraph.graph.state import AgentState

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("oracle", run_oracle)
graph.add_node("PineconeRetriever", run_tool)
graph.add_node("WebSearch", run_tool)
graph.add_node("EcommerceAgent", run_tool)
graph.add_node("FinalAnswer", run_tool)

# Entry point
graph.set_entry_point("oracle")

# Conditional routing
graph.add_conditional_edges("oracle", path=router)

# Routing tools back to oracle
graph.add_edge("PineconeRetriever", "oracle")
graph.add_edge("WebSearch", "oracle")
graph.add_edge("EcommerceAgent", "oracle")

# Final Answer terminates
graph.add_edge("FinalAnswer", END)

# Compile
runnable = graph.compile()

# Visualize the graph
from IPython.display import Image
Image(data=runnable.get_graph().draw_png())
