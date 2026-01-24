import os
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

search_tool = TavilySearchResults(search_depth="basic")

tools = [search_tool]

agent = create_tool_calling_agent(
            model=llm,
            tools=tools,
            system_prompt="You are a helpful assistant",
        )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

def test_agent(query: str):
    
    result = agent_executor.invoke(query)

    return result

if __name__=="__main__":
    query = "Find me twitter post which tells today whether in Bengluru"
    result = test_agent(query=query)
    print(result)