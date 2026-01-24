from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

load_dotenv()


class GraphState(TypedDict):
    messages: List[BaseMessage]

class TweetOutput(BaseModel):
    message: str = Field(description="Content of tweet or critique")
    user: str = Field(description="Tell its AI or CRITIQUE")# "AI" or "CRITIQUE"


generatio_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writting excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, responsd with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a vitual twitter influencer grading tweet, Generating critique and recommendation for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality style, etc."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


llm = ChatGroq(model="openai/gpt-oss-120b")
structured_llm = llm.with_structured_output(TweetOutput)

generation_chain = generatio_prompt | llm
reflection_chain = reflection_prompt | llm