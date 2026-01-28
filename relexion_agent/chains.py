from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from relexion_agent.schema import AnswerQuestion

load_dotenv()
parser = PydanticOutputParser(pydantic_object=AnswerQuestion)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a expert AI researcher
Current time : {time}

1. {first_instruction}
2. Reflect and critique your answer, Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries seperately** for
researching improvements, Do not include them inside the reflection
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time = lambda: datetime.now(),
)

llm = ChatGroq(model="openai/gpt-oss-120b")

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~200 words answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion" 
)
query = "What is RAG?"
response = first_responder_chain.invoke({"messages":[query]})


if __name__=="__main__":
    print(response.tool_calls[0]["args"])