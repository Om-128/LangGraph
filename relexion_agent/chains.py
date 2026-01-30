import os
import sys
from custom_exception import CustomException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from relexion_agent.schema import AnswerQuestion, ReviseAnswer
from langchain.messages import HumanMessage

load_dotenv()

try:
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
        tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    
    revise_instructions = """Revise your previous answer using the new information.

- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

    revisor_chain = actor_prompt_template.partial(
        first_instruction=revise_instructions
    ) | llm.bind_tools(
        tools=[ReviseAnswer], tool_choice="ReviseAnswer")

    response = first_responder_chain.invoke({
        "messages":[HumanMessage("What is RAG?")]
        })
except Exception as e:
    raise CustomException(e, sys)


if __name__=="__main__":
    tool_call = response.tool_calls[0]["args"]
    # parsed_output = AnswerQuestion(**tool_call["args"])
    print(tool_call)