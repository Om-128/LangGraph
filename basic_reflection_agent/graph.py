from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph

from basic_reflection_agent.chains import (
    generation_chain,
    reflection_chain,
    GraphState,
    TweetOutput
)

graph = StateGraph(GraphState)

GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state : GraphState):
    messages = state["messages"]

    result = generation_chain.invoke({"messages": messages})
    
    output = TweetOutput(
        message=result.content,
        user="AI"
    )

    return {
        "messages": messages + 
        [
            HumanMessage(
                content=output.message,
                additional_kwargs={"role":output.user}
            )
        ]
    }

def reflect_node(state: GraphState):
    messages = state["messages"]

    result = reflection_chain.invoke({"messages":messages})

    output = TweetOutput(
        message=result.content,
        user="CRITIQUE"
    )

    return {
        "messages": messages + 
        [
            HumanMessage(
                content=output.message,
                additional_kwargs={"role":output.user}
            )
        ]
    }

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state: GraphState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


graph.add_conditional_edges(
    GENERATE, 
    should_continue,
    {REFLECT : REFLECT, END: END}
)

graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

if __name__=="__main__":
    # print(app.get_graph().draw_mermaid())
    # app.get_graph().print_ascii()
    response = app.invoke({
        "messages":[HumanMessage(content="AI Agents taking over content creation")]
        })

    print(response)
