import os
import sys
from IPython.display import Image, display

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
import operator

class CountState(TypedDict):
    count : int
    addition : Annotated[int, operator.add]
    history : Annotated[List[int], operator.concat]

def increment(state: CountState) -> CountState:

    new_count = state['count'] + 1

    return {
        "count" : new_count,
        "addition" : new_count,
        "history" : [new_count]
    }

def should_continue(state: CountState):
    if state['count'] < 4:
        return "increment"
    else:
        return "END"

graph = StateGraph(CountState)

graph.add_node("increment", increment)
graph.set_entry_point("increment")

graph.add_conditional_edges(
    "increment",
    should_continue,
    {
        "increment": "increment",
        "END" : END
    }
)

app = graph.compile()

state = {
        "count" : 0,
        "addition": 0,
        "history": []
}

data = app.invoke(state)
print(data)


if __name__=="__main__":
    png_bytes = app.get_graph().draw_mermaid_png()

    with open("stategraph.png", "wb") as f:
        f.write(png_bytes)

