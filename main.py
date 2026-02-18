from __future__ import annotations
import operator 
from typing import TypedDict, List , Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph , START,END
from langgraph. types import Send

from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage , HumanMessage

import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



class Task(BaseModel):
    id:int
    title:str
    brief: str=Field(...,description="What to cover")

class Plan(BaseModel):
    blog_title : str
    tasks: list[Task]

class State(TypedDict):
    topic:str
    plan:Plan
    section: Annotated[list[str],operator.add]
    final: str

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    temperature = 0.7
)

def orchestrator(state: State)-> dict:
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5-7 section on th following topic"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic:{state['topic']}"
                )
            ),
            
        ]
    )
    return {"plan":plan}


def fanout(state: State):
    return [ Send("worker", 
                  {"task":task,
                   "topic": state["topic"],
                    "plan":state["plan"] })
                  
                for task in state["plan"].tasks]


def worker(payload: dict)->dict:

    task = payload["task"]
    topic= payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clean Markdown section"),
            HumanMessage(
                content=(
                    f"Blog : {blog_title}\n"
                    f"Topic : {topic}\n\n"
                    f"Section : {task.title}\n"
                    f"Brief : {task.brief}\n"
                    "Return only the section content in markdown "
                    
                )
            ),
        ]
    ).content.strip()

    return{"section":[section_md]}



def reducer(state: State)-> dict:
    
    title = state["plan"].blog_title
    body = "\n\n".join(state["section"]).strip()

    final_md = f"#{title}\n\n{body}\n"
   
    safe_title = re.sub(r'[^\w\s-]', '', title) 
    filename = safe_title.lower().replace(" ","-")+".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return{"final": final_md}


g= StateGraph(State)

g.add_node("orchestrator", orchestrator)
g.add_node("worker",worker)
g.add_node("reducer", reducer)


g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout,["worker"])
g.add_edge("worker","reducer")
g.add_edge("reducer",END)



app= g.compile()

output = app.invoke({"topic": "Write a blog on ai as career in indin", "section":[]})

