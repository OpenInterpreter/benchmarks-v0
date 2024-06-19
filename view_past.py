import ast
import asyncio
import csv
import json
import os
import sys
from hypercorn.asyncio import serve
from hypercorn.config import Config
from typing import List, TypedDict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from constants import RESULTS
from task import LMC, ResultStatus


class TaskPreview(TypedDict):
    task_id: str
    messages: List[LMC]
    status: ResultStatus


app = FastAPI()
templates = Jinja2Templates("templates")
csv.field_size_limit(sys.maxsize)


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    run_files = list(sorted([os.path.splitext(fn)[0] for fn in os.listdir(RESULTS)])[::-1])
    return templates.TemplateResponse(
        request,
        "runs.html.j2",
        {"run_files": run_files})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def specific_run(request: Request, run_id: str):
    """
    run_id should just be the name of the save file WITHOUT the csv on the end.
    """
    tasks: List[TaskPreview] = []
    path = RESULTS / f"{run_id}.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"couldn't find run '{run_id}'!")
    with open(RESULTS / f"{run_id}.csv") as f:
        dr = csv.DictReader(f)
        tasks = [{"task_id": r["task_id"], "messages": r["messages"], "status": r["status"]} for r in dr]  # type: ignore
    n_correct = sum(1 for t in tasks if t["status"] == "correct")
    total = len(tasks)
    return templates.TemplateResponse(
        request,
        "run.html.j2",
        {"tasks": tasks, "run_id": run_id, "n_correct": n_correct, "total": total}
    )


@app.get("/runs/{run_id}/{task_id}", response_class=HTMLResponse)
def specific_task(request: Request, run_id: str, task_id: str):
    run_path = RESULTS / f"{run_id}.csv"
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"when looking for a specific task, couldn't find run '{run_id}'!")
    with open(RESULTS / f"{run_id}.csv") as f:
        dr = csv.DictReader(f)
        for r in dr:
            messages = [m.get("content", "--") for m in ast.literal_eval(r["messages"])]
            task = {
                **r,
                "command": json.dumps(r["command"], indent=2),
                "messages": "\n".join(messages)
            }
            if r["task_id"] == task_id:
                return templates.TemplateResponse(
                    request,
                    "task.html.j2",
                    {"run_id": run_id, "task": task})
    raise HTTPException(status_code=404, detail=f"couldn't find task '{task_id}'!")


asyncio.run(serve(app, Config()))  # type: ignore
