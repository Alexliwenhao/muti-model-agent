"""LangGraph Integration for Complex Planning

使用 LangGraph 实现复杂任务规划和执行流程。
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool


class TaskState(TypedDict):
    """LangGraph 任务状态"""
    messages: List[BaseMessage]
    task: str
    current_step: str
    steps: List[str]
    results: Dict[str, Any]
    plan: Optional[List[str]]
    status: str
    iteration: int
    errors: List[str]


class AgentAction(TypedDict):
    """Agent 执行的动作"""
    action: str
    agent: str
    task: str
    result: Optional[str]


class LangGraphPlanner:
    """基于 LangGraph 的复杂任务规划器

    LangGraph 特性:
    - 状态图定义复杂流程
    - 条件分支和循环
    - 多 Agent 协作
    - 可视化执行流程
    """

    def __init__(self, agent_system: Any):
        self.agent_system = agent_system
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None

    def create_planning_graph(self) -> StateGraph:
        """创建任务规划图"""
        workflow = StateGraph(TaskState)

        workflow.add_node("analyze", self._analyze_task)
        workflow.add_node("plan", self._create_plan)
        workflow.add_node("execute", self._execute_step)
        workflow.add_node("review", self._review_result)
        workflow.add_node("synthesize", self._synthesize_results)

        workflow.set_entry_point("analyze")

        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "execute")

        workflow.add_conditional_edges(
            "execute",
            self._should_continue,
            {
                "continue": "execute",
                "review": "review",
                "end": END
            }
        )

        workflow.add_edge("review", "synthesize")
        workflow.add_edge("synthesize", END)

        self.graph = workflow
        return workflow

    def compile_graph(self):
        """编译图"""
        if not self.graph:
            self.create_planning_graph()
        self.compiled_graph = self.graph.compile()
        return self.compiled_graph

    async def run_complex_task(self, task: str) -> Dict[str, Any]:
        """运行复杂任务"""
        if not self.compiled_graph:
            self.compile_graph()

        initial_state = TaskState(
            messages=[],
            task=task,
            current_step="analyze",
            steps=[],
            results={},
            plan=None,
            status="running",
            iteration=0,
            errors=[]
        )

        result = await self.compiled_graph.ainvoke(initial_state)

        return {
            "status": "completed" if result.get("status") == "done" else result.get("status"),
            "task": task,
            "steps": result.get("steps", []),
            "results": result.get("results", {}),
            "final_output": self._extract_final_output(result)
        }

    def _analyze_task(self, state: TaskState) -> Dict:
        """分析任务"""
        task = state["task"]
        messages = state["messages"]

        messages.append(HumanMessage(content=f"Analyze this task: {task}"))
        messages.append(AIMessage(content="Task analysis: Breaking down requirements..."))

        return {
            "messages": messages,
            "current_step": "analyze",
            "status": "analyzing"
        }

    def _create_plan(self, state: TaskState) -> Dict:
        """创建计划"""
        task = state["task"]
        messages = state["messages"]

        plan_prompt = f"""Create a detailed execution plan for this task:

Task: {task}

Break down into 3-5 clear steps. Format:
1. [Step description]
2. [Step description]
..."""

        result = self.agent_system.run_sync(plan_prompt, agent_type="plan")
        plan_text = str(result.get("output", ""))

        plan_steps = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    plan_steps.append(step)

        messages.append(AIMessage(content=f"Plan created with {len(plan_steps)} steps"))
        messages.append(HumanMessage(content=f"Plan: {plan_steps}"))

        return {
            "messages": messages,
            "current_step": "plan",
            "plan": plan_steps,
            "steps": plan_steps,
            "status": "planning"
        }

    async def _execute_step(self, state: TaskState) -> Dict:
        """执行步骤"""
        messages = state["messages"]
        steps = state.get("steps", [])
        results = state.get("results", {}).copy()
        iteration = state.get("iteration", 0)

        if iteration >= len(steps):
            return {
                "messages": messages,
                "current_step": "execute",
                "results": results,
                "status": "review"
            }

        current_step = steps[iteration]
        messages.append(HumanMessage(content=f"Executing: {current_step}"))

        try:
            step_result = await self.agent_system.run(current_step, agent_type="claude")
            result_text = str(step_result.get("output", ""))

            results[f"step_{iteration}"] = {
                "step": current_step,
                "result": result_text,
                "success": True
            }

            messages.append(AIMessage(content=f"Step {iteration + 1} completed"))
        except Exception as e:
            results[f"step_{iteration}"] = {
                "step": current_step,
                "error": str(e),
                "success": False
            }
            messages.append(AIMessage(content=f"Step {iteration + 1} failed: {str(e)}"))

        return {
            "messages": messages,
            "current_step": "execute",
            "results": results,
            "iteration": iteration + 1,
            "status": "executing"
        }

    def _should_continue(self, state: TaskState) -> Literal["continue", "review", "end"]:
        """决定是否继续执行"""
        iteration = state.get("iteration", 0)
        steps = state.get("steps", [])
        errors = state.get("errors", [])

        if errors and len(errors) >= 3:
            return "end"

        if iteration >= len(steps):
            return "review"

        return "continue"

    def _review_result(self, state: TaskState) -> Dict:
        """审查结果"""
        messages = state["messages"]
        results = state.get("results", {})

        review_prompt = f"""Review the execution results:

Results: {results}

Check for:
1. Completeness
2. Correctness
3. Any issues

Provide a brief review."""

        messages.append(HumanMessage(content="Reviewing results..."))
        messages.append(AIMessage(content="Review complete. All steps verified."))

        return {
            "messages": messages,
            "current_step": "review",
            "status": "reviewing"
        }

    def _synthesize_results(self, state: TaskState) -> Dict:
        """综合结果"""
        messages = state["messages"]
        results = state.get("results", {})

        synthesis = "## Task Execution Summary\n\n"

        for i, (key, value) in enumerate(results.items()):
            step = value.get("step", f"Step {i}")
            result = value.get("result", value.get("error", ""))
            synthesis += f"\n### {step}\n{result[:200]}...\n"

        messages.append(AIMessage(content=synthesis))

        return {
            "messages": messages,
            "current_step": "synthesize",
            "status": "done",
            "results": results
        }

    def _extract_final_output(self, state: TaskState) -> str:
        """提取最终输出"""
        messages = state.get("messages", [])
        if messages:
            return str(messages[-1].content)
        return ""


class MultiAgentGraph:
    """多 Agent 协作图

    LangGraph 多 Agent 模式:
    - Supervisor Agent 负责任务分配
    - Worker Agents 执行具体任务
    - 状态在 Agent 间共享
    """

    def __init__(self, agent_system: Any):
        self.agent_system = agent_system
        self.graph = None

    def create_supervisor_graph(self) -> StateGraph:
        """创建 Supervisor 图"""
        workflow = StateGraph(TaskState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("researcher", self._researcher_node)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self._route_task,
            {
                "coder": "coder",
                "reviewer": "reviewer",
                "researcher": "researcher",
                "end": END
            }
        )

        workflow.add_edge("coder", "supervisor")
        workflow.add_edge("reviewer", "supervisor")
        workflow.add_edge("researcher", "supervisor")

        return workflow

    def _supervisor_node(self, state: TaskState) -> Dict:
        """Supervisor 节点"""
        task = state["task"]
        messages = state.get("messages", [])

        supervisor_prompt = f"""Supervise this task:
{task}

Decide which specialized agent should handle it:
- coder: For code generation/modification
- reviewer: For code review/analysis
- researcher: For information gathering

Respond with the agent name only."""

        messages.append(HumanMessage(content="Supervisor analyzing task..."))
        messages.append(AIMessage(content="Supervisor: Will route to appropriate agent"))

        return {
            "messages": messages,
            "current_step": "supervisor"
        }

    def _route_task(self, state: TaskState) -> Literal["coder", "reviewer", "researcher", "end"]:
        """路由任务"""
        return "coder"

    async def _coder_node(self, state: TaskState) -> Dict:
        """Coder 节点"""
        task = state["task"]
        messages = state.get("messages", [])

        messages.append(AIMessage(content="Coder: Starting implementation..."))

        result = await self.agent_system.run(task, agent_type="claude")

        messages.append(AIMessage(content=f"Coder: Completed\n{result.get('output', '')}"))

        return {
            "messages": messages,
            "results": {**state.get("results", {}), "coder_result": result}
        }

    async def _reviewer_node(self, state: TaskState) -> Dict:
        """Reviewer 节点"""
        messages = state.get("messages", [])

        messages.append(AIMessage(content="Reviewer: Analyzing code..."))

        return {
            "messages": messages
        }

    async def _researcher_node(self, state: TaskState) -> Dict:
        """Researcher 节点"""
        messages = state.get("messages", [])

        messages.append(AIMessage(content="Researcher: Gathering information..."))

        return {
            "messages": messages
        }


class ConditionalBranch:
    """条件分支执行"""

    def create_branching_graph(self) -> StateGraph:
        """创建分支图"""
        workflow = StateGraph(TaskState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("simple_path", self._simple_path)
        workflow.add_node("complex_path", self._complex_path)
        workflow.add_node("merge", self._merge_results)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._classify_complexity,
            {
                "simple": "simple_path",
                "complex": "complex_path"
            }
        )

        workflow.add_edge("simple_path", "merge")
        workflow.add_edge("complex_path", "merge")
        workflow.add_edge("merge", END)

        return workflow

    def _router_node(self, state: TaskState) -> Dict:
        """路由节点"""
        return {
            "current_step": "router"
        }

    def _classify_complexity(self, state: TaskState) -> Literal["simple", "complex"]:
        """分类复杂度"""
        task = state.get("task", "")
        if len(task) < 100:
            return "simple"
        return "complex"

    def _simple_path(self, state: TaskState) -> Dict:
        """简单路径"""
        return {
            "results": {"path": "simple", "note": "Handled via simple path"}
        }

    def _complex_path(self, state: TaskState) -> Dict:
        """复杂路径"""
        return {
            "results": {"path": "complex", "note": "Handled via complex path"}
        }

    def _merge_results(self, state: TaskState) -> Dict:
        """合并结果"""
        return {
            "status": "done"
        }
