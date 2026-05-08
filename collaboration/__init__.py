"""Multi-Model Collaboration Engine

多模型协同架构:
1. 模型注册和能力管理
2. 任务分解与分发
3. 并行/顺序/层级协同
4. 结果聚合与综合
5. 动态模型选择
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from config import Config, ModelConfig
from agents import SubAgentManager, SubAgentConfig, SubAgentType


class CollaborationMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    DEBATE = "debate"
    VOTE = "vote"


class TaskComplexity(Enum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4


@dataclass
class ModelCapability:
    model_name: str
    provider: str
    strengths: List[str]
    weaknesses: List[str]
    max_tokens: int
    cost_tier: str


@dataclass
class CollaborationTask:
    task_id: str
    description: str
    required_capabilities: List[str]
    complexity: TaskComplexity
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None


@dataclass
class CollaborationResult:
    task_id: str
    model_name: str
    content: str
    confidence: float
    execution_time: float
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalResult:
    status: str
    primary_result: str
    model_responses: List[CollaborationResult]
    synthesis: Optional[str] = None
    confidence_score: float = 0.0
    execution_time: float = 0.0


class ModelRegistry:
    """模型注册表，管理所有可用模型"""

    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, ModelCapability] = {}
        self._initialize_models()

    def _initialize_models(self):
        """从配置初始化模型"""
        for name, model_config in self.config.get_all_models().items():
            self.register_model(
                model_name=name,
                provider=model_config.provider,
                strengths=model_config.capabilities,
                weaknesses=[],
                max_tokens=model_config.max_tokens,
                cost_tier="medium"
            )

    def register_model(
        self,
        model_name: str,
        provider: str,
        strengths: List[str],
        weaknesses: List[str],
        max_tokens: int,
        cost_tier: str = "medium"
    ):
        """注册新模型"""
        self.models[model_name] = ModelCapability(
            model_name=model_name,
            provider=provider,
            strengths=strengths,
            weaknesses=weaknesses,
            max_tokens=max_tokens,
            cost_tier=cost_tier
        )

    def get_model_for_capability(self, capability: str) -> Optional[str]:
        """根据能力选择最佳模型"""
        best_model = None
        best_score = -1

        for name, model in self.models.items():
            if capability in model.strengths:
                score = len(model.strengths) - model.strengths.index(capability)
                if score > best_score:
                    best_score = score
                    best_model = name

        return best_model

    def get_models_for_task(self, required_capabilities: List[str]) -> List[str]:
        """获取适合任务的多个模型"""
        models = []
        for cap in required_capabilities:
            model = self.get_model_for_capability(cap)
            if model and model not in models:
                models.append(model)
        return models

    def get_all_models(self) -> List[str]:
        """获取所有模型名称"""
        return list(self.models.keys())


class CollaborationEngine:
    """多模型协同引擎

    Claude Code 的多模型协同特点:
    - 多个独立 Claude 实例同时运行
    - 共享任务列表，对等通信
    -  emergent behavior（涌现行为）
    """

    def __init__(
        self,
        config: Config,
        sub_agent_manager: Optional[SubAgentManager] = None
    ):
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.sub_agent_manager = sub_agent_manager or SubAgentManager(config)

        self.active_tasks: Dict[str, CollaborationTask] = {}
        self.task_results: Dict[str, List[CollaborationResult]] = {}

    async def execute_parallel(
        self,
        task: str,
        capabilities: Optional[List[str]] = None,
        models: Optional[List[str]] = None
    ) -> FinalResult:
        """并行协同执行

        多个模型同时处理任务，然后综合结果
        """
        start_time = datetime.now()
        task_id = str(uuid.uuid4())[:8]

        if models is None:
            models = self.model_registry.get_models_for_task(
                capabilities or ["reasoning", "analysis"]
            )

        async def query_model(model_name: str) -> CollaborationResult:
            model_config = self.config.get_model(model_name)
            if not model_config:
                return CollaborationResult(
                    task_id=task_id,
                    model_name=model_name,
                    content="",
                    confidence=0.0,
                    execution_time=0.0,
                    error=f"Model {model_name} not found"
                )

            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_openai import ChatOpenAI

                if model_config.provider == "anthropic":
                    llm = ChatAnthropic(
                        model=model_config.model,
                        anthropic_api_key=model_config.api_key,
                        max_tokens=model_config.max_tokens
                    )
                else:
                    llm = ChatOpenAI(
                        model=model_config.model,
                        api_key=model_config.api_key,
                        base_url=model_config.base_url
                    )

                response = await llm.ainvoke([{"role": "user", "content": task}])
                content = response.content if hasattr(response, "content") else str(response)

                execution_time = (datetime.now() - start_time).total_seconds()

                return CollaborationResult(
                    task_id=task_id,
                    model_name=model_name,
                    content=content,
                    confidence=self._calculate_confidence(content),
                    execution_time=execution_time
                )

            except Exception as e:
                return CollaborationResult(
                    task_id=task_id,
                    model_name=model_name,
                    content="",
                    confidence=0.0,
                    execution_time=0.0,
                    error=str(e)
                )

        results = await asyncio.gather(*[query_model(m) for m in models])
        self.task_results[task_id] = list(results)

        synthesis = await self._synthesize_results(task, results)
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        total_time = (datetime.now() - start_time).total_seconds()

        return FinalResult(
            status="completed",
            primary_result=synthesis or results[0].content if results else "",
            model_responses=list(results),
            synthesis=synthesis,
            confidence_score=avg_confidence,
            execution_time=total_time
        )

    async def execute_sequential(
        self,
        task: str,
        steps: List[str]
    ) -> FinalResult:
        """顺序协同执行

        模型按顺序处理任务的不同阶段
        """
        start_time = datetime.now()
        task_id = str(uuid.uuid4())[:8]
        results = []

        primary_model = self.config.get_primary_model()

        for i, step in enumerate(steps):
            step_prompt = f"{task}\n\nStep {i+1}/{len(steps)}: {step}"

            try:
                from langchain_anthropic import ChatAnthropic
                from langchain_openai import ChatOpenAI

                if primary_model.provider == "anthropic":
                    llm = ChatAnthropic(
                        model=primary_model.model,
                        anthropic_api_key=primary_model.api_key
                    )
                else:
                    llm = ChatOpenAI(
                        model=primary_model.model,
                        api_key=primary_model.api_key
                    )

                response = await llm.ainvoke([{"role": "user", "content": step_prompt}])
                content = response.content if hasattr(response, "content") else str(response)

                results.append(CollaborationResult(
                    task_id=f"{task_id}_{i}",
                    model_name=primary_model.name,
                    content=content,
                    confidence=self._calculate_confidence(content),
                    execution_time=0.0,
                    metadata={"step": i + 1, "description": step}
                ))

            except Exception as e:
                results.append(CollaborationResult(
                    task_id=f"{task_id}_{i}",
                    model_name=primary_model.name,
                    content="",
                    confidence=0.0,
                    execution_time=0.0,
                    error=str(e)
                ))

        synthesis = "\n\n".join([r.content for r in results if r.content])
        total_time = (datetime.now() - start_time).total_seconds()

        return FinalResult(
            status="completed",
            primary_result=synthesis,
            model_responses=results,
            synthesis=synthesis,
            confidence_score=sum(r.confidence for r in results) / len(results) if results else 0,
            execution_time=total_time
        )

    async def execute_hierarchical(
        self,
        task: str,
        levels: int = 3
    ) -> FinalResult:
        """层级协同执行

        多层模型协作，逐层精炼结果
        类似于 Claude Code 的 emergent behavior
        """
        start_time = datetime.now()
        task_id = str(uuid.uuid4())[:8]
        all_results = []

        models = self.model_registry.get_all_models()

        current_content = task
        for level in range(levels):
            if not models:
                break

            model_name = models[level % len(models)]

            async def process_level(content: str, model: str):
                model_config = self.config.get_model(model)
                if not model_config:
                    return None

                try:
                    from langchain_anthropic import ChatAnthropic
                    from langchain_openai import ChatOpenAI

                    if model_config.provider == "anthropic":
                        llm = ChatAnthropic(
                            model=model_config.model,
                            anthropic_api_key=model_config.api_key
                        )
                    else:
                        llm = ChatOpenAI(
                            model=model_config.model,
                            api_key=model_config.api_key
                        )

                    prompt = f"Process and refine this content (Level {level + 1}):\n\n{content}"
                    response = await llm.ainvoke([{"role": "user", "content": prompt}])

                    return CollaborationResult(
                        task_id=f"{task_id}_L{level}",
                        model_name=model,
                        content=response.content if hasattr(response, "content") else str(response),
                        confidence=self._calculate_confidence(response.content if hasattr(response, "content") else ""),
                        execution_time=0.0,
                        metadata={"level": level + 1}
                    )

                except Exception as e:
                    return None

            result = await process_level(current_content, model_name)
            if result:
                all_results.append(result)
                current_content = result.content

        total_time = (datetime.now() - start_time).total_seconds()

        return FinalResult(
            status="completed",
            primary_result=all_results[-1].content if all_results else task,
            model_responses=all_results,
            synthesis=all_results[-1].content if all_results else None,
            confidence_score=sum(r.confidence for r in all_results) / len(all_results) if all_results else 0,
            execution_time=total_time
        )

    async def execute_debate(
        self,
        task: str,
        models: Optional[List[str]] = None
    ) -> FinalResult:
        """辩论协同执行

        多个模型提出不同观点，然后综合
        """
        start_time = datetime.now()
        task_id = str(uuid.uuid4())[:8]

        if models is None:
            models = self.model_registry.get_all_models()[:3]

        async def debate_round(round_num: int) -> List[CollaborationResult]:
            round_results = []

            for model_name in models:
                model_config = self.config.get_model(model_name)
                if not model_config:
                    continue

                context = ""
                if round_num > 0 and self.task_results.get(task_id):
                    previous = self.task_results[task_id]
                    context = "\n\nPrevious arguments:\n" + "\n".join(
                        [f"[{r.model_name}]: {r.content[:200]}" for r in previous]
                    )

                try:
                    from langchain_anthropic import ChatAnthropic
                    from langchain_openai import ChatOpenAI

                    if model_config.provider == "anthropic":
                        llm = ChatAnthropic(
                            model=model_config.model,
                            anthropic_api_key=model_config.api_key
                        )
                    else:
                        llm = ChatOpenAI(
                            model=model_config.model,
                            api_key=model_config.api_key
                        )

                    prompt = f"""Task: {task}
{context}

Round {round_num + 1}: Present your argument or respond to others."""

                    response = await llm.ainvoke([{"role": "user", "content": prompt}])

                    round_results.append(CollaborationResult(
                        task_id=f"{task_id}_R{round_num}",
                        model_name=model_name,
                        content=response.content if hasattr(response, "content") else str(response),
                        confidence=self._calculate_confidence(response.content if hasattr(response, "content") else ""),
                        execution_time=0.0,
                        metadata={"round": round_num + 1}
                    ))

                except Exception as e:
                    round_results.append(CollaborationResult(
                        task_id=f"{task_id}_R{round_num}",
                        model_name=model_name,
                        content="",
                        confidence=0.0,
                        execution_time=0.0,
                        error=str(e)
                    ))

            return round_results

        all_results = []
        for round_num in range(2):
            round_results = await debate_round(round_num)
            all_results.extend(round_results)
            self.task_results[task_id] = round_results

        synthesis = await self._synthesize_results(task, all_results)
        total_time = (datetime.now() - start_time).total_seconds()

        return FinalResult(
            status="completed",
            primary_result=synthesis,
            model_responses=all_results,
            synthesis=synthesis,
            confidence_score=sum(r.confidence for r in all_results) / len(all_results) if all_results else 0,
            execution_time=total_time
        )

    async def execute(
        self,
        task: str,
        mode: CollaborationMode = CollaborationMode.PARALLEL,
        **kwargs
    ) -> FinalResult:
        """统一执行入口"""
        if mode == CollaborationMode.PARALLEL:
            return await self.execute_parallel(task, **kwargs)
        elif mode == CollaborationMode.SEQUENTIAL:
            steps = kwargs.get("steps", [task])
            return await self.execute_sequential(task, steps)
        elif mode == CollaborationMode.HIERARCHICAL:
            levels = kwargs.get("levels", 3)
            return await self.execute_hierarchical(task, levels)
        elif mode == CollaborationMode.DEBATE:
            return await self.execute_debate(task, **kwargs)
        else:
            return await self.execute_parallel(task, **kwargs)

    async def _synthesize_results(
        self,
        original_task: str,
        results: List[CollaborationResult]
    ) -> Optional[str]:
        """综合多个模型的结果"""
        if not results:
            return None

        successful_results = [r for r in results if r.content and not r.error]
        if not successful_results:
            return None

        if len(successful_results) == 1:
            return successful_results[0].content

        synthesis_prompt = f"""Synthesize the following responses into a coherent answer for the task:

Task: {original_task}

Responses:
{chr(10).join([f'[{r.model_name}]:\n{r.content}\n---' for r in successful_results])}

Provide a synthesized answer that combines the best insights from all responses."""

        primary_model = self.config.get_primary_model()

        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_openai import ChatOpenAI

            if primary_model.provider == "anthropic":
                llm = ChatAnthropic(
                    model=primary_model.model,
                    anthropic_api_key=primary_model.api_key
                )
            else:
                llm = ChatOpenAI(
                    model=primary_model.model,
                    api_key=primary_model.api_key
                )

            response = await llm.ainvoke([{"role": "user", "content": synthesis_prompt}])
            return response.content if hasattr(response, "content") else str(response)

        except Exception:
            return successful_results[0].content

    def _calculate_confidence(self, content: str) -> float:
        """计算结果置信度"""
        if not content:
            return 0.0

        length_score = min(len(content) / 1000, 1.0)
        structure_score = 1.0 if content.count('\n') > 3 else 0.5
        detail_score = 1.0 if len(content) > 200 else 0.5

        return (length_score * 0.3 + structure_score * 0.3 + detail_score * 0.4)

    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            "total_models": len(self.model_registry.models),
            "models": {
                name: {
                    "provider": m.provider,
                    "strengths": m.strengths,
                    "cost_tier": m.cost_tier
                }
                for name, m in self.model_registry.models.items()
            },
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results)
        }
