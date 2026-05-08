"""Performance Monitoring System

Claude Code 性能监控特性:
- Token 使用追踪
- 执行时间统计
- 工具调用次数
- 内存使用监控
- 性能分析报告
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
import psutil
import gc


class MetricType(Enum):
    """Metric types"""
    TOKENS = "tokens"
    TIME = "time"
    TOOL_CALLS = "tool_calls"
    MEMORY = "memory"
    ERRORS = "errors"


@dataclass
class MetricSample:
    """Single metric sample"""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[str] = None


@dataclass
class PerformanceRecord:
    """Performance record for a task"""
    task_id: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    tokens_used: int = 0
    tool_calls: int = 0
    errors: int = 0
    execution_time: float = 0.0
    metrics: List[MetricSample] = field(default_factory=list)


class PerformanceMonitor:
    """Performance monitoring system
    
    Claude Code 性能监控特点:
    - 实时指标追踪
    - 历史数据分析
    - 性能警报
    - 详细报告生成
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: Dict[str, PerformanceRecord] = {}
        self.metrics: List[MetricSample] = []
        self.start_time = datetime.now()
        self._current_task_id = None
    
    def start_task(self, task_id: str, task_description: str):
        """Start tracking a task"""
        if not self.enabled:
            return
        
        self._current_task_id = task_id
        self.records[task_id] = PerformanceRecord(
            task_id=task_id,
            task_description=task_description,
            start_time=datetime.now()
        )
    
    def end_task(self, task_id: str):
        """End tracking a task"""
        if not self.enabled or task_id not in self.records:
            return
        
        record = self.records[task_id]
        record.end_time = datetime.now()
        record.execution_time = (record.end_time - record.start_time).total_seconds()
        
        if self._current_task_id == task_id:
            self._current_task_id = None
    
    def record_metric(self, metric_type: MetricType, value: float, context: Optional[str] = None):
        """Record a metric"""
        if not self.enabled:
            return
        
        sample = MetricSample(
            metric_type=metric_type,
            value=value,
            context=context
        )
        
        self.metrics.append(sample)
        
        if self._current_task_id and self._current_task_id in self.records:
            self.records[self._current_task_id].metrics.append(sample)
            
            if metric_type == MetricType.TOKENS:
                self.records[self._current_task_id].tokens_used += int(value)
            elif metric_type == MetricType.TOOL_CALLS:
                self.records[self._current_task_id].tool_calls += int(value)
            elif metric_type == MetricType.ERRORS:
                self.records[self._current_task_id].errors += int(value)
    
    def record_tokens(self, tokens: int, context: Optional[str] = None):
        """Record token usage"""
        self.record_metric(MetricType.TOKENS, tokens, context)
    
    def record_tool_call(self, tool_name: str):
        """Record a tool call"""
        self.record_metric(MetricType.TOOL_CALLS, 1, tool_name)
    
    def record_error(self, error_type: str):
        """Record an error"""
        self.record_metric(MetricType.ERRORS, 1, error_type)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_average_execution_time(self) -> float:
        """Get average execution time"""
        completed = [r for r in self.records.values() if r.end_time]
        if not completed:
            return 0.0
        
        return sum(r.execution_time for r in completed) / len(completed)
    
    def get_total_tokens_used(self) -> int:
        """Get total tokens used"""
        return sum(r.tokens_used for r in self.records.values())
    
    def get_total_tool_calls(self) -> int:
        """Get total tool calls"""
        return sum(r.tool_calls for r in self.records.values())
    
    def get_error_rate(self) -> float:
        """Get error rate"""
        total = len(self.records)
        if total == 0:
            return 0.0
        
        errors = sum(r.errors for r in self.records.values())
        return errors / total
    
    def get_task_report(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for a specific task"""
        if task_id not in self.records:
            return None
        
        record = self.records[task_id]
        return {
            "task_id": task_id,
            "description": record.task_description,
            "start_time": record.start_time.isoformat(),
            "end_time": record.end_time.isoformat() if record.end_time else None,
            "execution_time": record.execution_time,
            "tokens_used": record.tokens_used,
            "tool_calls": record.tool_calls,
            "errors": record.errors,
            "metrics": [
                {
                    "type": m.metric_type.value,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "context": m.context
                }
                for m in record.metrics
            ]
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        completed_tasks = [r for r in self.records.values() if r.end_time]
        
        return {
            "system_start": self.start_time.isoformat(),
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "total_tasks": len(self.records),
            "completed_tasks": len(completed_tasks),
            "active_tasks": len(self.records) - len(completed_tasks),
            "average_execution_time": self.get_average_execution_time(),
            "total_tokens_used": self.get_total_tokens_used(),
            "total_tool_calls": self.get_total_tool_calls(),
            "error_rate": self.get_error_rate(),
            "current_memory_usage_mb": self.get_memory_usage(),
            "metrics_count": len(self.metrics)
        }
    
    def get_recent_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get most recent metrics"""
        recent = self.metrics[-count:]
        return [
            {
                "type": m.metric_type.value,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "context": m.context
            }
            for m in recent
        ]
    
    def clear(self):
        """Clear all records"""
        self.records.clear()
        self.metrics.clear()
        self.start_time = datetime.now()


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, context: str = ""):
        self.monitor = monitor
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.monitor.enabled:
            elapsed = time.time() - self.start_time
            self.monitor.record_metric(MetricType.TIME, elapsed, self.context)


class PerformanceDecorator:
    """Decorator for performance tracking"""
    
    def __init__(self, monitor: PerformanceMonitor, name: Optional[str] = None):
        self.monitor = monitor
        self.name = name
    
    def __call__(self, func):
        async def async_wrapper(*args, **kwargs):
            task_name = self.name or func.__name__
            
            self.monitor.start_task(f"task-{id(args)}", task_name)
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.monitor.end_task(f"task-{id(args)}")
        
        def sync_wrapper(*args, **kwargs):
            task_name = self.name or func.__name__
            
            self.monitor.start_task(f"task-{id(args)}", task_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.monitor.end_task(f"task-{id(args)}")
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
