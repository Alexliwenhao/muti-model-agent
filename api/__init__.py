"""REST API Service for Multi-Model Agent

Claude Code API features:
- REST endpoints for agent operations
- Task management API
- WebSocket support for real-time updates
- Authentication support
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import asyncio
import uvicorn


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    """Agent types"""
    CLAUDE = "claude"
    LANGCHAIN = "langchain"
    PLAN = "plan"
    CONVERSATIONAL = "conversational"
    MULTI = "multi"
    WRITER_REVIEWER = "writer_reviewer"


class TaskRequest(BaseModel):
    """Task request model"""
    task: str
    agent_type: Optional[AgentType] = AgentType.CLAUDE
    collaboration_mode: Optional[str] = None


class TaskResponse(BaseModel):
    """Task response model"""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    agent_type: Optional[str] = None


class SystemStatus(BaseModel):
    """System status model"""
    agent_type: str
    available_agents: List[str]
    tools_available: int
    memory_entries: int
    active_tasks: int
    sessions: int


class AgentAPIService:
    """REST API Service for the agent"""
    
    def __init__(self, agent_system: Any, host: str = "0.0.0.0", port: int = 8000):
        self.agent_system = agent_system
        self.host = host
        self.port = port
        self.app = FastAPI(title="Multi-Model Agent API", version="1.0")
        self._setup_routes()
        self._setup_cors()
        
        self.active_tasks: Dict[str, dict] = {}
        self.websocket_connections: List[WebSocket] = []
    
    def _setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        @self.app.post("/api/v1/task", response_model=TaskResponse)
        async def create_task(request: TaskRequest):
            """Create and execute a task"""
            import uuid
            task_id = str(uuid.uuid4())
            
            self.active_tasks[task_id] = {
                "status": "running",
                "task": request.task,
                "agent_type": request.agent_type.value
            }
            
            try:
                result = await self.agent_system.run(
                    request.task,
                    agent_type=request.agent_type.value
                )
                
                self.active_tasks[task_id]["status"] = "completed"
                
                await self._broadcast_update({
                    "type": "task_complete",
                    "task_id": task_id,
                    "status": "completed"
                })
                
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output=str(result.get("output", result)),
                    agent_type=request.agent_type.value
                )
            
            except Exception as e:
                self.active_tasks[task_id]["status"] = "failed"
                
                await self._broadcast_update({
                    "type": "task_complete",
                    "task_id": task_id,
                    "status": "failed"
                })
                
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/task/{task_id}", response_model=TaskResponse)
        async def get_task(task_id: str):
            """Get task status"""
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.active_tasks[task_id]
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus(task["status"]),
                agent_type=task.get("agent_type")
            )
        
        @self.app.get("/api/v1/tasks")
        async def list_tasks():
            """List all active tasks"""
            return [
                {
                    "task_id": tid,
                    "status": t["status"],
                    "task": t["task"],
                    "agent_type": t.get("agent_type")
                }
                for tid, t in self.active_tasks.items()
            ]
        
        @self.app.delete("/api/v1/task/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a task"""
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            del self.active_tasks[task_id]
            return {"status": "cancelled", "task_id": task_id}
        
        @self.app.get("/api/v1/status", response_model=SystemStatus)
        async def get_status():
            """Get system status"""
            status = self.agent_system.get_status()
            return SystemStatus(
                agent_type=status.get("agent_type", "unknown"),
                available_agents=status.get("available_agents", []),
                tools_available=len(status.get("tools_available", [])),
                memory_entries=status.get("memory", {}).get("total_memories", 0),
                active_tasks=status.get("cli_commands", {}).get("active_loops", 0),
                sessions=status.get("sessions", 0)
            )
        
        @self.app.get("/api/v1/agents")
        async def list_agents():
            """List available agents"""
            status = self.agent_system.get_status()
            return {
                "agents": status.get("available_agents", []),
                "default": "claude"
            }
        
        @self.app.get("/api/v1/tools")
        async def list_tools():
            """List available tools"""
            status = self.agent_system.get_status()
            return {"tools": status.get("tools_available", [])}
        
        @self.app.post("/api/v1/memory")
        async def add_memory(content: dict):
            """Add to memory"""
            if not content.get("content"):
                raise HTTPException(status_code=400, detail="Content required")
            
            await self.agent_system.add_memory(
                content=content["content"],
                memory_type=content.get("type", "context"),
                tags=content.get("tags", [])
            )
            
            return {"status": "success", "message": "Memory added"}
        
        @self.app.get("/api/v1/memory")
        async def search_memory(query: str = ""):
            """Search memory"""
            results = await self.agent_system.search_memory(query)
            return {"results": [r.to_dict() for r in results]}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    await websocket.send_text(f"Message received: {data}")
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    async def _broadcast_update(self, message: Dict):
        """Broadcast update to all WebSocket connections"""
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.websocket_connections.remove(ws)
    
    def start(self):
        """Start the API server"""
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    async def start_async(self):
        """Start the API server asynchronously"""
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    from enhanced_agent import MultiModelAgentSystem
    
    agent_system = MultiModelAgentSystem(verbose=False)
    api = AgentAPIService(agent_system)
    api.start()
