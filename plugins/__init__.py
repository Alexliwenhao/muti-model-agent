"""Plugin System for Multi-Model Agent

Claude Code Plugin Architecture:
- Dynamic extension system
- Plugin manifest with metadata
- Hooks for agent lifecycle events
- Type-safe plugin interface
"""
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib
import os
from pathlib import Path
import json
import sys


class PluginStatus(Enum):
    """Plugin status"""
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PluginHook(Enum):
    """Plugin hook points"""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_CALL = "tool_call"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"


@dataclass
class PluginManifest:
    """Plugin manifest definition"""
    name: str
    id: str
    version: str
    description: str
    author: Optional[str] = None
    hooks: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_skills: List[str] = field(default_factory=list)


@dataclass
class PluginInstance:
    """Loaded plugin instance"""
    manifest: PluginManifest
    module: Any
    status: PluginStatus = PluginStatus.LOADED
    error: Optional[str] = None
    loaded_at: datetime = field(default_factory=datetime.now)


class Plugin:
    """Base class for all plugins"""
    
    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest
    
    async def on_agent_start(self, agent: Any) -> None:
        """Called when agent starts"""
        pass
    
    async def on_agent_end(self, agent: Any) -> None:
        """Called when agent ends"""
        pass
    
    async def on_task_start(self, task: str) -> None:
        """Called when task starts"""
        pass
    
    async def on_task_end(self, task: str, result: Any) -> None:
        """Called when task ends"""
        pass
    
    async def on_tool_call(self, tool_name: str, args: Dict) -> None:
        """Called before tool call"""
        pass
    
    async def on_message_received(self, message: str) -> str:
        """Called when message received, can modify message"""
        return message
    
    async def on_message_sent(self, message: str) -> str:
        """Called when message sent, can modify message"""
        return message


class PluginManager:
    """Manages plugins for the agent
    
    Claude Code plugin system features:
    - Dynamic loading from plugins/ directory
    - Manifest-based plugin discovery
    - Lifecycle hooks
    - Tool and skill registration
    """
    
    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = Path(plugins_dir or "plugins")
        self.plugins: Dict[str, PluginInstance] = {}
        self.hooks: Dict[str, List[Plugin]] = {}
        self._initialize_hooks()
    
    def _initialize_hooks(self):
        """Initialize hook registry"""
        for hook in PluginHook:
            self.hooks[hook.value] = []
    
    def load_plugins(self):
        """Load all plugins from plugins directory"""
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            self._create_example_plugin()
            return
        
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            manifest_path = plugin_dir / "plugin.json"
            if not manifest_path.exists():
                continue
            
            try:
                self._load_plugin(plugin_dir, manifest_path)
            except Exception as e:
                print(f"Failed to load plugin {plugin_dir.name}: {e}")
    
    def _load_plugin(self, plugin_dir: Path, manifest_path: Path):
        """Load a single plugin"""
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        manifest = PluginManifest(**manifest_data)
        
        sys.path.insert(0, str(self.plugins_dir))
        try:
            module = importlib.import_module(f"{plugin_dir.name}.main")
            
            if hasattr(module, 'PluginClass'):
                plugin_instance = module.PluginClass(manifest)
                
                self.plugins[manifest.id] = PluginInstance(
                    manifest=manifest,
                    module=plugin_instance,
                    status=PluginStatus.ACTIVE
                )
                
                self._register_hooks(plugin_instance)
                
                print(f"Loaded plugin: {manifest.name} v{manifest.version}")
        finally:
            sys.path.pop(0)
    
    def _register_hooks(self, plugin: Plugin):
        """Register plugin hooks"""
        for hook_name in plugin.manifest.hooks:
            if hook_name in self.hooks:
                self.hooks[hook_name].append(plugin)
    
    def _create_example_plugin(self):
        """Create example plugin structure"""
        example_dir = self.plugins_dir / "example_plugin"
        example_dir.mkdir()
        
        manifest = {
            "name": "Example Plugin",
            "id": "example",
            "version": "1.0.0",
            "description": "Example plugin demonstrating hooks",
            "author": "Multi-Model Agent",
            "hooks": ["agent_start", "task_end"],
            "requires": [],
            "provides_tools": [],
            "provides_skills": []
        }
        
        with open(example_dir / "plugin.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        main_code = '''"""Example Plugin"""
from plugin import Plugin, PluginManifest

class PluginClass(Plugin):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
    
    async def on_agent_start(self, agent):
        print(f"Agent started!")
    
    async def on_task_end(self, task, result):
        print(f"Task completed: {task[:30]}...")
'''
        
        with open(example_dir / "main.py", 'w') as f:
            f.write(main_code)
        
        init_code = '''from .main import PluginClass
'''
        
        with open(example_dir / "__init__.py", 'w') as f:
            f.write(init_code)
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger a hook with arguments"""
        results = []
        
        if hook_name in self.hooks:
            for plugin in self.hooks[hook_name]:
                try:
                    method = getattr(plugin, f"on_{hook_name}", None)
                    if method:
                        result = await method(*args, **kwargs)
                        if result is not None:
                            results.append(result)
                except Exception as e:
                    print(f"Error in plugin {plugin.manifest.name}: {e}")
        
        return results
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInstance]:
        """Get a plugin by ID"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""
        return [
            {
                "id": p.manifest.id,
                "name": p.manifest.name,
                "version": p.manifest.version,
                "status": p.status.value,
                "hooks": p.manifest.hooks,
                "provides_tools": p.manifest.provides_tools
            }
            for p in self.plugins.values()
        ]
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].status = PluginStatus.DISABLED
            return True
        return False
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin"""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].status = PluginStatus.ACTIVE
            return True
        return False
    
    def get_provided_tools(self) -> List[str]:
        """Get all tools provided by plugins"""
        tools = []
        for plugin in self.plugins.values():
            if plugin.status == PluginStatus.ACTIVE:
                tools.extend(plugin.manifest.provides_tools)
        return tools
    
    def get_provided_skills(self) -> List[str]:
        """Get all skills provided by plugins"""
        skills = []
        for plugin in self.plugins.values():
            if plugin.status == PluginStatus.ACTIVE:
                skills.extend(plugin.manifest.provides_skills)
        return skills


def create_plugin_structure(plugin_name: str, output_dir: str = "plugins"):
    """Create a new plugin structure"""
    plugin_dir = Path(output_dir) / plugin_name.lower().replace(' ', '_')
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "name": plugin_name,
        "id": plugin_name.lower().replace(' ', '_'),
        "version": "1.0.0",
        "description": f"{plugin_name} plugin",
        "author": "",
        "hooks": [],
        "requires": [],
        "provides_tools": [],
        "provides_skills": []
    }
    
    with open(plugin_dir / "plugin.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    main_code = f'''\"\"\"{plugin_name} Plugin\"\"\"
from plugin import Plugin, PluginManifest

class PluginClass(Plugin):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
    
    async def on_agent_start(self, agent):
        print(f"{plugin_name} initialized")
'''
    
    with open(plugin_dir / "main.py", 'w') as f:
        f.write(main_code)
    
    with open(plugin_dir / "__init__.py", 'w') as f:
        f.write(f'from .main import PluginClass\n')
    
    return str(plugin_dir)
