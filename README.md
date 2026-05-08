# Multi-Model Agent

基于 Claude Code 架构的多模型协同 Agent 系统，使用 LangChain 实现。

## 🚀 核心特性

### Claude Code 架构
- **简单循环**: `while(tool_call)` 循环，模型决定何时调用工具
- **8个核心工具**: Bash, Read, Edit, Write, Grep, Glob, TodoWrite, Task
- **上下文窗口管理**: 200K token 上下文，自动压缩
- **子代理系统**: 独立上下文，防止污染

### 多模型协同
- **Parallel**: 多模型并行处理
- **Sequential**: 顺序处理
- **Hierarchical**: 层级递进
- **Debate**: 辩论模式

### LangChain 集成
- ReAct Agent
- Plan-and-Execute Agent
- Conversational Agent with Memory

### MCP (Model Context Protocol)
- 支持外部工具连接
- 文件系统访问
- Git 操作
- 持久化记忆

### 高级 CLI 命令
- `/loop` - 循环任务
- `/btw` - 附带问题
- `/plan` - 计划模式
- `/review` - 代码审查
- `/compact` - 上下文压缩
- `/tasks` - 任务管理

### 质量保证
- **Writer/Reviewer 模式**: 实现 + 审查，确保 2-3x 质量提升

## 📦 安装

```bash
pip install -r requirements.txt
```

## ⚙️ 配置

设置环境变量：
```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## 🚀 使用

### 基本用法

```bash
python main.py "Write a hello world program"
```

### 使用协同模式

```bash
python main.py --collab --mode parallel "Analyze this codebase"
python main.py --collab --mode hierarchical "Design a database"
```

### 使用不同 Agent

```bash
python main.py --agent langchain "Plan a web app"
python main.py --agent plan "Design an API"
python main.py --agent writer_reviewer "Implement user auth"
```

### 交互模式

```bash
python main.py

> /help
> /status
> /loop 60 check server status
> /plan design new feature
> /review src/main.py
> /exit
```

## 🏗️ 架构

```
multi_model_agent/
├── core/           # 核心 Agent 实现
├── agents/         # 子代理和 LangChain Agent
├── tools/          # Claude Code 风格工具
├── collaboration/  # 多模型协同引擎
├── mcp/           # MCP 协议支持
├── memory/        # 持久化记忆系统
├── config/        # 配置管理
└── main.py        # CLI 入口
```

## 🎯 Agent 类型

| Agent | 说明 |
|-------|------|
| `claude` | Claude Code 风格核心 Agent |
| `langchain` | LangChain ReAct Agent |
| `plan` | Plan-and-Execute Agent |
| `conversational` | 带记忆的对话 Agent |
| `multi` | 多模型协同 |
| `writer_reviewer` | Writer/Reviewer 质量模式 |

## 🔧 高级功能

### MCP 服务器配置

创建 `.mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    }
  }
}
```

### CLAUDE.md

创建 `CLAUDE.md` 文件来定义项目规范和记忆。

## 📚 参考

- [Claude Code Architecture](https://github.com/flying-coyote/claude-code-project-best-practices)
- [LangChain Documentation](https://docs.langchain.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
