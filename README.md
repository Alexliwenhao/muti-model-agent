# Multi-Model Agent

基于 Claude Code 架构的多模型协同 Agent 系统，使用 LangChain 实现。

## 架构特点

- **简单循环**: Claude Code 使用 `while(tool_call)` 循环，模型决定何时调用工具
- **8个核心工具**: Bash, Read, Edit, Write, Grep, Glob, TodoWrite, Task
- **上下文窗口管理**: 200K token 上下文，自动压缩
- **多模型协同**: 支持 Anthropic Claude、OpenAI GPT、本地模型

## 安装

```bash
pip install -r requirements.txt
```

## 配置

设置环境变量：
```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## 使用

```bash
python cli.py
```

## 核心组件

- `core/agent.py`: Agent 核心实现
- `tools/`: 工具系统
- `context/`: 上下文管理
- `config/`: 配置管理
