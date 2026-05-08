# Multi-Model Agent 配置指南

## 快速开始

### 1. 安装依赖

```bash
cd /workspace/multi_model_agent/langchain_multi_agent
pip install -r requirements.txt
```

### 2. 配置模型

创建 `.env` 文件配置你的模型接口：

```bash
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI
OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o

# 本地模型 (Ollama / vLLM / DeepSeek)
LOCAL_API_KEY=not-needed
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL=deepseek-coder

# Groq (快速推理)
GROQ_API_KEY=gsk_xxxxx
GROQ_MODEL=mixtral-8x7b-32768

# LM Studio
LMSTUDIO_API_KEY=not-needed
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model

# 通义千问
DASHSCOPE_API_KEY=sk-xxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

# 智谱 GLM
ZHIPU_API_KEY=xxxxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4

# 月之暗面 Kimi
MOONSHOT_API_KEY=sk-xxxxx
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=moonshot-v1-8k

# DeepSeek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-coder

# Ollama (本地)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=codellama
```

### 3. 运行测试

```bash
# 测试基本功能
python main.py "Write a hello world in Python"

# 测试特定模型
python main.py --model anthropic "Your task"

# 测试多模型协同
python main.py --collab --mode parallel "Analyze this codebase"

# 交互模式
python main.py
```

## 详细配置

### 配置文件 (`config/models.yaml`)

```yaml
models:
  anthropic:
    provider: anthropic
    model: claaude-3-5-sonnet-20241022
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.7
    max_tokens: 4096
    capabilities:
      - reasoning
      - code_generation
      - analysis
      - creative_writing

  openai:
    provider: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    temperature: 0.7
    max_tokens: 4096
    capabilities:
      - reasoning
      - code_generation
      - analysis
      - creative_writing

  local:
    provider: openai-compatible
    model: deepseek-coder
    base_url: http://localhost:11434/v1
    api_key: not-needed
    temperature: 0.5
    capabilities:
      - code_generation
      - fast_response

collaboration:
  max_concurrent_tasks: 3
  model_selection_strategy: capability_based
  fallback_enabled: true

agent:
  max_iterations: 100
  context_window: 200000
  auto_compact_threshold: 0.75
```

## 支持的模型提供商

| 提供商 | 配置方式 | 模型示例 |
|--------|----------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` | claaude-3-5-sonnet |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o, gpt-4-turbo |
| **Ollama** | `OLLAMA_BASE_URL` | llama3, codellama, mistral |
| **LM Studio** | `LMSTUDIO_BASE_URL` | local models |
| **vLLM** | `VLLM_BASE_URL` | vLLM served models |
| **Groq** | `GROQ_API_KEY` | mixtral, llama3 |
| **通义千问** | `DASHSCOPE_API_KEY` | qwen-plus, qwen-turbo |
| **智谱** | `ZHIPU_API_KEY` | glm-4, glm-4-flash |
| **Kimi** | `MOONSHOT_API_KEY` | moonshot-v1-8k |
| **DeepSeek** | `DEEPSEEK_API_KEY` | deepseek-coder, deepseek-chat |

## CLI 命令

```bash
# 基本使用
python main.py "你的任务"

# 使用不同模型
python main.py --model openai "你的任务"
python main.py --model local "你的任务"

# 多模型协同模式
python main.py --collab --mode parallel "复杂任务"
python main.py --collab --mode hierarchical "分析任务"
python main.py --collab --mode debate "决策任务"

# 使用不同 Agent 类型
python main.py --agent langchain "规划任务"
python main.py --agent plan "分析任务"
python main.py --agent writer_reviewer "实现任务"

# 交互模式
python main.py
```

## 交互模式命令

```
> /help          - 显示帮助
> /status        - 显示系统状态
> /agents        - 列出可用 Agent
> /tools         - 列出可用工具
> /model <name>   - 切换模型
> /loop <sec>    - 循环任务
> /plan          - 计划模式
> /review        - 代码审查
> /compact       - 压缩上下文
> /memory        - 记忆操作
> /skill <name>  - 执行技能
> /exit          - 退出
```

## 调试模式

```bash
# 详细输出
python main.py -v "你的任务"

# 测试 API 连接
python -c "from config import Config; c = Config(); print(c.get_all_models())"

# 启动 API 服务
python -c "from api import AgentAPIService; from enhanced_agent import MultiModelAgentSystem; api = AgentAPIService(MultiModelAgentSystem()); api.start()"
```
