#!/bin/bash
# Multi-Model Agent 快速设置脚本

echo "=========================================="
echo "  Multi-Model Agent 快速设置"
echo "=========================================="

# 1. 检查 Python 版本
echo ""
echo "📌 检查 Python 版本..."
python3 --version || echo "请安装 Python 3.8+"

# 2. 创建虚拟环境
echo ""
echo "📌 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "✅ 虚拟环境已存在"
fi

# 3. 激活虚拟环境
echo ""
echo "📌 激活虚拟环境..."
source venv/bin/activate

# 4. 安装依赖
echo ""
echo "📌 安装依赖包..."
pip install -r requirements.txt

# 5. 创建 .env 文件
echo ""
echo "📌 创建配置文件..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ 已创建 .env 文件"
    echo ""
    echo "⚠️  请编辑 .env 文件，填入你的 API Key:"
    echo "   nano .env"
else
    echo "✅ .env 文件已存在"
fi

# 6. 显示配置模板
echo ""
echo "=========================================="
echo "  支持的模型配置"
echo "=========================================="
echo ""
echo "在 .env 文件中配置以下模型:"
echo ""
echo "1️⃣  Anthropic Claude (推荐)"
echo "   ANTHROPIC_API_KEY=sk-ant-xxxxx"
echo ""
echo "2️⃣  OpenAI GPT"
echo "   OPENAI_API_KEY=sk-xxxxx"
echo ""
echo "3️⃣  本地模型 (Ollama)"
echo "   LOCAL_BASE_URL=http://localhost:11434/v1"
echo "   LOCAL_MODEL=deepseek-coder"
echo ""
echo "4️⃣  DeepSeek"
echo "   DEEPSEEK_API_KEY=sk-xxxxx"
echo "   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1"
echo ""
echo "5️⃣  通义千问"
echo "   DASHSCOPE_API_KEY=sk-xxxxx"
echo ""
echo "=========================================="
echo ""
echo "🎯 运行测试:"
echo "   python test_config.py"
echo ""
echo "🎯 运行交互模式:"
echo "   python main.py"
echo ""
echo "🎯 运行单次任务:"
echo "   python main.py \"你的任务\""
echo ""
echo "=========================================="
