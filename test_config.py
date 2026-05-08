#!/usr/bin/env python3
"""Model Configuration and Testing Script

参考 Claude CLI 配置方式，支持多种模型接口配置和测试。
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def print_header(text: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_model_config(model_name: str, config: dict):
    """打印模型配置"""
    print(f"\n📦 {model_name.upper()}")
    print("-" * 40)
    for key, value in config.items():
        if "key" in key.lower():
            value = f"{value[:8]}..." if value and len(str(value)) > 8 else value
        print(f"  {key}: {value}")


def test_env_vars():
    """测试环境变量配置"""
    print_header("环境变量配置检查")

    env_checks = {
        "ANTHROPIC_API_KEY": "Anthropic Claude",
        "OPENAI_API_KEY": "OpenAI GPT",
        "LOCAL_API_KEY": "Local Model",
        "LOCAL_BASE_URL": "Local Endpoint",
        "GROQ_API_KEY": "Groq",
        "DASHSCOPE_API_KEY": "通义千问",
        "ZHIPU_API_KEY": "智谱 GLM",
        "MOONSHOT_API_KEY": "Kimi",
        "DEEPSEEK_API_KEY": "DeepSeek",
    }

    configured = []
    missing = []

    for var, name in env_checks.items():
        value = os.getenv(var)
        if value:
            configured.append((var, name))
            print(f"✅ {var}: 已配置 ({name})")
        else:
            missing.append((var, name))
            print(f"❌ {var}: 未配置 ({name})")

    return configured, missing


async def test_model_connection(provider: str, config: dict) -> bool:
    """测试模型连接"""
    try:
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=config.get("api_key"))
            response = client.messages.create(
                model=config.get("model", "claude-3-5-sonnet-20241022"),
                max_tokens=100,
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}]
            )
            return bool(response.content)

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=config.get("api_key"))
            response = client.chat.completions.create(
                model=config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}]
            )
            return bool(response.choices)

        elif provider == "openai-compatible":
            from openai import OpenAI
            client = OpenAI(
                api_key=config.get("api_key", "not-needed"),
                base_url=config.get("base_url")
            )
            response = client.chat.completions.create(
                model=config.get("model"),
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}]
            )
            return bool(response.choices)

        return False

    except Exception as e:
        print(f"   错误: {str(e)[:80]}")
        return False


async def run_basic_tests():
    """运行基本测试"""
    print_header("基本功能测试")

    try:
        from enhanced_agent import MultiModelAgentSystem

        print("✅ 成功导入 MultiModelAgentSystem")

        # 创建系统实例
        system = MultiModelAgentSystem(verbose=False)

        # 获取状态
        status = system.get_status()
        print(f"✅ Agent 系统初始化成功")
        print(f"   - 可用 Agent: {', '.join(status.get('available_agents', []))}")
        print(f"   - 工具数量: {len(status.get('tools_available', []))}")

        # 测试简单任务
        print("\n🚀 测试简单任务执行...")
        result = await system.run("Say 'Hello, this is a test!' in one sentence.", agent_type="claude")

        if result.get("status") == "completed":
            print(f"✅ 任务执行成功")
            print(f"   输出: {result.get('output', '')[:100]}")
        else:
            print(f"⚠️ 任务状态: {result.get('status')}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_model():
    """测试多模型"""
    print_header("多模型测试")

    try:
        from config import Config

        config = Config()
        models = config.get_all_models()

        print(f"找到 {len(models)} 个已配置模型:\n")

        for name, model in models.items():
            print(f"📦 {name}:")
            print(f"   Provider: {model.provider}")
            print(f"   Model: {model.model}")
            print(f"   Capabilities: {', '.join(model.capabilities)}")

            api_key = model.api_key
            if api_key and len(str(api_key)) > 10:
                print(f"   API Key: {str(api_key)[:8]}...")
            else:
                print(f"   API Key: {'✅ 已设置' if api_key else '❌ 未设置'}")

            print()

        # 测试连接
        print("🔗 测试模型连接...")
        for name, model in models.items():
            if not model.api_key or model.api_key == "not-needed":
                continue

            print(f"\n测试 {name}...", end=" ")

            config_dict = {
                "provider": model.provider,
                "api_key": model.api_key,
                "model": model.model,
            }

            if model.base_url:
                config_dict["base_url"] = model.base_url

            success = await test_model_connection(model.provider, config_dict)

            if success:
                print("✅ 连接成功!")
            else:
                print("❌ 连接失败")

    except Exception as e:
        print(f"❌ 多模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


async def interactive_mode():
    """交互模式"""
    print_header("交互模式")

    try:
        from enhanced_agent import MultiModelAgentSystem

        system = MultiModelAgentSystem(verbose=True)

        print("\n🎯 输入你的任务，按 Enter 执行，输入 /help 查看命令")
        print("   输入 /exit 退出\n")

        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/exit", "/quit"]:
                    print("\n👋 再见!")
                    break

                elif user_input.lower() == "/help":
                    print("""
命令列表:
  /help          - 显示帮助
  /status        - 系统状态
  /model <name>  - 切换模型 (anthropic, openai, local)
  /agent <type>  - 切换 Agent (claude, langchain, plan, multi)
  /mode <mode>   - 协同模式 (parallel, sequential, hierarchical)
  /exit          - 退出
""")

                elif user_input.lower() == "/status":
                    import json
                    print(json.dumps(system.get_status(), indent=2, default=str))

                elif user_input.lower().startswith("/model "):
                    model_name = user_input[7:].strip()
                    print(f"切换到模型: {model_name}")

                elif user_input.lower().startswith("/agent "):
                    agent_type = user_input[7:].strip()
                    print(f"使用 Agent: {agent_type}")

                else:
                    result = await system.run(user_input, agent_type="claude")
                    print(f"\n结果:\n{result.get('output', result)}\n")

            except KeyboardInterrupt:
                print("\n\n👋 退出")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

    except Exception as e:
        print(f"❌ 交互模式启动失败: {e}")


def main():
    """主函数"""
    print_header("Multi-Model Agent 配置测试")

    # 检查环境变量
    configured, missing = test_env_vars()

    if not configured:
        print("\n⚠️  警告: 没有配置任何模型 API Key!")
        print("\n请在 .env 文件中配置至少一个模型 API Key:")
        print("""
# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx

# OpenAI
OPENAI_API_KEY=sk-xxxxx

# 本地 (Ollama)
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL=deepseek-coder
""")

    # 选择测试模式
    print("\n" + "-" * 40)
    print("选择测试模式:")
    print("  1. 基本功能测试")
    print("  2. 多模型连接测试")
    print("  3. 交互模式")
    print("  4. 全部测试")
    print("  5. 快速开始")
    print("-" * 40)

    choice = input("请输入选择 (1-5): ").strip()

    if choice == "1":
        asyncio.run(run_basic_tests())
    elif choice == "2":
        asyncio.run(test_multi_model())
    elif choice == "3":
        asyncio.run(interactive_mode())
    elif choice == "4":
        asyncio.run(run_basic_tests())
        asyncio.run(test_multi_model())
    elif choice == "5":
        asyncio.run(run_basic_tests())
    else:
        print("无效选择，运行快速测试...")
        asyncio.run(run_basic_tests())


if __name__ == "__main__":
    main()
