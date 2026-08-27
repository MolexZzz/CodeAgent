# 模型选择功能

## 功能概述

现在可以在 CLI 中直接管理和切换模型，无需手动编辑 `.env` 配置文件。支持多服务商（DeepSeek、OpenAI、Anthropic），每个服务商使用独立的 API 凭证。

## 多服务商支持

系统支持以下 API 服务商，每个服务商需要配置独立的凭证：

### DeepSeek
- 环境变量：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`（可选）
- 支持的模型：`deepseek-chat`、`deepseek-reasoner`

### OpenAI
- 环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）
- 支持的模型：`gpt-4o`、`gpt-4o-mini`、`gpt-4-turbo`

### Anthropic
- 环境变量：`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`（可选）
- 支持的模型：`claude-3-5-sonnet-20241022`、`claude-3-5-haiku-20241022`

**重要提示**：只有配置了凭证的服务商的模型才会出现在模型列表中。如果你没有配置某个服务商的 API Key，该服务商的模型将不会显示。

## 新增命令

### 1. `/models` - 列出可用模型

显示所有已知模型，按服务商分组，并标注凭证配置状态：

```
>>> /models
Available models:
DeepSeek (configured)
  * deepseek-chat
    deepseek-reasoner
OpenAI (no credentials)
    gpt-4o
    gpt-4o-mini
    gpt-4-turbo
Anthropic (no credentials)
    claude-3-5-sonnet-20241022
    claude-3-5-haiku-20241022
```

当前使用的模型会用 `*` 标记。

### 2. `/model` - 交互式选择模型

不带参数时会弹出交互式菜单，使用方向键选择模型（只显示已配置凭证的模型）：

```
>>> /model
Current model: deepseek-chat
? Select a model (this will be saved as default): 
❯ deepseek-chat
  deepseek-reasoner
```

选择后会自动保存到 `.env` 文件，作为默认模型。

### 3. `/model <name>` - 直接切换模型

可以直接指定模型名称快速切换：

```
>>> /model deepseek-reasoner
Model switched: deepseek-chat -> deepseek-reasoner (saved as default)
```

如果尝试切换到没有配置凭证的模型，会显示错误提示：

```
>>> /model gpt-4o
Cannot switch to 'gpt-4o': missing credentials for provider OpenAI. 
Set OPENAI_API_KEY in your .env file first.
```

## 配置持久化

- 选择的模型会自动保存到 `.env` 文件中的 `MEMCODE_MODEL` 变量
- 优先级：当前目录的 `.env` > `~/.memcode/.env`
- 下次启动 CLI 时会自动加载保存的默认模型

## 配置示例

在项目根目录或 `~/.memcode/` 下创建 `.env` 文件：

```bash
# DeepSeek 配置
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI 配置（可选）
# OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic 配置（可选）
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# 默认模型
MEMCODE_MODEL=deepseek-chat
```

## 实现细节

- 使用 `MODEL_REGISTRY` 维护模型与服务商的映射关系
- 每个模型定义了其所属的 `provider`、`api_key_env` 和 `base_url_env`
- `AVAILABLE_MODELS` 属性动态过滤，只返回已配置凭证的模型
- 每次 API 请求前，根据当前模型动态解析对应服务商的凭证
- 切换模型时会验证目标模型的凭证是否已配置

## 技术栈

- `questionary>=2.0.0` - 交互式 CLI 菜单
- `python-dotenv>=1.0.0` - 环境变量管理
- `prompt-toolkit>=3.0.0` - 自动补全支持
