

### 一、环境准备

#### 1.1 硬件与系统要求

SGLang支持多种硬件平台：



| 平台                     | 要求                                     |
| ---------------------- | -------------------------------------- |
| **NVIDIA GPU（推荐）**     | CUDA支持，sm80及以上（A10、A100、L4、L40S、H100等） |
| **AMD GPU**            | 需参考专门文档                                |
| **Apple Silicon（M系列）** | 需通过MLX后端运行                             |
| **CPU Only**           | 支持Intel Xeon CPU推理                     |





**推荐使用Linux系统 + NVIDIA GPU**进行生产部署。



#### 1.2 Python环境

SGLang要求 **Python 3.10或更高版本**。建议使用虚拟环境隔离依赖：



```bash
# 使用 conda
conda create -n sglang python=3.11 -y
conda activate sglang

# 或使用 venv
python3.11 -m venv sglang-env
source sglang-env/bin/activate
```







### 二、安装SGLang

SGLang提供多种安装方式，推荐使用 `uv` 进行快速安装。



#### 2.1 方法一：使用 pip/uv 安装（推荐）

```bash
# 升级 pip
pip install --upgrade pip

# 安装 uv（更快的Python包管理器）
pip install uv

# 安装 SGLang（默认使用CUDA 13）
uv pip install --prerelease=allow sglang
```



**如果使用CUDA 12**，需要指定CUDA版本：



```bash
uv pip install --prerelease=allow sglang
uv pip install --force-reinstall torch==2.11.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
```



#### 2.2 方法二：安装最新Nightly版本

如果想使用最新功能和修复：



```bash
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ sglang
```



#### 2.3 方法三：从源码安装

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```



#### 2.4 方法四：Docker安装

```bash
docker run --gpus all -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct
```



#### 2.5 常见安装问题修复

如果遇到 `OSError: CUDA_HOME environment variable is not set`：



```bash
export CUDA_HOME=/usr/local/cuda-<your-cuda-version>
```



或先单独安装FlashInfer：



```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/
```







### 三、启动推理服务

#### 3.1 基础启动命令

使用 `sglang.launch_server` 模块启动服务：



```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```



> **说明**：SGLang会自动从Hugging Face下载并缓存模型。首次启动可能需要几分钟下载模型文件。
>
>



#### 3.2 常用启动参数

| 参数                       | 说明               | 示例                                     |
| ------------------------ | ---------------- | -------------------------------------- |
| `--model-path`           | 模型ID或本地路径        | `Qwen/Qwen2.5-7B-Instruct`             |
| `--host`                 | 绑定地址             | `0.0.0.0`（允许外部访问）                      |
| `--port`                 | 服务端口             | `30000`                                |
| `--tensor-parallel-size` | 张量并行度（多GPU）      | `2`                                    |
| `--max-total-tokens`     | 最大序列长度           | `8192`                                 |
| `--chat-template`        | 自定义对话模板          | 覆盖Hugging Face默认模板                     |
| `--reasoning-parser`     | 推理模型解析器          | `deepseek-r1`、`qwen3`                  |
| `--disable-radix-cache`  | 禁用RadixAttention | 默认启用                                   |
| `--grammar-backend`      | 语法后端             | `xgrammar`（默认）、`outlines`、`llguidance` |
| `--quantization`         | 量化方式             | `awq`、`gptq`等                          |





#### 3.3 启动后验证

服务启动成功后会看到类似输出：



```plain&#x20;text
The server is fired up and ready to roll!
```



API文档地址：



* Swagger UI: <http://localhost:30000/docs>

* ReDoc: <http://localhost:30000/redoc>

* OpenAPI Spec: <http://localhost:30000/openapi.json>

### 四、发送推理请求

SGLang完全兼容OpenAI API，可以使用curl、OpenAI Python客户端或原生API。



#### 4.1 使用 cURL

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```



#### 4.2 使用 OpenAI Python 客户端

```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"  # SGLang不需要API Key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."}
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```



#### 4.3 流式输出

```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    temperature=0.8,
    max_tokens=128,
    stream=True,  # 启用流式
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```



#### 4.4 使用 Python requests

```python
import requests

url = "http://localhost:30000/v1/chat/completions"
data = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.7,
    "max_tokens": 128,
}

response = requests.post(url, json=data)
print(response.json())
```







### 五、高级功能实战

#### 5.1 结构化输出：强制生成JSON

SGLang的xGrammar后端可以**强制模型输出符合JSON Schema的内容**，确保输出100%合规。



**方式一：直接使用JSON Schema**



```python
import openai
import json

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"
)

# 定义JSON Schema
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w\\s]+$"},
        "price": {"type": "number"},
        "category": {"type": "string"},
        "in_stock": {"type": "boolean"}
    },
    "required": ["name", "price", "category"]
}

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Generate a JSON for a wireless headphone product."}
    ],
    temperature=0.1,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_schema",
            "schema": json_schema
        }
    }
)

print(response.choices[0].message.content)
# 输出一定是合法JSON：
# {"name": "Wireless Headphones", "price": 79.99, "category": "electronics", "in_stock": true}
```



**方式二：使用Pydantic定义Schema**



```python
import openai
from pydantic import BaseModel, Field

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"
)

class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")
    country: str = Field(..., description="Country name")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Generate information about the capital of France in JSON format."}
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "capital_info",
            "schema": CapitalInfo.model_json_schema()
        }
    }
)

# 自动验证
capital = CapitalInfo.model_validate_json(response.choices[0].message.content)
print(capital.model_dump_json())
```



#### 5.2 SGLang前端DSL：编程式LLM工作流

SGLang提供了类似Python的DSL，可以像写普通程序一样定义复杂的LLM工作流：



```python
import sglang as sgl

@sgl.function
def multi_turn_dialogue(s, system_prompt, user_questions):
    s += system_prompt
    for i, question in enumerate(user_questions):
        s += f"User: {question}\n"
        s += "Assistant: " + sgl.gen(f"response_{i}", max_tokens=256, temperature=0.7)
        s += "\n"

@sgl.function
def extract_product_json(s, description):
    s += f"Extract product information from this description: {description}\n"
    s += "Return a valid JSON object with fields: name, price, category.\n"
    s += sgl.gen(
        "product_json",
        max_tokens=128,
        temperature=0.1,
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "category": {"type": "string"}
            },
            "required": ["name", "price", "category"]
        }
    )

# 使用运行时执行
# state = sgl.RunState()
# extract_product_json(state, description="Noise-cancelling headphones with Bluetooth 5.0")
# print(state["product_json"])
```



#### 5.3 启用RadixAttention（自动前缀缓存）

RadixAttention默认启用。在多轮对话和RAG场景中，它会自动缓存公共前缀的KV，显著提升吞吐。



如需显式确认启用状态，启动时添加（默认已启用）：



```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-radix-cache
```







### 六、实际场景完整示例：RAG + 结构化输出

以下是一个完整的实际场景示例——**智能客服系统的产品信息抽取**，结合了RAG风格的多轮对话和结构化输出：



```python
#!/usr/bin/env python3
"""
SGLang 实战示例：智能客服产品信息抽取
场景：用户描述产品需求，系统返回结构化的产品JSON
"""

import openai
import json
from typing import Optional

# ============ 配置 ============
SGLANG_URL = "http://127.0.0.1:30000/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 根据实际部署的模型修改

client = openai.Client(
    base_url=SGLANG_URL,
    api_key="None"
)

# ============ 定义JSON Schema ============
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string", "description": "产品名称"},
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "food", "books", "other"],
            "description": "产品类别"
        },
        "estimated_price": {
            "type": "number",
            "description": "预估价格（美元）"
        },
        "key_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "关键特性列表（最多5个）"
        },
        "target_audience": {
            "type": "string",
            "description": "目标用户群体"
        }
    },
    "required": ["product_name", "category", "estimated_price", "key_features"]
}

# ============ 系统Prompt ============
SYSTEM_PROMPT = """You are a product analysis assistant. 
Extract product information from user descriptions and return structured JSON.
Always follow the JSON schema exactly. Be concise and accurate."""

# ============ 核心函数 ============
def extract_product_info(user_description: str) -> Optional[dict]:
    """从用户描述中抽取结构化产品信息"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this product: {user_description}"}
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "product_extraction",
                    "schema": PRODUCT_SCHEMA
                }
            }
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============ 多轮对话场景（演示RadixAttention） ============
def multi_turn_extraction():
    """多轮产品信息抽取（共享System Prompt，RadixAttention自动缓存）"""
    
    # 第一轮
    print("=" * 50)
    print("第一轮对话")
    desc1 = "A smart watch that tracks heart rate, sleep, and has GPS. Battery lasts 5 days. Priced around $250."
    result1 = extract_product_info(desc1)
    print(f"用户: {desc1}")
    print(f"抽取结果: {json.dumps(result1, indent=2, ensure_ascii=False)}")
    
    # 第二轮（System Prompt相同，RadixAttention复用KV Cache）
    print("\n" + "=" * 50)
    print("第二轮对话 (RadixAttention复用前缀缓存)")
    desc2 = "Wireless noise-cancelling headphones with 40-hour battery life and premium sound quality. Around $150."
    result2 = extract_product_info(desc2)
    print(f"用户: {desc2}")
    print(f"抽取结果: {json.dumps(result2, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    # 先确认服务已启动
    print(f"连接到 SGLang 服务: {SGLANG_URL}")
    print(f"使用模型: {MODEL_NAME}")
    print("\n开始产品信息抽取...\n")
    multi_turn_extraction()
```







### 七、部署检查清单

| 步骤 | 检查项      | 命令/方法                                |
| -- | -------- | ------------------------------------ |
| 1  | Python版本 | `python --version` ≥ 3.10            |
| 2  | CUDA环境   | `nvcc --version`                     |
| 3  | 虚拟环境     | conda/venv已激活                        |
| 4  | SGLang安装 | `pip show sglang`                    |
| 5  | 服务启动     | 查看 `The server is fired up` 日志       |
| 6  | API可达    | `curl http://localhost:30000/health` |
| 7  | 模型加载     | 查看Hugging Face缓存目录                   |





### 八、常见问题排查

\*\*Q1：服务启动报 \*\*`CUDA out of memory`



减小模型或调整参数：



```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \  # 使用更小的模型
  --max-total-tokens 4096 \                   # 限制最大序列长度
  --mem-fraction-static 0.7                   # 限制显存使用比例
```



**Q2：结构化输出不生效**



确认使用了正确的 `response_format` 参数，且模型支持该功能。如果使用旧版SGLang，尝试升级到最新版本。



**Q3：多GPU部署**



```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4  # 使用4张GPU
```



**Q4：首次下载模型慢**



设置Hugging Face镜像或提前下载模型到本地：



```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct
```

