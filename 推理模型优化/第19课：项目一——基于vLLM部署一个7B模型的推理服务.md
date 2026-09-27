##

### 一、项目介绍

前几课我们学了很多推理优化的理论——KV Cache、PagedAttention、连续批处理、投机解码……但光有理论是不够的。今天我们要把这些技术落地，做一个完整的实战项目：**基于vLLM部署一个7B模型的推理服务**。



为什么选vLLM？因为它是目前大模型推理领域事实标准之一。vLLM的核心优势有三点：**PagedAttention**将KV Cache的显存利用率提升了近**60%**；**连续批处理（Continuous Batching）** 能在请求到达时动态插入，GPU利用率从传统静态批处理的10-30%提升到**90%以上**；再加上它与Hugging Face生态的无缝集成，让模型加载和部署变得极其简单。



7B模型是推理部署的“入门门槛”——它足够大，能体现推理优化的价值；又足够小，单张消费级显卡就能跑。根据实测数据，在7B参数模型推理中，vLLM的吞吐量较传统方案提升**2.8倍**。单卡部署7B模型仅需约**12GB显存**，配合AWQ等量化技术可进一步压缩至**8GB**。



本讲分四个部分。第一部分，我们把环境搭好——安装vLLM、配置CUDA、验证GPU可用。第二部分，加载模型并启动服务——一条命令就能把7B模型变成一个兼容OpenAI API的推理服务。第三部分，做性能压测——用vLLM自带的benchmark工具测量TTFT、TPOT、吞吐量这些关键指标，然后根据结果调优参数。第四部分，实现PD分离架构——把Prefill（计算密集）和Decode（访存密集）拆到不同的实例上，让算力和带宽各尽其用。



这堂课是理论到实践的桥梁——做完之后，你就有一个真正可用的、生产级的推理服务了。







### 二、主要任务安排

**任务1：环境配置与vLLM安装**



在GPU服务器上完成vLLM的安装与验证。要求：



* 创建独立的Conda环境（Python 3.10+）

* 安装与CUDA版本匹配的PyTorch

* 安装vLLM及必要依赖

* 验证vLLM能正常导入、GPU可被识别

**任务2：模型加载与服务启动**



使用vLLM加载一个7B模型（Llama-2-7B、Qwen-7B或DeepSeek-R1-Distill-Qwen-7B），启动OpenAI兼容的API服务。要求：



* 使用`vllm serve`命令启动服务

* 配置合理的`tensor_parallel_size`、`gpu_memory_utilization`等参数

* 通过`curl`或OpenAI Python SDK验证服务可用

**任务3：性能压测与调优**



使用vLLM自带的benchmark工具或Locust对部署的服务进行压测。要求：



* 测量TTFT（首Token延迟）、TPOT（每Token输出延迟）、吞吐量（tokens/s）

* 至少测试2-3种并发度（如1、10、50）

* 根据压测结果调整至少2个服务端参数（如`--max-num-seqs`、`--gpu-memory-utilization`）

* 对比调优前后的性能变化

**任务4（进阶）：实现PD分离架构**



在vLLM 0.8.x及以上版本中，通过KV Transfer机制实现Prefill和Decode的分离部署。要求：



* 启动一个Prefill实例（producer）

* 启动一个Decode实例（consumer）

* 验证请求能正常处理，观察分离前后的性能差异

### 三、任务参考答案

**任务1：环境配置与vLLM安装**



**第一步：创建Conda环境**



```bash
# 创建Python 3.10环境
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
```



**第二步：安装PyTorch**



根据CUDA版本选择对应的PyTorch。以CUDA 12.1为例：



```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```



**第三步：安装vLLM**



```bash
# 安装稳定版
pip install vllm

# 或使用uv加速安装
pip install --upgrade pip
pip install uv
uv pip install -U vllm --torch-backend=auto
```



**第四步：验证安装**



```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```



如果输出vLLM版本号和`True`，说明安装成功。



**任务2：模型加载与服务启动**



**第一步：下载模型**



以Llama-2-7B为例（需要Hugging Face授权）：



```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b
```



也可以使用无需授权的模型，如`mistralai/Mistral-7B-Instruct-v0.1`。



**第二步：启动vLLM服务**



最基础的启动命令：



```bash
vllm serve ./models/llama-2-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code
```



生产环境推荐配置：



```bash
vllm serve ./models/llama-2-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code
```



**关键参数说明**：



| 参数                         | 含义       | 推荐值                         |
| -------------------------- | -------- | --------------------------- |
| `--tensor-parallel-size`   | 张量并行度    | 单卡为1，多卡按卡数设置                |
| `--gpu-memory-utilization` | GPU显存利用率 | 0.85-0.95                   |
| `--max-num-seqs`           | 最大并发序列数  | A100级从64-128起步              |
| `--max-model-len`          | 最大序列长度   | 根据模型和显存调整                   |
| `--dtype`                  | 数据类型     | auto（自动选择）或float16/bfloat16 |





**第三步：验证服务**



```bash
# 使用curl测试
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./models/llama-2-7b",
        "prompt": "解释什么是机器学习",
        "max_tokens": 100
    }'
```



或使用OpenAI Python SDK：



```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.completions.create(
    model="./models/llama-2-7b",
    prompt="解释什么是机器学习",
    max_tokens=100
)
print(response.choices[0].text)
```



**任务3：性能压测与调优**



**第一步：安装benchmark依赖**



```bash
pip install vllm[bench] aiohttp
```



**第二步：运行在线服务吞吐量基准测试**



```bash
# 使用vLLM自带的benchmark工具
vllm bench serve \
    --model ./models/llama-2-7b \
    --base-url http://localhost:8000 \
    --num-prompts 100 \
    --request-rate 10 \
    --max-tokens 256
```



或者使用更灵活的Locust进行压测。



**第三步：解读关键指标**



| 指标       | 含义         | 优化方向               |
| -------- | ---------- | ------------------ |
| **TTFT** | 首Token延迟   | 影响用户体验，Prefill阶段决定 |
| **TPOT** | 每Token输出延迟 | Decode阶段效率，受显存带宽限制 |
| **吞吐量**  | tokens/s   | 整体效率，受批处理影响        |





**第四步：调优实践**



**调优1：调整**`--max-num-seqs`



```bash
# 从默认值提高到128或更高
vllm serve ./models/llama-2-7b --max-num-seqs 128
```



**调优2：调整**`--gpu-memory-utilization`



```bash
# 从0.9提高到0.95
vllm serve ./models/llama-2-7b --gpu-memory-utilization 0.95
```



**调优3：启用前缀缓存**（适用于有共享前缀的场景）



```bash
vllm serve ./models/llama-2-7b --enable-prefix-caching
```



**调优4：关闭日志减少开销**



```bash
vllm serve ./models/llama-2-7b --disable-log-requests
```



**调优前后的对比示例**：



| 配置                              | 吞吐量 (tokens/s) | TTFT (ms) |
| ------------------------------- | -------------- | --------- |
| 默认配置                            | 1200           | 180       |
| `--max-num-seqs=128`            | 1600           | 200       |
| `--gpu-memory-utilization=0.95` | 1700           | 190       |
| 组合优化                            | 1850           | 195       |





**任务4（进阶）：实现PD分离架构**



vLLM从0.8.x版本开始通过KV Transfer机制支持PD分离（1P1D场景）。核心思路是将Prefill（计算密集）和Decode（访存密集）拆分到不同实例上。



**第一步：启动Prefill实例（Producer）**



```python
# prefill_instance.py
from vllm import LLM
from vllm.config import KVTransferConfig

ktc = KVTransferConfig.from_cli(
    '{"kv_connector":"PyNcclConnector", '
    '"kv_role":"kv_producer", '
    '"kv_rank":0, '
    '"kv_parallel_size":2}'
)

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config=ktc,
    tensor_parallel_size=1,
)

# Prefill实例生成KV Cache并传输给Decode实例
outputs = llm.generate(prompts, sampling_params)
```



**第二步：启动Decode实例（Consumer）**



```python
# decode_instance.py
from vllm import LLM
from vllm.config import KVTransferConfig

ktc = KVTransferConfig.from_cli(
    '{"kv_connector":"PyNcclConnector", '
    '"kv_role":"kv_consumer", '
    '"kv_rank":1, '
    '"kv_parallel_size":2}'
)

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config=ktc,
    tensor_parallel_size=1,
)

# Decode实例接收KV Cache并继续生成
outputs = llm.generate(prompts, sampling_params)
```



**PD分离的工作原理**：



Prefill实例处理输入提示（Prompt），一次性生成所有Token的KV Cache。然后通过PyNCCL或Mooncake Store等通信后端，将KV Cache传输给Decode实例。Decode实例基于接收到的KV Cache进行自回归迭代生成输出Token。



**当前局限**：



* 主要支持1P1D（一个Prefill配一个Decode），多实例场景支持有限

* 负载均衡、自动扩缩容等高级功能需要额外组件

### 四、重点知识点和代码的分步骤详细讲解

**知识点1：vLLM的架构与核心优势**



vLLM的架构可以理解为三层：



1. **调度层（Scheduler）** ：管理请求队列，决定哪些请求进入当前批次。连续批处理（Continuous Batching）让调度器可以在每个迭代动态插入新请求，而不是等待整个批次完成。

2. **KV Cache管理层（KV Cache Manager）** ：PagedAttention将KV Cache分页管理，像操作系统管理内存一样管理显存。传统方案中KV Cache是连续分配的，容易产生碎片；PagedAttention将KV Cache分成固定大小的块（Block），按需分配，显存利用率提升近60%。

3) **执行层（Worker）** ：实际执行模型推理的GPU进程。支持张量并行（TP）将模型切分到多卡。

**知识点2：vLLM服务启动的完整参数体系**



```plain&#x20;text
vllm serve <model_path> \
    --host <ip> \                    # 绑定的IP地址
    --port <port> \                  # 监听的端口
    --tensor-parallel-size <N> \     # 张量并行度
    --pipeline-parallel-size <N> \   # 流水线并行度
    --gpu-memory-utilization <0-1> \ # GPU显存利用率
    --max-num-seqs <N> \             # 最大并发序列数
    --max-model-len <N> \            # 最大序列长度
    --dtype <auto|float16|bfloat16> \ # 数据类型
    --quantization <awq|gptq> \      # 量化方案
    --enable-prefix-caching \        # 启用前缀缓存
    --disable-log-requests \         # 关闭请求日志
    --enforce-eager \                # 强制eager模式
    --trust-remote-code              # 信任远程代码
```



**知识点3：性能压测的核心指标**



* **TTFT（Time To First Token）** ：从发送请求到收到第一个Token的时间。用户感知的“响应速度”主要由TTFT决定。Prefill阶段（处理Prompt）是TTFT的主要组成部分。

* **TPOT（Time Per Output Token）** ：生成每个输出Token的平均时间。Decode阶段（自回归生成）决定TPOT。

* **吞吐量（Throughput）** ：单位时间处理的Token数或请求数。受批处理效率影响最大。

**知识点4：PD分离的价值与实现**



Prefill和Decode两个阶段对硬件资源的需求完全不同：



* **Prefill阶段**：处理输入Prompt，一次性计算所有Token的KV Cache。**计算密集**，需要大量算力（Tensor Core），但对显存带宽要求不高。

* **Decode阶段**：基于KV Cache逐Token生成输出。**访存密集**，需要高显存带宽来读取KV Cache，但对算力要求相对较低。

传统方案将P和D放在同一个实例中，导致资源浪费——Prefill时显存闲置，Decode时算力闲置。PD分离将两者拆开，让Prefill实例使用高算力GPU，Decode实例使用高带宽GPU，理论上可将推理成本降低**60%** 。



PD分离的核心技术是**KV Transfer**——Prefill实例将生成的KV Cache通过高速通信（PyNCCL或Mooncake Store）传输给Decode实例。







### 五、常见问题与排查

**问题1：CUDA out of memory**



**原因**：显存不足。7B模型FP16约需14GB显存，加上KV Cache可能超过单卡容量。



**解决方案**：



* 降低`--gpu-memory-utilization`（如从0.9降到0.8）

* 减小`--max-num-seqs`

* 使用量化（AWQ/INT4）

* 启用`--enforce-eager`减少CUDA图缓存

**问题2：服务启动慢**



**原因**：模型加载和CUDA图编译需要时间。



**解决方案**：



* 首次启动后，后续启动会快一些（缓存机制）

* 使用`--enforce-eager`可跳过CUDA图编译，但可能影响性能

**问题3：压测时延迟突然飙升**



**原因**：超过服务容量，请求排队。



**解决方案**：



* 降低压测并发度

* 增加`--max-num-seqs`提高并发能力

* 启用连续批处理（默认已启用）

* 考虑多卡部署（增加`--tensor-parallel-size`）

**问题4：PD分离中KV Cache传输失败**



**原因**：网络配置或通信后端问题。



**解决方案**：



* 确保Prefill和Decode实例网络互通

* 检查`kv_rank`和`kv_parallel_size`配置是否正确

* 尝试更换通信后端（如从PyNccl切换到Mooncake Store）

### 六、验收清单

| 验收项          | 标准                   | 自查结果 |
| ------------ | -------------------- | ---- |
| **环境配置**     | vLLM成功安装，GPU可被识别     | ☐    |
| **模型加载**     | 7B模型成功加载，无报错         | ☐    |
| **服务启动**     | API服务正常运行，`curl`可访问  | ☐    |
| **压测执行**     | 成功运行benchmark，获得性能数据 | ☐    |
| **参数调优**     | 调整至少2个参数并观察到性能变化     | ☐    |
| **PD分离（进阶）** | 成功启动P和D两个实例并验证       | ☐    |



