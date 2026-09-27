大模型推理面临的核心矛盾是什么？**显存不够用，GPU吃不饱**。传统推理框架中，KV Cache的显存管理方式导致了两个严重问题：内存碎片化和静态批处理的低效。vLLM正是为解决这两个问题而生的。



vLLM的核心哲学是 **“不动模型结构，只动系统架构”** 。它不追求修改Attention机制或压缩模型权重，而是借鉴操作系统中的虚拟内存+分页管理思想，将KV缓存拆解为固定大小的“块（Block）”，通过全局Block Manager统一调度，实现显存的非连续分配与高效复用。



本课将深入解析vLLM的三大核心——架构设计、PagedAttention和连续批处理，并通过代码示例展示如何进行性能调优。



## 一、vLLM的架构概览

### 1.1 分层架构设计

vLLM采用分层执行器-工作器架构，清晰地分离了调度逻辑和执行逻辑：



```plain&#x20;text
用户请求
    ↓
Entrypoints (LLM / AsyncLLMEngine / OpenAI-compatible API)
    ↓
LLMEngine (核心引擎：调度、内存管理、输出处理)
    ↓
Executor (执行器：管理分布式工作器)
    ↓
Workers (工作器：在GPU上实际运行模型)
    ↓
CUDA/HIP Kernels (高度优化的底层内核)
```



**四层架构的核心职责**：



1. **调度层（Scheduler Layer）** ：接收用户请求，根据模型状态和硬件资源动态分配任务。采用多级优先级队列，支持不同优先级的请求调度。

2. **执行层（Execution Layer）** ：包含模型加载、张量计算和结果生成三个子模块。集成CUDA加速库（CuBLAS、Triton），支持混合精度计算。

3) **内存管理层（Memory Management Layer）** ：vLLM的核心创新所在。通过分页显存分配、权重共享和缓存复用，实现高效内存利用。

4) **接口层（API Layer）** ：提供RESTful API和gRPC服务，支持OpenAI兼容的API格式。

### 1.2 一个通俗的类比

可以把vLLM想象成高速公路上的智能调度系统：



* **收费站（Entrypoints）** ：用户的请求从这里进入系统

* **交通指挥中心（Scheduler）** ：智能决定谁先走、怎么组队

* **分页停车场（PagedAttention）** ：像操作系统的虚拟内存，把KV Cache分成小块灵活管理

* **高速车道（Workers）** ：实际的GPU计算单元

### 1.3 离线推理入门示例

```python
from vllm import LLM, SamplingParams

# 准备 prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 初始化 LLM 引擎
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 执行推理
outputs = llm.generate(prompts, sampling_params)

# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
```



### 1.4 服务化部署入门示例

```bash
# 启动 vLLM 服务 (OpenAI 兼容 API)
vllm serve meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```



启动后，可以通过OpenAI API客户端调用：



```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="NOT_NEEDED"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Explain AI in one sentence"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```







## 二、PagedAttention的核心原理：分页管理KV Cache

### 2.1 传统KV Cache管理的痛点

在标准Transformer推理中，每个请求的KV Cache需要存储在**连续的物理显存**中。这带来了三个严重问题：



1. **内存碎片化**：不同请求的序列长度各异，分配和释放连续内存块会产生大量碎片，导致显存利用率下降。

2. **预留浪费**：必须为每个请求预先分配最大可能长度的显存，造成大量浪费。

3) **并发受限**：碎片化和预留浪费共同限制了可并发处理的请求数量。

### 2.2 PagedAttention的核心思想

PagedAttention借鉴了操作系统中的**虚拟内存分页技术**，将KV Cache划分为固定大小的块（Block），每个块包含固定数量Token的注意力键和值。



**关键创新点**：



* **非连续存储**：每个请求的KV Cache可以存储在**非连续的物理显存**中

* **按需分配**：需要多少块就分配多少块，无需预留

* **块级共享**：多个请求可以共享相同的KV块（如共享的System Prompt）

* **零碎片**：释放的块直接回归资源池，无碎片产生

### 2.3 KV Block的存储结构

vLLM将每个请求的KV Cache划分为KV Blocks。每个块存储**固定数量（BLOCK\_SIZE）的Token**在一个注意力头上的键和值数据。



块大小公式：



```plain&#x20;text
单块显存占用 = 2 × block_size × num_kv_heads × head_size × dtype_bytes
```



默认 `block_size=16`，对于LLaMA-7B（num\_kv\_heads=32，head\_size=128，dtype=FP16）：



```plain&#x20;text
单块大小 = 2 × 16 × 32 × 128 × 2 = 262,144 字节 ≈ 256 KB
```



### 2.4 前缀缓存（Prefix Caching）

PagedAttention的块级管理自然支持了**前缀缓存**。vLLM将所有KV块存储在一个全局哈希表中：



* 每个KV块通过 `hash(前缀Tokens + 块内Tokens)` 唯一标识

* 新请求若共享相同前缀，可直接复用已缓存的KV块，**无需重新计算**

* 当缓存满时，采用 **LRU（最近最少使用）** 淘汰策略，优先淘汰引用计数为0的块

**实战收益**：在FAQ类应用或RAG场景中，如果system prompt完全固定（连标点和空格都一致），前缀缓存可以带来**QPS的显著提升**。







## 三、连续批处理（Continuous Batching）的实现机制

### 3.1 传统静态批处理的缺陷

传统推理系统采用**静态批处理（Static Batching）** ：



* 批次大小在推理开始前固定

* 必须等待批次中**所有请求完成**才能处理下一批

* 长请求会阻塞整个批次的完成

* 不同长度请求需要**填充（Padding）** 到相同长度，浪费计算资源

vLLM的连续批处理将系统吞吐量提升**23倍**，同时降低P50延迟。



### 3.2 Continuous Batching的核心机制

Continuous Batching的核心是 **“每解码一步更新一次批次”** ：



**工作流程**：



1. 调度器维护一个**运行队列（Running Queue）** 和**等待队列（Waiting Queue）**

2. 每完成一个解码步骤，检查是否有请求完成

3) 一旦某个请求完成生成，**立即用新请求填充空位**

4) 新请求可以处于**Prefill阶段**或**Decode阶段**

**类比理解**：就像餐厅的“翻台率”优化——客人的菜上完了立刻结账走人，空出的座位马上安排新客人坐下，**座位永远不空着**。



### 3.3 Prefill与Decode的统一调度

在vLLM调度器的视角中，**不存在Prefill阶段和Decode阶段的严格区分**。每个请求主要关注：



* `num_computed_tokens`：已经计算过的Token数（含前缀缓存命中的部分）

* `num_tokens_to_generate`：还需要生成的Token数

调度器在每个步骤中：



1. 从等待队列中选择最多 `max_num_seqs` 个序列

2. 确保总Token数不超过 `max_num_batched_tokens`

3) 混合调度Prefill请求和Decode请求

### 3.4 分块预填充（Chunked Prefill）

当输入序列长度超过 `max_num_batched_tokens` 时，vLLM会自动将Prefill阶段的输入**切分为多个Chunk**进行处理。



**分块预填充的优势**：



* 将大的Prefill分成较小的块，与Decode请求一起批处理

* 避免长Prompt独占GPU，阻塞其他Decode请求

* 在vLLM V1中默认启用

### 3.5 抢占与恢复（Preemption）

当KV Cache空间不足时，vLLM支持请求的**抢占（Preemption）** ：



* **交换（Swapping）** ：将抢占请求的KV Cache从GPU交换到CPU内存

* **重计算（Recomputation）** ：丢弃KV Cache，后续重新计算

PagedAttention的块级管理使这些操作更加灵活高效。







## 四、vLLM的性能调优实践

### 4.1 核心调优参数

| 参数                         | 作用                | 调优建议                  |
| -------------------------- | ----------------- | --------------------- |
| `--gpu-memory-utilization` | GPU显存预分配比例（默认0.9） | 尽可能调高以提供更多KV Cache空间  |
| `--max-num-seqs`           | 单批次最大并发序列数        | 高吞吐场景可设256；低延迟场景应减小   |
| `--max-num-batched-tokens` | 单批次最大Token总数      | 推荐 >8192（特别是小模型+大GPU） |
| `--max-model-len`          | 最大序列长度            | 按实际业务需求设置，避免浪费        |
| `--enable-prefix-caching`  | 启用前缀缓存            | 高重复前缀场景必开             |
| `--enable-chunked-prefill` | 启用分块预填充           | vLLM V1默认启用           |





### 4.2 吞吐优先的配置

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching
```



### 4.3 延迟优先的配置

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048
```



**原理**：较小的 `max_num_batched_tokens`（如2048）能获得更好的Token间延迟（ITL），因为更少的Prefill会拖慢Decode。



### 4.4 客户端请求塑形

为了最大化Continuous Batching的效果，客户端需要配合：



```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="NOT_NEEDED")

# 控制并发请求数
SEM = asyncio.Semaphore(128)

async def ask(msg: str):
    async with SEM:
        stream = await client.chat.completions.create(
            model="your-vllm-model",
            messages=[{"role": "system", "content": "You are concise."},
                      {"role": "user", "content": msg}],
            temperature=0.7,
            max_tokens=256,  # 控制输出长度，避免拖垮batch
            stream=True
        )
        out = []
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            out.append(token)
        return "".join(out)

async def main():
    questions = [f"Question {i}" for i in range(500)]
    answers = await asyncio.gather(*[ask(q) for q in questions])
```



**关键建议**：



* `max_tokens` 控制在256-512（聊天场景）

* System Prompt**完全固定**（连标点和空格都不能改），最大化前缀缓存命中率

* 动态内容（用户数据、时间戳）放到消息末尾

### 4.5 推测解码（Speculative Decoding）

在GPU预算紧张时，推测解码值得尝试：



```bash
vllm serve main-model \
    --speculative-draft-model tiny-draft-model
```



小模型先提议Token，大模型负责验证，整体步数可减少。在temperature 0.3-0.9区间、中等长度输出的场景效果最好。



### 4.6 调优决策树

```plain&#x20;text
你的优化目标是什么？
├── 最大化吞吐（离线批处理）
│   ├── 增大 max-num-seqs (如 256)
│   ├── 增大 max-num-batched-tokens (如 8192 或更高)
│   └── 提高 gpu-memory-utilization (0.95)
│
├── 最小化延迟（在线交互）
│   ├── 减小 max-num-seqs (如 16-32)
│   ├── 减小 max-num-batched-tokens (如 2048)
│   └── 启用流式输出 (stream=True)
│
└── 平衡吞吐与延迟
    ├── 启用 chunked-prefill
    ├── 启用 prefix-caching（高重复前缀场景）
    └── 使用 auto_tune.sh 自动调优
```

##
