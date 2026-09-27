##

如果说vLLM是用**系统架构的创新**解决了显存管理的瓶颈，那么TensorRT-LLM就是用**硬件级的深度优化**将NVIDIA GPU的每一分算力榨取到极致。



TensorRT-LLM是NVIDIA官方推出的大模型推理加速库，基于TensorRT深度学习编译框架构建，整合了FastTransformer中高效的Kernel实现，并通过NCCL完成设备间通信。它的核心定位是**充分发挥NVIDIA GPU硬件能力，提供极致的推理性能**。



与vLLM“通用、易用、开源友好”的定位不同，TensorRT-LLM选择了一条更激进的道路：**为NVIDIA硬件量身定制，在性能上不留余地**。正如一位资深工程师所言：“如果你在NVIDIA硬件上提供服务，且每一毫秒都至关重要，TensorRT-LLM就是答案。”



## 一、TensorRT-LLM的架构设计

### 1.1 整体架构概览

TensorRT-LLM是一个专门为大语言模型推理设计优化方案的工具包。其架构可以分为以下几个核心层次：



```plain&#x20;text
用户应用程序
        ↓
┌─────────────────────────────────────────────┐
│  LLM API (Python)                           │
│  - 统一的High-Level入口                       │
│  - 自动管理Tokenization/Detokenization       │
│  - 支持Hugging Face模型直接加载               │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  PyExecutor / Executor (C++/Python)         │
│  - 异步请求执行                              │
│  - In-flight Batching调度                   │
│  - 资源管理（KV Cache分配/回收）             │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Model Engine                               │
│  - 模型前向计算引擎                         │
│  - 支持TensorRT和PyTorch两种Backend         │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  TensorRT Runtime / CUDA Kernels            │
│  - 图优化与算子融合                         │
│  - Tensor Core加速                          │
│  - CUDA Graph执行                          │
└─────────────────────────────────────────────┘
        ↓
NVIDIA GPU (A100/H100/L40S/Blackwell)
```



**两个值得注意的设计特点**：



1. **双Backend架构**：TensorRT-LLM不仅支持TensorRT作为后端，PyTorch也可以作为后端。这意味着你可以在PyTorch生态中获得TensorRT-LLM的优化能力。

2. **LLM API统一入口**：`tensorrt_llm.LLM`类自动管理Tokenization和Detokenization过程，大幅降低了使用门槛。

### 1.2 核心组件详解

**PyExecutor**：类似TensorRT后端的Executor API，PyExecutor是PyTorch后端的执行器。它的单步执行流程包括：



1. 从请求队列获取新请求

2. 调度部分请求

3) 对调度到的请求执行模型前向计算

4) 使用模型输出运行解码器

5. 为每个请求添加输出Token，处理完成的请求

**Scheduler（调度器）** ：分为两个层次：



* **CapacityScheduler**：判断是否有足够的资源容纳新请求

* **MicroBatchScheduler**：选择哪些请求在当前步骤执行前向计算

**ResourceManager（资源管理器）** ：负责分配和管理单个请求推理所需的资源，包括KV Cache等。



### 1.3 入门示例：LLM API

```python
from tensorrt_llm import LLM, SamplingParams

# 1. 初始化LLM实例（自动加载模型并构建TensorRT引擎）
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 2. 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

# 3. 执行推理
prompts = [
    "The future of AI is",
    "In a world where machines think",
]
outputs = llm.generate(prompts, sampling_params)

# 4. 输出结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```



这是最简化的入门方式——`LLM`类会自动完成模型加载、TensorRT引擎构建和推理执行。在v1.0版本中，TensorRT-LLM还引入了`trtllm-serve`命令行工具，用于部署OpenAI兼容的服务。







## 二、图优化与算子融合

### 2.1 图优化：从“散兵游勇”到“集团作战”

在标准PyTorch推理中，模型的计算图由大量细粒度的算子（Kernel）组成。每个算子的执行都伴随着：



* **内核启动开销**：CPU向GPU发送指令

* **内存读写开销**：算子间传递中间结果

当模型有数百层时，这些开销累积起来相当可观。



TensorRT-LLM的图优化核心思路是：**将多个细粒度算子融合为一个粗粒度CUDA内核**，减少内核启动次数和内存访问。



**融合的典型模式**：



* LayerNorm + GELU + 矩阵乘法 → 单个融合Kernel

* 量化线性层 + 激活函数 → 融合量化GEMM

* MoE（混合专家）的相关计算模式

### 2.2 算子融合的微观示例

以一个典型的Transformer层为例：



**优化前（多个独立Kernel）** ：



```plain&#x20;text
Input → [MatMul] → [BiasAdd] → [LayerNorm] → [GELU] → [MatMul] → Output
         ↑ 7次Kernel启动，6次中间结果读写
```



**优化后（融合Kernel）** ：



```plain&#x20;text
Input → [Fused_MLP_Kernel] → Output
         ↑ 1次Kernel启动，0次中间结果读写
```



这就是为什么TensorRT-LLM在同样硬件上能比原生PyTorch快2-3倍的原因之一。



### 2.3 加载后融合（Post-Load Fusion）

TensorRT-LLM还支持一种称为“加载后融合”的优化阶段，它利用已加载的权重、设备张量或最终的分片图结构来应用性能优化。这意味着优化不仅发生在编译时，还发生在运行时，能够根据实际硬件状态进行动态调整。



### 2.4 CUDA Graph：将“多次启动”变为“一次启动”

CUDA Graph是TensorRT-LLM另一个重要的图优化技术。它将一系列GPU操作“录制”为一个图，后续执行时只需一次启动即可完成所有操作。



**CUDA Graph的效果**：



* 消除数千次内核启动的开销

* 减少CPU-GPU同步等待

* 对短序列、小Batch场景尤其有效

**类比理解**：普通执行就像每次做菜都要重新洗锅、切菜、开火；CUDA Graph则像把整套菜谱预先排好，每次只需按一个“开始”按钮。







## 三、In-flight Batching与KV Cache管理

### 3.1 In-flight Batching：vLLM的“连续批处理”在NVIDIA生态中的实现

TensorRT-LLM支持**In-flight Batching（IFB）** ——也就是我们在第7课学习的连续批处理（Continuous Batching）或迭代级批处理（Iteration-level Batching）。



**核心机制**：在每个生成步骤动态地将新请求添加到运行中的批次，同时处理Prefill阶段和Decode阶段的请求。



**关键实现细节**：为了效率，TensorRT-LLM要求处于Prefill阶段的请求在输入张量中**排在Decode阶段请求之前**。这种“无填充（No Padding）”的打包方式避免了将单Token的Decode请求填充到最大序列长度的资源浪费。



**类比理解**：传统静态批处理像火车——必须等车厢坐满才发车，且中途不能上下客；In-flight Batching像地铁——每站都有人上下车，车厢永远在高效运转。



### 3.2 KV Cache管理：分页、量化、重用与卸载

TensorRT-LLM在KV Cache管理上提供了比vLLM更丰富的优化手段。



**（1）分页KV Cache（Paged KV Cache）**



与vLLM的PagedAttention类似，TensorRT-LLM也将KV Cache划分为固定大小的块（Block），每个块存储固定数量Token的KV数据。块是KV Cache分配的最小单位。



**层次结构**：



* **Pool（池）** ：连续的显存缓冲区，存储实际的KV数据。有主池（GPU显存）和辅助池（CPU或Offload内存）

* **Block（块）** ：KV Cache分配的最小逻辑单元，持有元数据（metadata）而非实际数据

* **Page（页）** ：在代码中常与Block互换使用

**（2）KV Cache量化**



TensorRT-LLM支持**量化KV Cache**，进一步压缩显存占用。这对于长上下文场景尤为重要——KV Cache可能比模型权重本身占用更多显存。



**（3）循环缓冲区KV Cache（Circular Buffer KV Cache）**



对于滑动窗口注意力（Sliding Window Attention）等场景，TensorRT-LLM支持循环缓冲区KV Cache，只保留窗口内的KV，丢弃窗口外的历史数据。



**（4）KV Cache重用（KV Cache Reuse）**



TensorRT-LLM提供了基于优先级的驱逐和事件感知路由，实现对KV Cache的精细控制。多个请求可以共享相同的KV块（如相同的System Prompt）。



**（5）KV Cache Offloading**



TensorRT-LLM支持将KV Cache从GPU卸载到主机内存（Host Memory）。当GPU显存不足时，可以将不活跃请求的KV Cache暂存到CPU内存，需要时再换回。



**（6）KV Cache Connector**



这是一个灵活的接口，允许开发者实现自定义的KV Cache加载、保存和管理逻辑，甚至可以将KV Cache持久化到外部存储（磁盘、数据库、分布式缓存等）。



### 3.3 KV Cache调优实战

TensorRT-LLM默认会尝试分配**90%的空闲GPU显存**给KV Cache。如果这个比例过于激进，可以调低：



```python
from tensorrt_llm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_config={
        "free_gpu_memory_fraction": 0.7,  # 从默认0.9降到0.7
    }
)
```



**核心调优参数**：



| 参数               | 含义                     | 调优建议                                       |
| ---------------- | ---------------------- | ------------------------------------------ |
| `max_batch_size` | 单批次最大请求数               | 设置足够高以不成为吞吐瓶颈；运行时动态调整                      |
| `max_seq_len`    | 单请求最大序列长度              | 默认设为`max_position_embeddings`，除非显存不足否则不需调整 |
| `max_num_tokens` | 每批次最大Token数（去Padding后） | 默认8192，推荐调优以获得最佳性能                         |





## 四、TensorRT-LLM vs. vLLM：选型对比

### 4.1 核心差异总览

| 维度       | vLLM               | TensorRT-LLM                    |
| -------- | ------------------ | ------------------------------- |
| **核心定位** | 通用高性能LLM推理         | NVIDIA硬件极致优化                    |
| **内存管理** | PagedAttention     | 分页KV Cache + 量化 + Offloading    |
| **优化手段** | 系统架构创新             | CUDA Kernel + 图融合 + Tensor Core |
| **模型支持** | Hugging Face生态（灵活） | 针对LLaMA、Mistral、GPT、Qwen等优化     |
| **硬件生态** | GPU优先，支持CUDA/ROCm  | 最佳在NVIDIA A100/H100/L40S        |
| **易用性**  | 易于集成HF/开源工具        | 配置较复杂，依赖NVIDIA SDK              |
| **开源生态** | 开源，社区活跃            | 开源但深度绑定NVIDIA栈                  |
| **性能**   | 大Batch下吞吐优秀        | NVIDIA GPU上峰值性能                 |





### 4.2 性能对比的定性分析

**vLLM的优势场景**：



* 需要快速集成Hugging Face模型

* 混合使用不同厂商GPU（NVIDIA + AMD）

* 开源社区优先，希望减少供应商锁定

* 长上下文场景（PagedAttention在长上下文下表现优异）

**TensorRT-LLM的优势场景**：



* 纯NVIDIA GPU环境（尤其是A100/H100/Blackwell）

* 追求极致性能（每一毫秒都重要）

* 已有NVIDIA生态投资（Triton、NeMo等）

* 需要KV Cache Offloading等高级内存管理功能

**一个实用的判断标准**：如果你的场景是“NVIDIA GPU + 极致性能要求”，TensorRT-LLM是首选；如果是“快速上线 + 灵活部署 + 混合硬件”，vLLM更合适。



### 4.3 代码对比：同样的任务，不同的写法

**vLLM（离线推理）** ：



```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```



**TensorRT-LLM（离线推理）** ：



```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```



> 有趣的是，两者的High-Level API在vLLM V1和TensorRT-LLM v1.0之后已经**趋于一致**。真正的差异在底层——TensorRT-LLM在构建引擎时会进行图优化和内核融合，而vLLM则在运行时通过PagedAttention优化内存。
>
>



### 4.4 选型决策树

```plain&#x20;text
你的GPU环境？
├── 纯NVIDIA（A100/H100/L40S）
│   ├── 追求极致性能（<100ms延迟要求）
│   │   └── TensorRT-LLM + FP8量化
│   ├── 需要快速迭代、灵活部署
│   │   └── vLLM（或TensorRT-LLM的LLM API）
│   └── 已有Triton/NVIDIA生态
│       └── TensorRT-LLM
│
├── 混合GPU（NVIDIA + AMD）
│   └── vLLM（支持ROCm）
│
└── 需要KV Cache Offloading/高级内存管理
    └── TensorRT-LLM
```

###
