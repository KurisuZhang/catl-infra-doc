

### 一、设计哲学与核心原则

#### 1.1 为什么需要vLLM？

大模型推理面临三大核心挑战：



* **吞吐低**：自回归解码的串行特性导致GPU利用率不足

* **显存墙**：KV Cache随序列长度和并发数线性增长，显存成为瓶颈

* **调度僵**：静态批处理导致短请求被长请求阻塞，GPU空闲等待

vLLM正是为系统性地解决这三个问题而设计的大模型推理引擎。其核心设计哲学可以概括为三大原则：



**（1）内存连续性：PagedAttention**



PagedAttention是vLLM最核心的创新。其核心思想是将每个请求的KV Cache划分为固定大小的**KV块（KV Blocks）**。每个块包含固定数量Token的注意力键和值。PagedAttention算法允许这些块存储在**非连续的物理内存**中，从而通过按需分配内存来消除内存碎片。



这一机制借鉴了操作系统中的虚拟内存分页技术。在PagedAttention期间，这些块充当索引结构，将Token映射到其计算出的KV Cache块。每个KV块可以通过块内的Token和块前缀的Token进行唯一哈希标识，从而实现自动前缀缓存。



**（2）计算并行：连续批处理与多级并行**



如果说PagedAttention解决了显存“空间”问题，连续批处理（Continuous Batching）则解决了“时间”效率问题。vLLM在每个解码步骤动态更新批次，而不是每批固定不变。这种设计消除了队头阻塞和空闲槽位，使静态批处理的低效问题得到根本解决。



在并行策略层面，vLLM支持：



* **张量并行（TP）** ：将单层权重切分到多卡

* **流水线并行（PP）** ：按层切分模型

* **数据并行（DP）** ：多副本并发处理

**（3）零拷贝推理：异步与共享内存**



vLLM通过多进程架构和ZMQ通信实现CPU任务并行化，显式分离API层与推理核心。这种设计将推理延迟降低40%，吞吐量提升1.7倍。



#### 1.2 vLLM的核心定位

vLLM的核心定位可以概括为：**通过极致的显存管理和调度优化，解决大模型推理中的吞吐低、显存墙和调度僵三大问题**。



vLLM通过PagedAttention和Continuous Batching两大杀手锏，将推理吞吐量提升20倍以上。







### 二、源码目录结构导读

vLLM的源码组织清晰，主要分为以下核心模块：



| 目录                     | 职责                                 | 关键文件                                                               |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------ |
| `vllm/entrypoints/`    | **入口点**：离线推理与在线服务的统一入口             | `llm.py`（LLM类）、`openai/api_server.py`（API服务器）、`cli/main.py`（CLI命令） |
| `vllm/engine/`         | **引擎**：LLMEngine与AsyncLLMEngine核心类 | `llm_engine.py`                                                    |
| `vllm/core/`           | **调度与内存**：调度器、KV Cache管理           | `block_manager.py`、`scheduler.py`                                  |
| `vllm/model_executor/` | **模型执行**：模型加载、前向计算、采样              | `layers/`（算子层）、`models/`（模型注册）                                     |
| `vllm/v1/`             | **V1引擎**：多进程架构核心实现                 | `engine/core.py`（EngineCore）、`core/sched/scheduler.py`（统一调度器）      |
| `vllm/workers/`        | **工作节点**：GPU Worker与分布式执行          | `gpu_worker.py`、`gpu_model_runner.py`                              |





**LLMEngine是vLLM的入口类**，其初始化过程揭示了框架的关键配置：



* **模型加载**：通过ModelExecutor抽象硬件后端，支持动态切换CUDA/ROCm

* **调度器**：Scheduler负责请求排序、批处理分组

* **内存管理**：MemoryManager初始化时预分配连续内存块

### 三、V1多进程架构解析

vLLM V1采用**多进程架构**来分离关注点并最大化吞吐量。理解此架构对于在部署中正确规划CPU资源至关重要。



#### 3.1 进程架构全景

V1架构包含四种核心进程类型：



```plain&#x20;text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Server 进程                                  │
│              (HTTP处理、输入预处理、ZMQ通信)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ ZMQ (多对多拓扑)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Engine Core 进程                                   │
│               (调度器、KV Cache管理、GPU Worker协调)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GPU Worker 进程群                                     │
│           (每个GPU一个，加载模型权重、执行前向传播)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```



#### 3.2 API Server进程

API服务器进程负责处理HTTP请求（如OpenAI兼容API），执行输入处理（分词、多模态数据加载），并将结果流式传输回客户端。



**关键特性**：



* **进程数量**：默认1个API服务器进程；使用数据并行时自动扩展以匹配数据并行大小

* **手动配置**：可通过 `--api-server-count` 标志手动配置

* **通信方式**：通过ZMQ以**多对多拓扑**连接到所有Engine Core，使任何API服务器都能将请求路由到任何Engine Core

* **多线程**：每个API服务器进程使用多个CPU线程进行媒体加载（由 `VLLM_MEDIA_LOADING_THREAD_COUNT` 控制，默认为8）

**代码位置**：`vllm/entrypoints/openai/api_server.py` 和 `vllm/v1/utils.py`



#### 3.3 Engine Core进程

Engine Core进程运行调度器、管理KV Cache、协调GPU Worker执行模型。它运行一个**忙循环（Busy Loop）** ，持续调度请求并将工作分派给GPU Worker。



**关键特性**：



* **进程数量**：每个数据并行秩（Data Parallel Rank）1个Engine Core进程。例如，`--data-parallel-size 4` 时有4个Engine Core进程

* **核心实现**：`EngineCore`类实现了vLLM引擎的内循环，协调调度器与模型执行器

**隔离式EngineCore的设计优势**：



* **稳定性增强**：API服务与推理核心物理隔离，避免因请求洪峰导致核心服务崩溃

* **资源隔离**：不同模型实例可分配独立GPU资源，防止资源争抢

* **扩展性提升**：支持横向扩展推理节点

**代码位置**：`vllm/v1/engine/core.py` 和 `vllm/v1/core/sched/scheduler.py`



#### 3.4 GPU Worker进程

每个GPU由一个专用的Worker进程管理。



**核心职责**：



* 加载模型权重

* 执行前向传播

* 管理GPU内存

**进程数量**：



* 每个GPU 1个Worker进程

* 总数 = `tensor_parallel_size × pipeline_parallel_size`（每个Engine Core）

**代码位置**：`vllm/v1/worker/gpu_worker.py` 和 `vllm/v1/worker/gpu_model_runner.py`



#### 3.5 DP Coordinator进程

当使用数据并行（`--data-parallel-size > 1`）时，会额外启动一个协调器进程。



**核心职责**：



* 管理跨DP秩的负载均衡

* 协调MoE模型的同步前向传播

**进程数量**：仅在启用数据并行时有1个DP Coordinator进程



**代码位置**：`vllm/v1/engine/coordinator.py`



#### 3.6 进程数量汇总

对于一个部署配置（N张GPU、TP张量并行大小、DP数据并行大小、A个API服务器）：



| 进程类型               | 数量                       | 说明             |
| ------------------ | ------------------------ | -------------- |
| **API Server**     | A（默认=DP）                 | 处理HTTP请求和输入处理  |
| **Engine Core**    | DP（默认1）                  | 调度器和KV Cache管理 |
| **GPU Worker**     | N（= DP × TP）             | 每个GPU一个，执行前向传播 |
| **DP Coordinator** | 1（如果DP>1）                | 跨DP秩的负载均衡      |
| **总计**             | A + DP + N + (1 if DP>1) |                |





**典型场景**：单节点4张GPU部署（`vllm serve`），TP=4，DP=1（默认），API Server=1：



* API Server: 1

* Engine Core: 1

* GPU Worker: 4

* DP Coordinator: 0

* **总计: 6个进程**

#### 3.7 进程间通信：ZMQ

API Server与Engine Core之间通过**ZMQ（ZeroMQ）** 套接字进行通信。



**通信特性**：



* **多对多拓扑**：每个API Server连接到所有Engine Core

* **请求路由**：任何API Server都可以将请求路由到任何Engine Core

* **数据并行支持**：同一ZMQ机制也用于前端与Engine Core之间的通信

**为什么选择ZMQ？** ZMQ提供了高性能的异步消息传递，支持多对多通信模式，且易于与Python的异步框架集成。







### 四、核心入口点分析

vLLM提供了两种主要的系统交互入口点。



#### 4.1 LLM类：离线推理

`LLM`类提供了主要的Python接口，用于**离线推理**——即在不使用独立模型推理服务器的情况下与模型进行交互。



**使用示例**：



```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```



**代码位置**：`vllm/entrypoints/llm.py`



#### 4.2 在线服务：vLLM Serve

vLLM的第二个主要接口是通过其**在线服务器**，可通过 `vllm serve` 命令启动。



```bash
vllm serve <model>
```



**CLI代码位置**：`vllm/entrypoints/cli/main.py`



**注意事项**：



* 直接使用 `python -m vllm.entrypoints.openai.api_server --model <model>` 的方式**已被弃用**，未来版本可能不再支持

* 推荐使用 `vllm serve` CLI命令

**API服务器代码位置**：`vllm/entrypoints/openai/api_server.py`

