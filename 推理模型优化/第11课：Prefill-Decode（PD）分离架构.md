## 引言：当“连续批处理”遇到天花板

在第7课中，我们学习了vLLM的Continuous Batching——它通过每步动态调整批次，解决了静态批处理中“长请求阻塞短请求”的问题。



然而，Continuous Batching虽然解决了“批次内等待”的问题，却引入了一个更深层的矛盾：**Prefill和Decode被强行塞进同一个批次、同一块GPU里运行**。



想象一个场景：批次中有一个新请求刚进入Prefill阶段（需要计算2048个Token的KV Cache，属于计算密集型操作，耗时较长），同时批次中还有多个请求正在Decode阶段（每个Token只需计算一个新Token，属于访存密集型操作）。因为共享GPU资源，Prefill的计算会抢占Decode的带宽，导致Decode请求的TPOT（Token间延迟）急剧恶化。



这就像一条高速公路上，重型卡车（Prefill）和跑车（Decode）共用同一车道——卡车起步慢、占道宽，跑车被堵在后面无法发挥速度优势。**PD分离架构**正是为解决这一矛盾而生。







## 一、Prefill阶段与Decode阶段的特性差异

在理解PD分离之前，我们必须先深刻理解两个阶段的本质差异。



### 1.1 Prefill阶段：计算密集型

**Prefill阶段**负责处理用户的输入Prompt，一次性计算所有输入Token的Key和Value向量，并生成第一个输出Token。



**核心特征**：



* **计算密集型**：完整运行模型的前向传播，计算所有输入token的注意力。复杂度与输入长度**平方相关**（O(n²)）。

* **并行化处理**：输入的所有Token可并行计算，充分利用GPU等硬件加速。

* **高算力需求**：Prefill阶段GPU利用率可达**90%以上**。

* **显存利用率相对较低**：算力需求高但显存可能闲置。

### 1.2 Decode阶段：访存密集型

**Decode阶段**基于KV Cache进行自回归迭代，逐个生成后续Token。



**核心特征**：



* **访存密集型**：依赖KV Cache读取历史数据，内存带宽成为主要瓶颈。复杂度与序列长度**线性相关**（O(n)）。

* **串行化生成**：每次只能生成一个Token，严格的自回归过程。

* **低算力利用率**：Decode阶段GPU利用率通常只有**25%-30%**。

* **高显存带宽需求**：每个Token都需要频繁访问KV Cache。

### 1.3 核心差异对比

| 维度         | Prefill阶段    | Decode阶段    |
| ---------- | ------------ | ----------- |
| **计算特性**   | 计算密集型        | 访存密集型       |
| **GPU利用率** | 可达90%以上      | 通常25%-30%   |
| **复杂度**    | O(n²)        | O(n)        |
| **并行度**    | 高（所有Token并行） | 低（逐Token串行） |
| **瓶颈**     | 算力（FLOPS）    | 显存带宽        |
| **优化目标**   | 降低TTFT       | 降低TPOT      |





## 二、PD分离的核心思想：计算密集 vs. 访存密集

### 2.1 混合部署的“三宗罪”

传统方案将Prefill和Decode整合在同一个GPU实例上，存在三大缺陷：



**（1）PD时延互相干扰**



Prefill和Decode阶段互相等待——Prefill计算抢占GPU资源时，Decode请求被迫等待；反之亦然。为了保证用户体验（时延小于100ms），必须牺牲并发，降低吞吐率。



**（2）计算与访存冲突**



Prefill阶段是计算密集型任务，Decode阶段是访存密集型任务，混合在同一节点上运行，导致算力和显存资源的竞争冲突。



**（3）资源利用不足**



两阶段硬件需求差异较大，为保证效果需要算力、显存资源**过配置**，混合部署难以充分利用资源。同一块GPU在Prefill时跑90%利用率，在Decode时只有25%-30%，大量算力被浪费。



### 2.2 PD分离的核心理念

PD分离的核心思想是：**将Prefill和Decode分别部署到不同的GPU实例上，针对各自特性进行专门优化**。



* **Prefill实例（P实例）** ：专注高算力任务，快速生成KV Cache

* **Decode实例（D实例）** ：专注高带宽任务，消费KV Cache生成输出

这种分离式设计带来的核心收益：



1. **消除PD间时延干扰**：Decode可以使用更大的Batch Size，计算效率更高

2. **PD灵活配比调节**：可以根据PD需要的计算资源，独立调整资源比例

3) **PD资源解耦**：P实例和D实例使用不同的硬件资源，避免资源冲突

### 2.3 PD分离的“生产-消费”模型

整个PD分离系统可以抽象为**生产-消费模型**：



```plain&#x20;text
P实例（生产KV） → KV传输 → D实例（消费KV生成Token）
```



* **P实例**：生产KV Cache（计算密集型）

* **传输层**：通过RDMA等方式将KV从P节点传输到D节点

* **D实例**：消费KV Cache生成输出Token（访存密集型）

三者组成Pipeline完成大模型推理。当三者中任一速率低并成为瓶颈，就会产生请求堆积，影响整体吞吐量和时延。



**配比调优的核心原则**：使Prefill速率、Decode速率、传输速率三者互为短板——任何一个环节过快或过慢都会导致整体效率下降。







## 三、PD分离的工程实现

### 3.1 vLLM中的PD分离实现

vLLM从0.8.x版本开始，通过**KV Transfer机制**支持PD分离（1P1D场景）。



**工作流程**：



1. **P实例**以非阻塞方式将生成的KV Cache插入缓冲区（LookupBuffer）

2. **D实例**以阻塞方式从缓冲区获取KV Cache

3) 数据传递通过管道（pipe）实现，支持PyNCCL或Mooncake Store等通信后端

**代码示例**（基于vLLM的KV Transfer）：



```python
from vllm import LLM
from vllm.config import KVTransferConfig

# ============ Prefill节点配置 (Producer) ============
ktc_producer = KVTransferConfig.from_cli('{{
    "kv_connector": "PyNcclConnector",
    "kv_role": "kv_producer",
    "kv_rank": 0,
    "kv_parallel_size": 2
}}')

llm_producer = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    kv_transfer_config=ktc_producer
)

# Producer执行Prefill，生成KV Cache并发送给Consumer
llm_producer.generate(prompts, sampling_params)

# ============ Decode节点配置 (Consumer) ============
ktc_consumer = KVTransferConfig.from_cli('{{
    "kv_connector": "PyNcclConnector",
    "kv_role": "kv_consumer",
    "kv_rank": 1,
    "kv_parallel_size": 2
}}')

llm_consumer = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    kv_transfer_config=ktc_consumer
)

# Consumer接收KV Cache，执行Decode生成后续Token
outputs = llm_consumer.generate(prompts, sampling_params)
```



**当前局限**：



* 仅支持1P1D（一个Prefill实例对应一个Decode实例），缺乏多实例（如xPyD）扩展

* 未集成负载均衡、自动扩缩容等高级调度功能

* Chunk Prefill等优化尚未完全适配

### 3.2 NVIDIA Dynamo的PD分离方案

NVIDIA Dynamo（原名TensorRT-LLM Multi-GPU Runtime）为PD分离提供了更完善的解决方案。



**核心功能**：



* **AIConfigurator**：可根据用户的SLO和可用GPU资源，自动推荐合适的PD分离配置和并行策略，并生成一键部署脚本

* **SLO-based Planner**：与Kubernetes联动，自动调节Prefill和Decode Worker数量，确保性能与资源使用率达成最佳水平

### 3.3 昇腾MindIE的PD分离实现

华为昇腾的MindIE推理引擎也提供了PD分离部署能力。



**架构组件**：



| 组件                            | 职责                                             |
| ----------------------------- | ---------------------------------------------- |
| **MindIE MS**                 | P/D实例生命周期管理、状态采集、请求调度（含Controller和Coordinator） |
| **MindIE Motor**              | 通过endpoint方式接收Coordinator推理请求                  |
| **MindIE LLM BatchScheduler** | 调度batch能力，单独调度prefill或decode类型的请求              |
| **CANN KV库**                  | 提供基于RDMA的KV Cache传输能力                          |





**配置示例**：



```json
{
    "InferMode": "dmi"  // dmi为PD分离模式
}
```



### 3.4 KV Cache传输：PD分离的“生命线”

PD分离中，KV Cache从P实例传输到D实例的效率直接决定系统性能。主流传输方案包括：



| 方案                 | 特点              | 适用场景               |
| ------------------ | --------------- | ------------------ |
| **PyNCCL**         | NVIDIA集合通信库，低延迟 | NVIDIA GPU集群       |
| **Mooncake Store** | 分布式KV缓存存储       | 大规模PD分离部署          |
| **RDMA**           | 远程直接内存访问，高带宽    | 高速网络环境（建议200Gbps+） |





## 四、PD分离的性能收益

### 4.1 量化收益数据

PD分离带来的性能提升已被多个生产环境验证：



* **DeepSeek 671B模型**：在多种场景下实现推理集群总吞吐性能**30%-72%的提升**，并发能力提升**2倍**，同等吞吐条件下推理成本下降最高达**42%**

* **昇腾大规模专家并行+PD分离**：将吞吐量进一步提升**30%以上**

* **理论分析**：Prefill阶段GPU利用率可达90%以上，而Decode阶段仅25%-30%，PD分离可让两阶段各自运行在最适合的硬件上

### 4.2 Goodput：比吞吐量更有意义的指标

PD分离的真正价值不仅在于提升原始吞吐量（Throughput），更在于提升**有效吞吐量（Goodput）** 。



**Throughput vs. Goodput**：



* **Throughput（吞吐量）** ：系统单位时间内处理的Token数或请求数——不反映延迟表现

* **Goodput（有效吞吐量）** ：系统在满足延迟约束（如TTFT/TPOT SLO）的前提下，真正完成的请求数量

**Goodput的定义**：



```plain&#x20;text
Goodput = 在满足SLO（如P90 TTFT < 200ms 且 P90 TPOT < 50ms）的前提下，
         系统每秒能完成的有效请求数
```



PD分离通过消除Prefill和Decode的相互干扰，在保持高吞吐的同时**显著降低了长尾延迟**，从而提升了Goodput。



### 4.3 不同场景下的收益差异

PD分离的收益并非在所有场景下都相同：



| 场景类型        | PD分离收益 | 原因                       |
| ----------- | ------ | ------------------------ |
| **长输入序列**   | 高      | Prefill计算量大，分离后不影响Decode |
| **高并发在线服务** | 高      | 消除PD相互干扰，稳定TPOT          |
| **短输入序列**   | 中等     | Prefill开销小，分离收益有限        |
| **低频B端调用**  | 低      | 负载低，分离的调度开销可能超过收益        |





## 五、分离部署 vs. 合并部署的权衡

### 5.1 合并部署（Aggregation）的适用场景

PD合并部署指Prefill和Decode运行在同一个GPU实例上。



**适用场景**：



* **短序列/低频请求场景**：Prefill计算量小，对Decode的干扰有限

* **资源受限环境**：无法承担多实例部署的硬件成本

* **简化的运维需求**：架构简单，部署和调试成本低

**合并部署的优化技术**：即使不采用完全的PD分离，也可以通过**Chunked Prefill（SplitFuse）** 等技术缓解PD冲突——将长Prompt切分为小块，与Decode请求混合调度，减小对Decode的影响。



### 5.2 分离部署（Disaggregation）的适用场景

**适用场景**：



* **高并发在线服务**：需要同时保证低TTFT和低TPOT

* **长输入序列场景**：RAG、文档摘要等

* **严格SLO要求**：需要同时满足TTFT和TPOT的SLO约束

* **大规模集群部署**：有足够的GPU资源进行PD分离

### 5.3 分离 vs. 合并：决策矩阵

| 考量维度       | 合并部署           | 分离部署             |
| ---------- | -------------- | ---------------- |
| **硬件成本**   | 低（单实例）         | 高（多实例，需额外GPU）    |
| **运维复杂度**  | 低              | 高（需管理P/D实例、KV传输） |
| **TTFT性能** | 中等（受Decode干扰）  | 优（P实例专注计算）       |
| **TPOT性能** | 中等（受Prefill干扰） | 优（D实例专注访存）       |
| **资源利用率**  | 低（两阶段需求冲突）     | 高（各自针对性优化）       |
| **弹性扩缩容**  | 困难             | 灵活（P/D独立扩缩）      |
| **适用负载**   | 低负载/短序列        | 高负载/长序列          |





### 5.4 混合策略：动态PD配比

最新的研究（如TaiChi系统）提出**统一PD分离与聚合**的方案——根据实时负载动态调整PD配比，在任意TTFT和TPOT的SLO组合下实现最优Goodput。



**动态配比的核心思想**：



* 将整个PD分离系统看作**生产-消费模型**，P实例生产KV，传输层传递KV，D实例消费KV

* PD配比寻优的原则是：**使Prefill速率、Decode速率、传输速率三者互为短板**

* 通过SLO-based Planner与Kubernetes联动，自动调节Prefill和Decode Worker数量

##
