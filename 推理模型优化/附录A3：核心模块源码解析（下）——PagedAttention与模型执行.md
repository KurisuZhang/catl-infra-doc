

> 📌 **源码版本说明**：本课基于vLLM V1架构。PagedAttention Kernel位于 `csrc/attention/attention_kernels.cu`。模型执行相关代码位于 `vllm/v1/worker/gpu/model_runner.py`。分布式执行器相关代码位于 `vllm/executor/executor_base.py` 和 `vllm/v1/executor/ray_executor.py`。
>
>



### 一、PagedAttention原理与CUDA Kernel解析

#### 1.1 设计原理回顾：从“连续内存”到“分页存储”

PagedAttention是vLLM最核心的创新。其设计原理可以概括为三个层次：



**（1）传统Attention的内存瓶颈**



标准Transformer的注意力计算中，每个请求的KV Cache需要在**连续的物理显存**中分配。这导致了三大问题：内存碎片化、预留浪费（必须为每个请求预分配最大可能长度）、批处理受限。



**（2）PagedAttention的解决方案**



PagedAttention借鉴操作系统虚拟内存分页技术，将KV Cache划分为固定大小的**块（Block）** 。每个Block存储固定数量Token（默认16个）的Key和Value数据。这些Block可以存储在**非连续的物理显存**中，通过Block Table维护逻辑块到物理块的映射。



**（3）PagedAttention的数学等价性**



PagedAttention在数学上与标准Attention**完全等价**。分页存储不影响注意力计算的数学结果——无论KV在物理上如何分布，只要通过Block Table正确寻址，计算出的注意力分数与连续存储时完全一致。



#### 1.2 PagedAttention CUDA Kernel的输入与模板参数

vLLM使用自己实现的多头查询注意力Kernel，位于 `csrc/attention/attention_kernels.cu`。该Kernel专为分页KV Cache设计，Key和Value存储在不同的块中。



**Kernel函数签名**（来自官方文档）：



```c++
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, 
         int NUM_THREADS, int PARTITION_SIZE = 0>
__global__ void paged_attention_kernel(
    const scalar_t* __restrict__ q,      // Query指针
    const scalar_t* __restrict__ k_cache, // Key Cache指针
    const scalar_t* __restrict__ v_cache, // Value Cache指针
    scalar_t* __restrict__ out,          // 输出指针
    // ... 其他运行时参数
)
```



**模板参数解析**：



| 模板参数             | 含义                  | 典型值      |
| ---------------- | ------------------- | -------- |
| `scalar_t`       | 数据类型（FP16/BF16/FP8） | `half`   |
| `HEAD_SIZE`      | 每个注意力头的维度           | 128      |
| `BLOCK_SIZE`     | 每个KV块中的Token数       | 16       |
| `NUM_THREADS`    | 每个线程块中的线程数          | 128/256  |
| `PARTITION_SIZE` | 张量并行GPU数量           | 0（TP禁用时） |





**三个核心输入指针**：



* `q`：指向Query数据，形状为 `[num_seqs, num_heads, head_size]`

* `k_cache`：指向分页存储的Key Cache

* `v_cache`：指向分页存储的Value Cache

**关键概念**：这是一个**单查询注意力Kernel**（single-query attention kernel），每个序列只有一个查询Token。因此 `num_seqs` 等于批次中处理的总Token数。



#### 1.3 内存布局：Paged KV Cache的存储格式

Paged KV Cache在GPU显存中采用**分块存储**格式：



```plain&#x20;text
k_cache: [num_blocks, block_size, num_heads, head_size]
v_cache: [num_blocks, block_size, num_heads, head_size]
```



* `num_blocks`：物理块总数（由 `gpu_memory_utilization` 和模型大小决定）

* `block_size`：每个块存储的Token数（默认16）

* `num_heads`：注意力头数

* `head_size`：每个头的维度

**与标准KV Cache的对比**：



| 维度    | 标准KV Cache                             | Paged KV Cache                       |
| ----- | -------------------------------------- | ------------------------------------ |
| 存储方式  | 连续内存 `[seq_len, num_heads, head_size]` | 分块存储 `[num_blocks, block_size, ...]` |
| 物理连续性 | 连续                                     | 离散（通过Block Table寻址）                  |
| 内存分配  | 预分配最大长度                                | 按需分配Block                            |
| 内存碎片  | 严重                                     | 消除                                   |





#### 1.4 内存访问优化：从全局内存到共享内存

PagedAttention Kernel依赖**专门设计的内存布局和访问方法**实现高性能，特别是在线程将数据从全局内存读取到共享内存时。



**核心优化策略**：



1. **向量化加载（Vectorized Load）** ：使用向量类型（如 `float4`、`uint4`）一次性加载16字节数据，减少内存事务数量。

2. **线程组协作（Thread Group）** ：少量线程（`THREAD_GROUP_SIZE`）组成小组，一次获取并计算一个查询Token和一个键Token。

3) **共享内存缓存**：将分页的KV数据加载到共享内存，减少对全局内存的重复访问。

4) **计算与访存重叠**：通过流水线设计，在等待全局内存数据时执行计算。

**向量大小计算**（以FP16为例）：



* 如果 `scalar_t` 是FP16（2字节），`THREAD_GROUP_SIZE` 为2

* `VEC_SIZE = 4`（每个线程组一次获取16字节）

* `V_VEC_SIZE = 8`（每个线程一次获取16字节）

#### 1.5 PagedAttention的计算流程

**Step 1: 准备阶段**



Kernel首先计算当前线程的头索引、块索引等必要变量。每个线程根据其在线程块中的位置，确定需要处理的数据范围。



**Step 2: 迭代KV块**



对于当前查询Token，Kernel遍历其Block Table中记录的所有物理块：



```c++
// 伪代码：PagedAttention核心循环
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block_id = block_table[block_idx];
    
    // 从k_cache加载该块的Key数据到共享内存
    load_k_block_to_shared(k_cache, physical_block_id);
    // 从v_cache加载该块的Value数据到共享内存
    load_v_block_to_shared(v_cache, physical_block_id);
    __syncthreads();
    
    // 在共享内存上计算注意力
    for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
        // 计算Q与K的点积
        float score = dot_product(q, k_shared[token_in_block]);
        // 更新最大值和Softmax分母
        update_softmax_stats(score);
    }
    __syncthreads();
}
```



**Step 3: 在线Softmax**



由于KV是分块存储的，Kernel需要在遍历块的同时**在线更新Softmax统计量**（最大值 `m` 和和 `sum`）——与FlashAttention的在线Softmax类似。



**Step 4: 输出写回**



完成所有块的遍历后，将最终计算结果写回 `out` 指针指向的全局内存。



#### 1.6 Block Size的工程约束

PagedAttention Kernel对 `BLOCK_SIZE` 的支持有限制，以减少编译时间：



```c++
// attention_kernels.cu#L663-L664
// 不支持 block_size = 1/2/4/64/128/256
// 支持的 block_size 通常为 8 或 16
```



vLLM默认使用 `BLOCK_SIZE=16`，在内存利用率和Kernel效率之间取得平衡。



### 二、模型执行与分布式机制

#### 2.1 Worker与ModelRunner：模型执行的“引擎室”

在vLLM V1架构中，模型执行的核心组件是 **Worker** 和 **ModelRunner**。



**Worker进程**：每个GPU由一个专用的Worker进程管理，负责加载模型权重和执行前向传播。Worker进程数量由并行配置决定：`num_workers = tensor_parallel_size × pipeline_parallel_size`。



**GPUModelRunner**：是模型执行的核心协调器，位于 `vllm/v1/worker/gpu/model_runner.py`。它负责：



* 协调模型前向传播（model forward pass）

* 管理GPU上的请求状态

* 协调KV Cache分配

* 与采样和投机解码子系统集成

* 准备注意力元数据和批次组装

**两种实现**：

| 实现                      | 说明                                                 |
| ----------------------- | -------------------------------------------------- |
| **GPUModelRunner（稳定版）** | 在单个类结构中包含文本和多模态模型的逻辑                               |
| **Model Runner V2**     | V1引擎的模块化架构，保持核心Runner最小化，将特定行为委托给采样、投机解码、多模态处理等子组件 |





**GPUModelRunner的关键数据结构**：



| 数据结构       | 代码实体               | 用途                           |
| ---------- | ------------------ | ---------------------------- |
| 请求状态       | `RequestState`     | 管理每个请求的元数据和Token历史，使用UVA节省显存 |
| 输入批次       | `InputBatch`       | 封装单次迭代的GPU张量                 |
| CUDA Graph | `CudaGraphManager` | 管理模型前向传播的捕获和重放，消除内核启动开销      |
| 注意力分组      | `AttentionGroup`   | 共享注意力元数据和后端配置的模型层逻辑分组        |





#### 2.2 ModelRunner的执行流水线

`execute_model` 方法是单次推理迭代的入口点：



```python
# vllm/v1/worker/gpu/model_runner.py (简化示意)
@torch.inference_mode()
def execute_model(
    self, 
    scheduler_output: "SchedulerOutput"
) -> Optional[ModelRunnerOutput]:
    """
    单次推理迭代的执行入口
    1. 更新请求状态
    2. 准备输入批次
    3. 执行模型前向传播
    4. 处理输出
    """
    # 1. 从SchedulerOutput提取请求信息
    # 2. 构建InputBatch (GPU张量)
    # 3. 调用 model.forward() 执行前向传播
    # 4. 采样并返回 ModelRunnerOutput
```



**执行流水线的关键步骤**：



1. **状态更新**：根据调度器的输出更新每个请求的状态

2. **输入准备**：将请求的Token ID、位置信息、Block Table等组装为GPU张量

3) **模型前向传播**：调用底层模型（如LLaMA、Mistral）的 `forward` 方法

4) **输出处理**：执行采样（Sampling）、获取Token ID、更新KV Cache

#### 2.3 分布式执行器（Executor）：并行策略的“指挥中心”

Executor是vLLM分布式推理的**指挥中心**，负责在单设备或多设备上执行模型。其源码位于 `vllm/v1/executor/abstract.py` 和 `vllm/executor/executor_base.py`。



**Executor的抽象层次**：



```plain&#x20;text
ExecutorBase (抽象基类)
    ↓
DistributedExecutorBase (分布式执行器抽象)
    ↓
具体实现: RayDistributedExecutor / MultiprocessingExecutor / UniProcExecutor
```



**DistributedExecutorBase的核心职责**：



* 管理分布式Worker的生命周期

* 协调多个Worker之间的模型执行

* 处理Tensor Parallelism的通信

* 处理Pipeline Parallelism的批次调度

**V1 Executor的三种实现**：



| Executor                  | 适用场景 | 特点           |
| ------------------------- | ---- | ------------ |
| `UniProcExecutor`         | 单机推理 | 单进程，适用于离线推理  |
| `MultiprocessingExecutor` | 单机多卡 | 多进程，通过NCCL通信 |
| `RayDistributedExecutor`  | 多机多卡 | 基于Ray，支持PP   |





#### 2.4 张量并行（TP）的实现

张量并行将单层权重切分到多张GPU上，通过NCCL All-Reduce通信同步中间结果。



**TP的核心实现**位于 `vllm.model_executor.layers.linear`：



```python
# vllm/model_executor/layers/linear.py (简化示意)
class ColumnParallelLinear(nn.Module):
    """列并行线性层：将权重沿列维度切分"""
    def forward(self, input_):
        # 每个GPU计算部分输出维度
        # 无需通信，直接返回分片结果
        return output_partial

class RowParallelLinear(nn.Module):
    """行并行线性层：将权重沿行维度切分"""
    def forward(self, input_):
        # 每个GPU计算部分输入维度
        # 需要All-Reduce求和
        return all_reduce(output_partial)
```



**TP在模型定义中的替换**：构建模型时，将标准 `nn.Linear` 替换为 `ColumnParallelLinear` 或 `RowParallelLinear`。



**PagedAttention Kernel中的TP支持**：PagedAttention Kernel的模板参数 `PARTITION_SIZE` 表示张量并行GPU数量。当 `PARTITION_SIZE > 0` 时，Kernel在计算注意力时考虑TP的分片布局。



#### 2.5 流水线并行（PP）的实现

流水线并行将模型的不同层分布到多张GPU上，每张GPU处理连续的若干层。



**PP在V0中的实现**：vLLM V0通过**虚拟引擎（Virtual Engine）** 支持PP——创建多个虚拟引擎匹配流水线阶段数，每个虚拟引擎拥有独立的调度器、BlockManager和Cache Engine，可同时调度多个批次。



**PP在V1中的实现**：vLLM V1通过 **RayDistributedExecutor** 支持PP：



```python
# vllm/v1/executor/ray_executor.py
class RayDistributedExecutor(Executor):
    """Ray分布式执行器，支持流水线并行"""
    supports_pp: bool = True  # 支持PP
```



PP的层切分逻辑位于 `vllm/distributed/utils.py`。启用PP时，需要设置 `--pipeline-parallel-size > 1` 并启用分块预填充 `--enable-chunked-prefill`。



**动态分块流水线并行（CPP）** ：vLLM-Ascend实现了动态分块PP，通过性能分析动态调整块大小以均衡各阶段延迟，消除流水线气泡。



#### 2.6 从Scheduler到Executor的完整调用链

```plain&#x20;text
Scheduler.schedule() 
    → 生成 SchedulerOutput
    → Executor.execute_model(SchedulerOutput)
        → Worker.execute_model()
            → GPUModelRunner.execute_model()
                → 构建 InputBatch
                → model.forward()  (GPU计算)
                → 采样生成Token
                → 返回 ModelRunnerOutput
    → Scheduler.update_from_output(ModelRunnerOutput)
```



**关键接口**：`Executor.execute_model()` 接收 `ExecuteModelRequest`，返回 `List[SamplerOutput]`。只有驱动Worker（driver worker）返回采样结果，其他Worker仅参与计算。

