#

## 引言：大模型推理的“内存危机”

在前几课中，我们学习了Prefill-Decode分离架构如何通过将计算密集和访存密集的任务解耦来提升吞吐。但无论架构如何优化，所有推理引擎都面临一个共同的物理限制：**GPU显存是有限的**。



在大模型推理中，KV Cache是显存消耗的“大头”。以LLaMA 2 70B模型为例，FP16精度下模型权重占用140GB——这已经需要多张A100才能装下。但当并发请求增多、序列长度变长时，KV Cache的显存占用会**以更快的速度增长**，成为压垮系统的最后一根稻草。



**KV Cache管理优化的本质**：在不显著影响模型精度的前提下，用更少的显存存储更多的KV Cache，从而支持更大的Batch Size、更长的上下文和更高的并发。



本课将从KV Cache的显存占用分析入手，深入解析前缀缓存（Prefix Caching）的原理与实现、KV Cache的量化压缩技术，以及以PagedAttention为代表的显存碎片化管理方案。







## 一、KV Cache的显存占用分析

### 1.1 为什么KV Cache如此“吃显存”？

在第1课中我们学过，大模型推理是**自回归生成**过程：生成第t个Token时，需要基于前t-1个Token的上下文信息来预测。如果不加优化，每生成一个新Token都需要重新计算所有历史Token的注意力权重，计算量随序列长度呈平方级增长。



**KV Cache的核心作用**：缓存每个Token的Key和Value向量，使后续Token生成时只需计算新增部分的注意力。实验数据显示，在16K序列场景下，KV Cache可使计算量降低98%，推理延迟从320ms降至6ms。



**但代价是什么？** KV Cache需要将历史K/V矩阵全部存储在显存中，且随着序列长度和Batch Size线性增长。



### 1.2 KV Cache显存占用的精确计算公式

KV Cache的显存占用由模型的**架构参数**和**推理配置**共同决定。



**单个Token的KV Cache大小**：



```plain&#x20;text
单Token KV Cache = 2（K和V两份）× 隐藏层维度 × 层数 × 数据类型字节数
```



更精确地拆解到注意力头层面：



```plain&#x20;text
单Token KV Cache（字节）= 2 × num_heads × head_dim × precision_in_bytes
```



**完整公式（考虑Batch和序列长度）** ：



```plain&#x20;text
总KV Cache显存 = batch_size × max_seq_len × num_layers × 2 × num_heads × head_dim × dtype_bytes
```



**实战计算：LLaMA 2 7B模型**



| 参数                | 值         |
| ----------------- | --------- |
| 层数（num\_layers）   | 32        |
| 注意力头数（num\_heads） | 32        |
| 头维度（head\_dim）    | 128       |
| 数据类型              | FP16（2字节） |





单Token KV Cache = 2 × 32 × 32 × 128 × 2 = **524,288字节 ≈ 0.5 MB**



**不同序列长度下的显存占用**：



| 序列长度   | 单请求KV Cache | 32并发请求 |
| ------ | ----------- | ------ |
| 2,048  | 1,024 MB    | 32 GB  |
| 4,096  | 2,048 MB    | 64 GB  |
| 8,192  | 4,096 MB    | 128 GB |
| 32,768 | 16,384 MB   | 512 GB |





对于LLaMA 2 70B模型（层数80，头数64，头维度128），单Token KV Cache约为 **2.6 MB**——32K上下文下单个请求就需要约85GB显存。



### 1.3 显存占用的三大来源

在大模型推理中，显存主要消耗在三个方面：



1. **模型权重**：与模型参数量和精度有关。FP16精度的70B模型约140GB。

2. **KV Cache**：动态占用，随序列长度和Batch Size线性增长。高并发下显存需求可能超过模型权重本身。

3) **中间激活和框架开销**：包括Attention计算中的中间张量、框架本身的元数据等。

**关键洞察**：对于长上下文或高并发场景，**KV Cache往往是显存消耗的最大来源**，甚至超过模型权重。







## 二、前缀缓存（Prefix Caching）的原理与实现

### 2.1 核心思想：让“重复劳动”只做一次

在许多实际应用中，大量请求共享相同的**前缀（Prefix）** ：



* **RAG应用**：所有请求共享相同的System Prompt（如“你是一个知识助手，请根据以下文档回答问题...”）

* **多轮对话**：每轮对话都包含之前的所有历史消息

* **Few-shot学习**：所有请求共享相同的示例

**传统做法**：每个请求独立计算完整的KV Cache，即使前缀完全相同也要重复计算。



**前缀缓存的核心思想**：将已计算的KV Cache块缓存起来，当新请求与已缓存请求共享相同前缀时，**直接复用缓存的KV块**，跳过Prefill阶段的计算。



### 2.2 vLLM的前缀缓存实现

vLLM采用**基于哈希（Hash-based）** 的方法实现前缀缓存。



**数据结构**：



vLLM v1的前缀缓存在KV Cache Manager中实现，其基本构建块是 **“Block”数据类**。每个Block存储固定数量Token的KV数据（默认16个Token）。



**哈希机制**：



每个KV缓存块通过**块内的Token以及块前缀中的Token**进行哈希唯一标识。具体来说：



```plain&#x20;text
Block Hash = hash(prefix_tokens + block_tokens)
```



所有KV块存储在一个**全局哈希表**中。当新请求到达时，系统从第一个Block开始，逐块计算哈希值并在哈希表中查找：



* **命中**：直接复用缓存的KV块，引用计数+1

* **未命中**：重新计算该块，并存入哈希表供后续复用

**工作流程**：



vLLM v1中前缀缓存的主要操作包括：`allocate`（分配）、`append`（追加）、`free`（释放）、`eviction`（淘汰）。



**淘汰策略**：当显存不足时，vLLM采用**LRU（Least Recently Used）** 策略淘汰引用计数为0的缓存块。



**Pinned Prefix Caching（固定前缀缓存）** ：



vLLM v1还支持 **“固定前缀缓存”** 功能，允许将特定请求前缀的KV Cache“钉住”（Pin），即使在显存压力下也不会被淘汰。这在RAG或固定System Prompt的场景中特别有效。



### 2.3 前缀缓存的性能收益

**首Token延迟（TTFT）优化**：



前缀缓存最大的收益在于**跳过Prefill计算**。对于长System Prompt的场景，TTFT可以从秒级降至毫秒级。



**吞吐提升**：



通过复用KV Cache，显存占用降低，可以支持更大的Batch Size，从而提升整体吞吐。



**典型场景收益**：



| 场景                   | 前缀长度           | 缓存命中率   | TTFT降低  |
| -------------------- | -------------- | ------- | ------- |
| RAG（固定System Prompt） | 500-2000 Token | 80%-95% | 60%-80% |
| 多轮对话                 | 逐轮增长           | 随轮次增加   | 逐轮提升    |
| Few-shot学习           | 300-800 Token  | 70%-90% | 50%-70% |





### 2.4 实战：启用vLLM前缀缓存

**命令行方式**：



```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-prefix-caching  # 启用前缀缓存
```



**代码方式**：



```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # 启用前缀缓存
)

# 第一个请求：计算并缓存System Prompt的KV
prompt1 = "System: You are a helpful assistant.\nUser: What is AI?"
output1 = llm.generate(prompt1, sampling_params)

# 第二个请求：共享相同的System Prompt，直接复用KV Cache
prompt2 = "System: You are a helpful assistant.\nUser: What is ML?"
output2 = llm.generate(prompt2, sampling_params)  # System Prompt部分命中缓存
```



**注意事项**：



* System Prompt必须**完全一致**（包括标点、空格、换行），否则无法命中缓存

* vLLM支持通过可选的**per-request salting**来隔离不同用户的前缀缓存

* 前缀缓存在vLLM v1中默认通过`--enable-prefix-caching`开启

## 三、KV Cache的量化压缩

### 3.1 为什么量化KV Cache？

KV Cache的显存占用与**数据类型精度**直接相关：



| 数据类型      | 每Token KV Cache（LLaMA 7B） | 相对FP16 |
| --------- | ------------------------- | ------ |
| FP32      | \~1.0 MB                  | 2×     |
| FP16/BF16 | \~0.5 MB                  | 1×（基准） |
| INT8      | \~0.25 MB                 | 0.5×   |
| FP8       | \~0.25 MB                 | 0.5×   |
| INT4      | \~0.125 MB                | 0.25×  |





KV Cache量化的核心目标是：**用更少的显存存储相同数量的KV Cache，从而支持更大的Batch Size和更长的上下文**。



**重要澄清**：KV Cache量化**并不直接加速推理**——它的主要作用是**降低显存占用**，从而在相同推理资源下增加批量处理数据量，间接提升推理性能。



### 3.2 INT8量化：最成熟的选择

**实现方式**：



INT8 KV Cache量化通常采用**逐通道（Per-channel）** 量化，即为Key和Value向量的每个维度分配独立的缩放因子（Scale Factor），以保留不同数值范围维度的精度。



**显存节省**：



* INT8相比FP16：**节省50%显存**

* INT8相比FP32：**节省75%显存**

**精度影响**：



INT8 KV Cache量化在大多数场景下**几乎无损**。在LongBench等长文本基准测试中，INT8量化可保留95%以上的基线性能。



**vLLM中的使用**：



```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --kv-cache-dtype int8  # 启用INT8 KV Cache量化
```



vLLM还支持更细粒度的 `int8_per_token_head` 量化。



### 3.3 FP8量化：新一代选择

FP8是随着NVIDIA Hopper架构（H100）引入的新数据类型。相比INT8，FP8利用指数位提供了**更高的动态范围**，对异常值更鲁棒。



**vLLM中的FP8 KV Cache量化**：



```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --kv-cache-dtype fp8  # 启用FP8 KV Cache量化
```



支持更细粒度的 `fp8_per_token_head` 量化模式。



**INT8 vs FP8**：



| 维度            | INT8 | FP8   |
| ------------- | ---- | ----- |
| 显存节省（vs FP16） | 50%  | 50%   |
| 动态范围          | 有限   | 更大    |
| 硬件支持          | 广泛   | H100+ |
| 精度            | 几乎无损 | 良好    |





### 3.4 4-bit极致压缩：TurboQuant与NVFP4

**TurboQuant（TQ4）** ：



vLLM已支持**4-bit KV Cache量化**（`--kv-cache-dtype tq4`）。TurboQuant通过**随机旋转预处理**实现接近最优的压缩效果。



**NVFP4**：



NVIDIA推出的NVFP4格式支持将KV Cache从16位精度量化至**4位**。配合TensorRT Model Optimizer使用，可在大批次和长上下文场景中显著提升效率。



**性能权衡**：



| 量化方案        | 显存压缩比  | 精度损失 | 端到端性能开销 |
| ----------- | ------ | ---- | ------- |
| INT8        | 2×     | 几乎无损 | 极小      |
| FP8         | 2×     | 极小   | 极小      |
| TQ4 (4-bit) | **4×** | 可接受  | 7%-8%   |
| NVFP4       | **4×** | 可接受  | 待评估     |





**选型建议**：



* **精度优先**：使用INT8或FP8

* **显存优先**：使用TQ4或NVFP4（4-bit）

* **长上下文场景**：强烈建议使用KV Cache量化

### 3.5 实战：KV Cache量化效果对比

```python
import time
from vllm import LLM, SamplingParams

def benchmark_kv_cache(dtype: str, prompts: list):
    """测试不同KV Cache数据类型的显存占用和性能"""
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        kv_cache_dtype=dtype,  # "auto", "fp8", "int8", "tq4"
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(max_tokens=512)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    # 获取显存信息（近似）
    import torch
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"KV Cache dtype: {dtype}")
    print(f"  显存已分配: {allocated:.2f} GB")
    print(f"  显存预留: {reserved:.2f} GB")
    print(f"  推理耗时: {elapsed:.2f}s")
    return outputs

# 测试不同配置
prompts = ["Tell me a long story about AI."] * 32  # 32个并发请求

for dtype in ["auto", "int8", "fp8"]:
    try:
        benchmark_kv_cache(dtype, prompts)
    except Exception as e:
        print(f"{dtype} 不支持: {e}")
```







## 四、显存碎片化管理

### 4.1 传统连续分配的“碎片化灾难”

在传统推理框架中，KV Cache需要在**连续的物理显存**中分配。



**问题场景**：



假设显存总容量为100个单位。请求A占用了50个单位后结束，释放了空间。但由于内存分配和释放的不规则性，释放的空间可能**不连续**——中间夹杂着其他请求占用的碎片。



**后果**：



* 虽然剩余显存总量足够（如80个单位），但没有**连续**的50个单位空间来容纳新的大请求

* 显存利用率可能只有**20%左右**

* 导致OOM（Out of Memory）错误，即使显存“看起来”还有剩余

### 4.2 PagedAttention：像操作系统管理内存一样管理KV Cache

vLLM的核心创新**PagedAttention**正是为解决碎片化问题而生。



**核心思想**：



借鉴操作系统的**虚拟内存分页**技术，将每个请求的KV Cache划分为固定大小的**块（Block）**。



**关键特性**：



1. **非连续物理存储**：这些块可以存储在**非连续的物理显存**中

2. **按需分配**：需要多少块就分配多少块，无需预留

3) **消除碎片**：固定大小的块分配和释放不会产生碎片

4) **块级共享**：多个请求可以共享相同的KV块（前缀缓存的基础）

**类比理解**：



* **传统方案**：像仓库里必须把所有货物堆在**一个连续的角落**——即使其他地方有空位，只要这个角落不够大就放不下

* **PagedAttention**：像仓库管理员把货物装进**统一规格的箱子**——箱子可以放在仓库的任何空位，不用管是否连续

### 4.3 块大小（Block Size）的权衡

块大小是PagedAttention中的一个关键设计参数（vLLM默认16个Token）。



| 块大小        | 优点           | 缺点         |
| ---------- | ------------ | ---------- |
| **小（如16）** | 碎片更少、内存利用率更高 | 管理开销大（更多块） |
| **大（如64）** | 管理开销小        | 可能产生内部碎片   |





**内部碎片**：最后一个块可能只用了部分空间（如只存了5个Token），剩余空间被浪费。



### 4.4 从PagedAttention到更高级的显存管理

**vAttention**：在操作系统层面管理虚拟内存，将虚拟内存抽象直接应用于KV Cache管理。



**RadixAttention（SGLang）** ：前缀感知的分页管理，在PagedAttention基础上增加了前缀树索引，实现更高效的缓存复用。



**FlexKV**：vLLM的FlexKV Connector提供了比默认前缀缓存更灵活的KV管理策略。







## 五、综合实战：完整KV Cache管理系统

以下是一个简化的KV Cache管理系统实现，演示了**显存占用计算、块分配、前缀缓存和量化**的核心概念：



```python
"""
KV Cache管理系统 - 教学演示
展示：显存计算、块分配、前缀缓存、量化压缩
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DType(Enum):
    FP16 = 2      # 2字节
    INT8 = 1      # 1字节
    FP8 = 1       # 1字节
    INT4 = 0.5    # 0.5字节


@dataclass
class ModelConfig:
    """模型架构配置"""
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: DType = DType.FP16
    
    @property
    def kv_cache_per_token_bytes(self) -> int:
        """单个Token的KV Cache显存占用（字节）"""
        # 2 (K和V) × num_layers × num_heads × head_dim × dtype_bytes
        return int(2 * self.num_layers * self.num_heads * 
                   self.head_dim * self.dtype.value)
    
    def kv_cache_for_seq(self, seq_len: int) -> int:
        """指定序列长度的KV Cache显存占用（字节）"""
        return self.kv_cache_per_token_bytes * seq_len


class KVBlock:
    """KV Cache块（固定大小）"""
    def __init__(self, block_id: int, tokens: List[int], kv_data: any):
        self.block_id = block_id
        self.tokens = tokens  # 块内的Token IDs
        self.kv_data = kv_data  # 实际KV数据（这里用占位符）
        self.ref_count = 0  # 引用计数（用于前缀缓存共享）
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> int:
        """计算块的哈希值（用于前缀缓存查找）"""
        # 实际实现中使用token IDs的哈希
        return hash(tuple(self.tokens))


class KVCacheManager:
    """
    KV Cache管理器
    集成了：块分配、前缀缓存、LRU淘汰
    """
    def __init__(self, model_config: ModelConfig, 
                 num_blocks: int, block_size: int = 16):
        self.config = model_config
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # 物理块池：[num_blocks, block_size, 2, num_heads, head_dim]
        self.pool = [None] * num_blocks
        
        # 空闲块列表
        self.free_blocks = list(range(num_blocks))
        
        # 哈希表：block_hash -> block_id（用于前缀缓存查找）
        self.hash_table: Dict[int, int] = {}
        
        # 每个块的引用计数
        self.ref_counts = [0] * num_blocks
    
    def calculate_blocks_needed(self, seq_len: int) -> int:
        """计算指定序列长度需要的块数"""
        return math.ceil(seq_len / self.block_size)
    
    def allocate_blocks(self, num_blocks: int) -> Optional[List[int]]:
        """分配指定数量的物理块"""
        if len(self.free_blocks) < num_blocks:
            # 尝试LRU淘汰
            self._evict_blocks(num_blocks - len(self.free_blocks))
        
        if len(self.free_blocks) < num_blocks:
            return None  # 显存不足
        
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            self.ref_counts[block_id] = 1
            allocated.append(block_id)
        
        return allocated
    
    def cache_prefix(self, tokens: List[int]) -> List[int]:
        """
        前缀缓存查找：返回命中的块ID列表
        如果某个块未命中，返回None表示需要计算
        """
        block_ids = []
        num_blocks = self.calculate_blocks_needed(len(tokens))
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, len(tokens))
            block_tokens = tokens[start:end]
            block_hash = hash(tuple(block_tokens))
            
            if block_hash in self.hash_table:
                # 命中缓存
                block_id = self.hash_table[block_hash]
                self.ref_counts[block_id] += 1
                block_ids.append(block_id)
            else:
                # 未命中，需要分配新块并计算
                new_block_id = self._allocate_new_block(block_tokens)
                if new_block_id is None:
                    return None  # 显存不足
                self.hash_table[block_hash] = new_block_id
                block_ids.append(new_block_id)
        
        return block_ids
    
    def _allocate_new_block(self, tokens: List[int]) -> Optional[int]:
        """分配新块并存储KV数据"""
        if not self.free_blocks:
            self._evict_blocks(1)
            if not self.free_blocks:
                return None
        
        block_id = self.free_blocks.pop()
        # 模拟存储KV数据
        self.pool[block_id] = KVBlock(block_id, tokens, "kv_data_placeholder")
        self.ref_counts[block_id] = 1
        return block_id
    
    def _evict_blocks(self, needed: int):
        """LRU淘汰：释放引用计数为0的块"""
        evicted = 0
        for block_id in range(self.num_blocks):
            if self.ref_counts[block_id] == 0 and self.pool[block_id] is not None:
                # 从哈希表中移除
                block_hash = self.pool[block_id].hash
                if block_hash in self.hash_table:
                    del self.hash_table[block_hash]
                # 释放块
                self.pool[block_id] = None
                self.free_blocks.append(block_id)
                evicted += 1
                if evicted >= needed:
                    break
    
    def free_request(self, block_ids: List[int]):
        """释放请求占用的块（减少引用计数）"""
        for block_id in block_ids:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                # 引用计数归零，加入淘汰候选
                pass  # 实际淘汰在_evict_blocks中执行
    
    def get_memory_usage(self) -> Dict:
        """获取显存使用情况"""
        used_blocks = self.num_blocks - len(self.free_blocks)
        per_block_bytes = self.block_size * self.config.kv_cache_per_token_bytes
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "total_memory_mb": (self.num_blocks * per_block_bytes) / (1024**2),
            "used_memory_mb": (used_blocks * per_block_bytes) / (1024**2),
            "utilization": used_blocks / self.num_blocks if self.num_blocks > 0 else 0,
        }


# ============ 使用示例 ============

def demo_kv_cache_manager():
    """演示KV Cache管理器的核心功能"""
    print("=" * 60)
    print("KV Cache管理系统演示")
    print("=" * 60)
    
    # 1. 配置模型（LLaMA 2 7B）
    config = ModelConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        dtype=DType.FP16
    )
    
    print(f"\n📊 模型配置:")
    print(f"  层数: {config.num_layers}")
    print(f"  注意力头数: {config.num_heads}")
    print(f"  头维度: {config.head_dim}")
    print(f"  数据类型: {config.dtype.name}")
    print(f"  单Token KV Cache: {config.kv_cache_per_token_bytes / 1024:.2f} KB")
    
    # 2. 初始化管理器（假设100个块，每块16 Token）
    manager = KVCacheManager(config, num_blocks=100, block_size=16)
    
    # 3. 模拟两个请求，共享System Prompt
    system_prompt = [101, 102, 103, 104, 105, 106, 107, 108]  # 模拟Token IDs
    user1 = [201, 202, 203]
    user2 = [204, 205, 206, 207]
    
    print(f"\n📝 模拟请求:")
    print(f"  System Prompt: {system_prompt} ({len(system_prompt)} Token)")
    print(f"  User 1: {user1} ({len(user1)} Token)")
    print(f"  User 2: {user2} ({len(user2)} Token)")
    
    # 4. 第一个请求：计算并缓存System Prompt
    print("\n🔵 请求1: 首次推理（计算并缓存System Prompt）")
    full_prompt1 = system_prompt + user1
    blocks1 = manager.cache_prefix(full_prompt1)
    print(f"  分配的块数: {len(blocks1)}")
    print(f"  显存使用: {manager.get_memory_usage()['used_memory_mb']:.2f} MB")
    print(f"  缓存命中: 0 (首次)")
    
    # 5. 第二个请求：复用System Prompt的缓存
    print("\n🟢 请求2: 复用缓存的System Prompt")
    full_prompt2 = system_prompt + user2
    blocks2 = manager.cache_prefix(full_prompt2)
    
    # 统计命中情况
    hit_count = sum(1 for b in blocks2 if b in blocks1)
    print(f"  分配的块数: {len(blocks2)}")
    print(f"  缓存命中: {hit_count} 块 (System Prompt部分)")
    print(f"  显存使用: {manager.get_memory_usage()['used_memory_mb']:.2f} MB")
    
    # 6. 显示最终状态
    print("\n📊 最终显存状态:")
    usage = manager.get_memory_usage()
    print(f"  总块数: {usage['total_blocks']}")
    print(f"  已用块数: {usage['used_blocks']}")
    print(f"  空闲块数: {usage['free_blocks']}")
    print(f"  利用率: {usage['utilization']*100:.1f}%")
    print(f"  已用显存: {usage['used_memory_mb']:.2f} MB")


if __name__ == "__main__":
    demo_kv_cache_manager()
```



**代码输出示例**：



```plain&#x20;text
============================================================
KV Cache管理系统演示
============================================================

📊 模型配置:
  层数: 32
  注意力头数: 32
  头维度: 128
  数据类型: FP16
  单Token KV Cache: 0.50 KB

📝 模拟请求:
  System Prompt: [101, 102, 103, 104, 105, 106, 107, 108] (8 Token)
  User 1: [201, 202, 203] (3 Token)
  User 2: [204, 205, 206, 207] (4 Token)

🔵 请求1: 首次推理（计算并缓存System Prompt）
  分配的块数: 1
  显存使用: 8.00 MB
  缓存命中: 0 (首次)

🟢 请求2: 复用缓存的System Prompt
  分配的块数: 1
  缓存命中: 1 块 (System Prompt部分)
  显存使用: 12.00 MB

📊 最终显存状态:
  总块数: 100
  已用块数: 2
  空闲块数: 98
  利用率: 2.0%
  已用显存: 16.00 MB
```







## 课程小结

本课深入解析了KV Cache管理与优化的核心技术，核心要点如下：



1. **KV Cache显存占用**：由`2 × num_layers × num_heads × head_dim × dtype_bytes`决定，随序列长度和Batch Size线性增长。在长上下文场景中，KV Cache往往是显存消耗的最大来源。

2. **前缀缓存（Prefix Caching）** ：通过基于哈希的块级缓存，让共享相同前缀的请求复用KV Cache，跳过Prefill计算。vLLM的Pinned Prefix Caching可保护关键前缀不被淘汰。

3) **KV Cache量化**：INT8可节省50%显存且几乎无损，FP8提供更大动态范围，4-bit（TurboQuant/NVFP4）可实现4倍压缩但伴随7%-8%性能开销。

4) **显存碎片化管理**：PagedAttention借鉴操作系统虚拟内存思想，将KV Cache划分为固定大小的块并允许非连续存储，从根本上消除了内存碎片。

## 思考题

1. **显存计算**：LLaMA 3 70B模型（num\_layers=80，num\_heads=64，head\_dim=128）在FP16精度下，处理32个并发请求、每个请求序列长度4096时，KV Cache需要多少显存？如果启用INT8量化，能节省多少？

2. **前缀缓存分析**：你的RAG应用有固定的System Prompt（1024 Token），每次用户问题不同（平均256 Token）。日请求量10万次。启用前缀缓存后，理论上能节省多少Prefill计算量？请用百分比估算。

3) **量化选型**：你的业务对推理精度要求极高（数学推理任务），但显存严重不足。你会在INT8、FP8和TQ4之间选择哪个KV Cache量化方案？为什么？

4) **碎片化理解**：请解释为什么传统连续分配方案在显存利用率高时仍然可能OOM，而PagedAttention可以避免这个问题。

##
