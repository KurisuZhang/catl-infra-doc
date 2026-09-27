第7课的理论告诉我们：PagedAttention的核心在于将KV Cache拆分为固定大小的“块”，实现非连续的物理存储。在真正的vLLM代码中，PagedAttention kernel是用CUDA/Triton手写的，直接通过block table寻址，避免显式复制数据。但为了让你直观理解“块表（Block Table）”和“物理块存储”的交互过程，我们将用纯PyTorch构建一个教学版的PagedAttention系统。



### 一、整体架构设计

我们将构建一个简化版PagedAttention系统，包含以下组件：



1. **KV Cache Pool（显存池）**：连续分配的物理存储空间，划分为固定大小的块

2. **Block Allocator（块分配器）**：管理哪些块空闲、哪些被占用

3) **Block Table（块表）**：记录每个请求的逻辑块ID到物理块ID的映射

4) **PagedAttention Kernel（模拟）**：根据块表从物理池中读取KV并计算注意力

**场景模拟**：两个并发请求（Request A生成300 Token，Request B生成50 Token），展示如何动态分配/回收块。







### 二、核心代码实现

#### 2.1 引入依赖与参数定义

```python
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型配置（模拟LLaMA-7B风格）
NUM_HEADS = 32          # 注意力头数量
HEAD_DIM = 128          # 每个头的维度
BLOCK_SIZE = 16         # 每个块存储16个Token的KV
NUM_BLOCKS = 64         # 物理块总数（显存池大小）
DTYPE = torch.float16   # 使用FP16节省显存

# 每个块占用的显存 (2表示K和V两个矩阵)
# 2 * BLOCK_SIZE * NUM_HEADS * HEAD_DIM * 2字节(FP16) ≈ 2*16*32*128*2 = 262KB
bytes_per_block = 2 * BLOCK_SIZE * NUM_HEADS * HEAD_DIM * 2
print(f"每个物理块占用: {bytes_per_block / 1024:.1f} KB")
print(f"总KV池显存: {bytes_per_block * NUM_BLOCKS / 1024 / 1024:.1f} MB")
```



#### 2.2 构建KV Cache Pool（物理显存池）

```python
class KVCachePool:
    """模拟GPU显存池，存储所有物理块"""
    def __init__(self, num_blocks, block_size, num_heads, head_dim, dtype=torch.float16):
        # 物理存储: [num_blocks, block_size, 2, num_heads, head_dim]
        # 维度2: 0表示K, 1表示V
        self.pool = torch.zeros(
            num_blocks, block_size, 2, num_heads, head_dim,
            dtype=dtype, device=device
        )
        # 记录每个块当前存储了多少个有效Token (最后一个块可能不满)
        self.block_used_lens = torch.zeros(num_blocks, dtype=torch.long, device=device)
        self.num_blocks = num_blocks
        self.block_size = block_size
        
    def write_kv(self, block_id: int, offset: int, k: torch.Tensor, v: torch.Tensor):
        """
        将K/V写入指定物理块的指定偏移位置
        k/v: [seq_len, num_heads, head_dim]
        """
        seq_len = k.shape[0]
        # 确保不超出块边界
        assert offset + seq_len <= self.block_size, "超出块容量"
        self.pool[block_id, offset:offset+seq_len, 0] = k
        self.pool[block_id, offset:offset+seq_len, 1] = v
        # 更新有效长度
        self.block_used_lens[block_id] = max(self.block_used_lens[block_id], offset + seq_len)
        
    def read_kv(self, block_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从物理块读取K和V (返回实际有效长度)"""
        valid_len = self.block_used_lens[block_id]
        k = self.pool[block_id, :valid_len, 0]   # [valid_len, num_heads, head_dim]
        v = self.pool[block_id, :valid_len, 1]
        return k, v
```



#### 2.3 块分配器（Block Allocator）

```python
class BlockAllocator:
    """管理物理块的分配与释放（类似内存池）"""
    def __init__(self, num_blocks: int):
        # free_blocks: 可用块ID队列
        self.free_blocks = list(range(num_blocks))
        # ref_count: 每个块被多少请求引用 (用于共享前缀)
        self.ref_count = torch.zeros(num_blocks, dtype=torch.int)
        # allocated: 已分配集合 (快速查找)
        self.allocated = set()
        
    def allocate(self, num_blocks_needed: int) -> List[int]:
        """分配指定数量的物理块"""
        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(f"KV Cache不足! 需要{num_blocks_needed}块, 仅有{len(self.free_blocks)}块空闲")
        
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.allocated.add(block_id)
            self.ref_count[block_id] = 1
            allocated_blocks.append(block_id)
        return allocated_blocks
    
    def free(self, block_ids: List[int]):
        """释放物理块 (引用计数归零)"""
        for bid in block_ids:
            if bid in self.allocated:
                self.ref_count[bid] -= 1
                if self.ref_count[bid] == 0:
                    self.allocated.remove(bid)
                    self.free_blocks.append(bid)
                    
    def share_block(self, block_id: int):
        """增加引用计数 (用于前缀共享场景)"""
        if block_id in self.allocated:
            self.ref_count[block_id] += 1
    
    def num_free(self) -> int:
        return len(self.free_blocks)
```



#### 2.4 请求上下文（携带Block Table）

```python
class RequestContext:
    """每个请求的推理状态"""
    def __init__(self, request_id: str, prompt_len: int, max_new_tokens: int):
        self.request_id = request_id
        self.prompt_len = prompt_len      # 输入长度
        self.max_new_tokens = max_new_tokens
        # 当前已经计算/缓存的Token数 (包括prompt + 已生成)
        self.num_cached_tokens = 0
        # Block Table: 逻辑块索引 -> 物理块ID
        # 例如: {0: 5, 1: 12, 2: 7} 表示逻辑块0映射到物理块5
        self.block_table: Dict[int, int] = {}
        # 是否已完成生成
        self.is_finished = False
        # 生成的Token序列 (模拟)
        self.generated_tokens = []
```



#### 2.5 核心：模拟PagedAttention Kernel

这是本课的灵魂——展示如何通过**Block Table + 物理KV池**计算注意力，而不需要连续显存。



```python
def paged_attention_kernel(
    q: torch.Tensor,                # [1, num_heads, head_dim] 当前Query
    kv_pool: KVCachePool,            # 物理存储池
    block_table: Dict[int, int],     # 逻辑块->物理块映射
    total_seq_len: int               # 该请求的实际序列长度
) -> torch.Tensor:
    """
    模拟PagedAttention的前向计算:
    1. 遍历所有逻辑块 -> 根据block_table找到物理块ID
    2. 从物理块读取K/V
    3. 拼接成连续序列 (为了教学演示，实际kernel会直接做分块计算)
    """
    # 1. 根据block_table收集所有物理块
    num_blocks = len(block_table)
    k_list, v_list = [], []
    for logical_idx in sorted(block_table.keys()):
        physical_block_id = block_table[logical_idx]
        k, v = kv_pool.read_kv(physical_block_id)
        k_list.append(k)
        v_list.append(v)
    
    # 2. 沿序列维度拼接 -> 得到完整的K/V
    # k_combined: [total_seq_len, num_heads, head_dim]
    k_combined = torch.cat(k_list, dim=0)
    v_combined = torch.cat(v_list, dim=0)
    
    # 截断到实际的序列长度 (最后一个块可能不满)
    k_combined = k_combined[:total_seq_len]
    v_combined = v_combined[:total_seq_len]
    
    # 3. 标准Attention计算 (Scaling Dot-Product)
    # q: [1, num_heads, head_dim] -> [num_heads, 1, head_dim]
    q = q.squeeze(0)  # [num_heads, head_dim]
    # k: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
    k_combined = k_combined.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
    v_combined = v_combined.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
    
    # 计算注意力分数: Q @ K^T / sqrt(d)
    scores = torch.matmul(q.unsqueeze(1), k_combined.transpose(-2, -1))  # [num_heads, 1, seq_len]
    scores = scores / math.sqrt(HEAD_DIM)
    
    # Softmax + 加权求和
    attn_weights = F.softmax(scores, dim=-1)  # [num_heads, 1, seq_len]
    output = torch.matmul(attn_weights, v_combined)  # [num_heads, 1, head_dim]
    output = output.squeeze(1)  # [num_heads, head_dim]
    
    return output
```



#### 2.6 调度器：Continuous Batching的简化版

```python
class SimpleScheduler:
    """简化的连续批处理调度器 (模拟每个步骤)"""
    def __init__(self, kv_pool: KVCachePool, allocator: BlockAllocator):
        self.kv_pool = kv_pool
        self.allocator = allocator
        self.running_requests: List[RequestContext] = []
        self.waiting_requests: List[RequestContext] = []
        
    def add_request(self, req: RequestContext):
        """新请求加入等待队列"""
        self.waiting_requests.append(req)
        
    def schedule_step(self):
        """
        模拟一个解码步骤:
        1. 从等待队列中拉取新请求 (分配KV块)
        2. 对运行中的请求执行一步解码 (调用PagedAttention)
        3. 检查是否有请求完成 (释放KV块)
        """
        # ---- 步骤1: 尝试将等待队列中的请求调度到运行队列 ----
        # 限制最大并发数 (这里取3)
        while self.waiting_requests and len(self.running_requests) < 3:
            req = self.waiting_requests.pop(0)
            self._allocate_kv_blocks(req)
            self.running_requests.append(req)
            print(f"  [调度] 请求 {req.request_id} 加入运行队列, 分配 {len(req.block_table)} 个块")
        
        # ---- 步骤2: 对每个运行中的请求执行一步生成 ----
        completed_requests = []
        for req in self.running_requests:
            if req.is_finished:
                continue
            
            # 模拟解码一步: 生成一个新Token
            req.num_cached_tokens += 1
            # 模拟: 如果生成了足够的Token, 标记完成
            if req.num_cached_tokens >= req.prompt_len + req.max_new_tokens:
                req.is_finished = True
                completed_requests.append(req)
                print(f"  [完成] 请求 {req.request_id} 生成完毕! (总Token: {req.num_cached_tokens})")
                continue
            
            # ---- 模拟PagedAttention调用 (这里仅演示查表) ----
            # 实际场景中, q来自模型当前步的输出
            # 我们生成一个随机q来演示kernel调用
            q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
            
            # 调用模拟的PagedAttention Kernel
            output = paged_attention_kernel(
                q=q,
                kv_pool=self.kv_pool,
                block_table=req.block_table,
                total_seq_len=req.num_cached_tokens  # 当前已缓存的Token数
            )
            # 实际使用中, output会输入到MLP层继续生成下一Token
            # 这里仅打印占位信息
            # print(f"    请求 {req.request_id} 第 {req.num_cached_tokens} 步, 注意力输出shape: {output.shape}")
        
        # ---- 步骤3: 清理已完成请求 ----
        for req in completed_requests:
            self.running_requests.remove(req)
            self._free_kv_blocks(req)
            
    def _allocate_kv_blocks(self, req: RequestContext):
        """为请求分配KV块 (根据prompt长度)"""
        # 计算需要的块数: 向上取整
        num_blocks_needed = (req.prompt_len + req.max_new_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        # 分配物理块
        physical_blocks = self.allocator.allocate(num_blocks_needed)
        # 构建Block Table: 逻辑索引 -> 物理块ID
        req.block_table = {i: physical_blocks[i] for i in range(num_blocks_needed)}
        # 初始化第一个块写入prompt (模拟)
        # 实际场景会从模型读取K/V写入, 这里用随机数填充占位
        for logical_idx, phys_id in req.block_table.items():
            offset = 0 if logical_idx == 0 else BLOCK_SIZE
            # 填充随机KV (模拟实际写入)
            fake_k = torch.randn(BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
            fake_v = torch.randn(BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
            self.kv_pool.write_kv(phys_id, 0, fake_k, fake_v)
        req.num_cached_tokens = req.prompt_len
        
    def _free_kv_blocks(self, req: RequestContext):
        """释放请求占用的物理块"""
        phys_blocks = list(req.block_table.values())
        self.allocator.free(phys_blocks)
        print(f"  [释放] 请求 {req.request_id} 释放了 {len(phys_blocks)} 个块")
```







### 三、运行演示：模拟连续批处理过程

```python
print("\n" + "="*60)
print("【PagedAttention + Continuous Batching 模拟运行】")
print("="*60)

# 1. 初始化组件
kv_pool = KVCachePool(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, DTYPE)
allocator = BlockAllocator(NUM_BLOCKS)
scheduler = SimpleScheduler(kv_pool, allocator)

print(f"初始空闲块数: {allocator.num_free()} / {NUM_BLOCKS}")

# 2. 注入三个不同长度的请求
req1 = RequestContext("A", prompt_len=20, max_new_tokens=80)   # 总共需要 100 Token
req2 = RequestContext("B", prompt_len=50, max_new_tokens=30)   # 总共需要 80 Token
req3 = RequestContext("C", prompt_len=10, max_new_tokens=120)  # 总共需要 130 Token

scheduler.add_request(req1)
scheduler.add_request(req2)
scheduler.add_request(req3)

print("\n--- 开始模拟解码迭代 (每步调度一次) ---\n")

# 3. 模拟运行 10 个解码步骤 (每个步骤可能调度新请求, 也可能完成旧请求)
for step in range(1, 15):
    print(f"\n===== 步骤 {step} =====")
    print(f"当前运行队列大小: {len(scheduler.running_requests)}")
    print(f"当前空闲块数: {scheduler.allocator.num_free()}")
    scheduler.schedule_step()
    
    # 检查是否所有请求都已完成
    if not scheduler.running_requests and not scheduler.waiting_requests:
        print("\n所有请求处理完毕!")
        break

# 4. 最终状态
print("\n--- 最终统计 ---")
print(f"最后剩余空闲块数: {scheduler.allocator.num_free()}")
print("Block Table 演示完毕: 每个请求的KV以块为单位离散存储, 而非连续内存!")
```







### 四、进阶演示：Prefix Caching（前缀共享）

vLLM的一大亮点是**前缀缓存**：多个请求共享相同的System Prompt时，只需存一份KV。



```python
print("\n" + "="*60)
print("【进阶：前缀缓存模拟 (共享System Prompt)】")
print("="*60)

# 重置环境
kv_pool = KVCachePool(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, DTYPE)
allocator = BlockAllocator(NUM_BLOCKS)

# 假设 system prompt 长度为 48 Token (占3个完整块)
system_prompt_len = 48
system_blocks_needed = system_prompt_len // BLOCK_SIZE  # =3

# 为 system prompt 分配物理块
system_phys_blocks = allocator.allocate(system_blocks_needed)  # 假设得到 [0, 1, 2]
# 写入 system prompt 的 KV (用随机数模拟)
for i, phys_id in enumerate(system_phys_blocks):
    fake_k = torch.randn(BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    fake_v = torch.randn(BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    kv_pool.write_kv(phys_id, 0, fake_k, fake_v)

# 创建两个请求, 共享相同的 system prompt
req_share1 = RequestContext("S1", prompt_len=system_prompt_len + 10, max_new_tokens=30)
req_share2 = RequestContext("S2", prompt_len=system_prompt_len + 20, max_new_tokens=40)

# 关键: 两个请求的 block_table 前3个逻辑块指向相同的物理块 [0,1,2]
# 只有用户问题部分需要新分配物理块
req_share1.block_table = {0: 0, 1: 1, 2: 2}   # 共享系统前缀
req_share2.block_table = {0: 0, 1: 1, 2: 2}   # 完全相同的映射

# 为新请求的额外Token分配新块
extra_blocks1 = allocator.allocate(1)
req_share1.block_table[3] = extra_blocks1[0]
extra_blocks2 = allocator.allocate(2)
req_share2.block_table[3] = extra_blocks2[0]
req_share2.block_table[4] = extra_blocks2[1]

print(f"请求S1 Block Table: {req_share1.block_table}")
print(f"请求S2 Block Table: {req_share2.block_table}")
print("👉 物理块 [0, 1, 2] 被两个请求共享, 显存节省了 3块 的存储!")
print(f"物理块 [0] 的引用计数: {allocator.ref_count[0]}")
print(f"物理块 [1] 的引用计数: {allocator.ref_count[1]}")
print(f"物理块 [2] 的引用计数: {allocator.ref_count[2]}")
```







### 五、本代码与真实vLLM的差距对照表

| 本教学演示               | 真实 vLLM 实现                   | 差异说明             |
| ------------------- | ---------------------------- | ---------------- |
| 用 `torch.cat` 拼接KV块 | CUDA Kernel直接通过Block Table寻址 | 真实实现**零拷贝**，无需拼接 |
| 所有Head统一处理          | 每个Head独立并行计算                 | 真实实现利用GPU Warp并行 |
| 随机KV写入              | 从模型推理读取真实K/V                 | 演示重点在内存管理，非数值计算  |
| 纯Python调度循环         | C++/CUDA异步执行                 | 真实实现有独立的CPU调度线程  |
| 固定Block Size=16     | 可配置, 通常16或32                 | 概念一致             |
| 简单的FIFO调度           | 多级优先级 + 抢占策略                 | 真实调度更复杂          |





### 六、学习路线建议

学完本课，如果你想深入阅读 vLLM 的源码，建议按此顺序：



1. `vllm/core/block_manager.py` → 查看真实的`BlockAllocator`和`BlockTable`实现

2. `vllm/attention/backends/flash_attn.py` → 了解PagedAttention如何与FlashAttention融合

3) `vllm/core/scheduler.py` → 研究连续批处理的调度策略源码

4) `vllm/worker/cache_engine.py` → KV Cache的分配与交换逻辑

###

