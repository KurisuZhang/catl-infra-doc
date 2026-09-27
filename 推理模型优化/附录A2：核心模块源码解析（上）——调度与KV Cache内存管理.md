

> 📌 **源码版本说明**：本节基于vLLM V1架构（`vllm/v1/core/sched/scheduler.py`），与V0版本（`vllm/core/scheduler.py`）有较大差异。V1调度器采用**基于Token预算的统一调度模型**。
>
>



### 一、Scheduler调度器源码精读

#### 1.1 核心职责与模块定位

调度器是vLLM推理引擎的**中央决策中枢**。它的源码位于 `vllm/v1/core/sched/scheduler.py`，约**2232行**。在V1架构中，调度器运行在**Engine Core进程**内部，负责在每个调度步骤（对应一次模型前向传播）中决定：



* 哪些请求被处理

* 每个请求分配多少Token

* 何时抢占已运行的请求

* 如何管理KV Cache资源

**模块文件清单**：



| 文件                   | 行数   | 核心职责                                                             |
| -------------------- | ---- | ---------------------------------------------------------------- |
| `scheduler.py`       | 2232 | 核心调度器实现，包含调度算法、抢占逻辑、KV Connector集成                               |
| `output.py`          | 263  | 调度器输出数据结构：`SchedulerOutput`、`NewRequestData`、`CachedRequestData` |
| `interface.py`       | 244  | 调度器抽象接口 `SchedulerInterface`，定义所有调度器必须实现的协议                      |
| `request_queue.py`   | 208  | 请求队列实现：FCFS双端队列和优先级堆                                             |
| `async_scheduler.py` | 60   | 异步调度器，继承Scheduler，增加推测解码占位符机制                                    |





**V1调度器的核心设计理念**：打破传统"prefill phase"和"decode phase"的二分法。每个请求只有一个核心度量——`num_computed_tokens`（已计算Token数）vs `num_tokens_with_spec`（含推测Token的总Token数）。调度器在每个步骤中，尽可能让前者追赶后者——这统一覆盖了**分块预填充、前缀缓存、推测解码**等所有场景。



#### 1.2 关键数据结构：等待队列与运行队列

在 `scheduler.py` 的初始化方法中，调度器维护两个核心队列：



```python
# vllm/v1/core/sched/scheduler.py (约L60-L70)
class Scheduler(SchedulerInterface):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, ...):
        # 调度约束
        self.max_num_running_reqs = scheduler_config.max_num_seqs           # 默认128
        self.max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens  # 默认2048
        
        # KV Cache管理
        self.kv_cache_manager = KVCacheManager(...)
        
        # 请求队列
        self.waiting_queue = create_request_queue(...)   # 等待队列
        self.running_queue = create_request_queue(...)   # 运行队列
```



* **等待队列（**`waiting_queue`**）** ：存放等待被调度的新请求

* **运行队列（**`running_queue`**）** ：存放当前正在执行的请求

两个队列底层使用不同的数据结构：FCFS策略下使用双端队列（`deque`），优先级策略下使用最小堆。



调度器支持的两种策略通过 `SchedulerPolicy` 定义：



```python
# vllm/config/scheduler.py
SchedulerPolicy = Literal['fcfs', 'priority']
```



#### 1.3 调度流程：`Scheduler.schedule()` 方法

`schedule()` 方法是调度器的核心入口。V1调度器采用**基于Token预算**的调度模型：每一步持有固定的 `max_num_scheduled_tokens` 预算，在运行请求和等待请求之间**贪心地填充**。



**调度流程的源码结构**（基于 `scheduler.py` 约500-800行的核心逻辑）：



```python
# vllm/v1/core/sched/scheduler.py (简化示意)
def schedule(self) -> SchedulerOutput:
    # 1. 调度运行中的请求（running队列）
    scheduled_running_reqs = []
    for request in self.running_queue:
        if self._can_schedule(request):
            # 分配Token预算
            num_tokens = self._get_num_new_tokens(request)
            scheduled_running_reqs.append(request)
            self.num_scheduled_tokens -= num_tokens
    
    # 2. 调度等待中的请求（waiting队列）
    scheduled_waiting_reqs = []
    for request in self.waiting_queue:
        if self.num_scheduled_tokens <= 0:
            break
        if self._can_allocate_kv_cache(request):
            # 分配KV Cache块
            self.kv_cache_manager.allocate_slots(request)
            scheduled_waiting_reqs.append(request)
            # 更新预算
            self.num_scheduled_tokens -= request.num_tokens
    
    # 3. 构建SchedulerOutput并返回
    return SchedulerOutput(
        scheduled_running_reqs=scheduled_running_reqs,
        scheduled_waiting_reqs=scheduled_waiting_reqs,
        ...
    )
```



**关键约束条件**：



| 约束          | 源码位置                                     | 说明         |
| ----------- | ---------------------------------------- | ---------- |
| 最大运行请求数     | `self.max_num_running_reqs`              | 默认128      |
| 最大调度Token数  | `self.max_num_scheduled_tokens`          | 默认2048     |
| KV Cache可用性 | `kv_cache_manager.get_num_free_blocks()` | 检查是否有足够空闲块 |





**Prefill与Decode的混合调度**：V1调度器不再区分"prefill phase"和"decode phase"，而是通过 `num_computed_tokens` 统一处理。调度器会将请求分类为"prefill/chunked-prefill"和"decode"两组，但在同一批次中可以混合执行。



#### 1.4 抢占（Preemption）机制

当KV Cache耗尽且请求无法分配新块时，调度器执行**抢占**：



```python
# vllm/v1/core/sched/scheduler.py (抢占逻辑，约L570-L603)
def _preempt_request(self, request: Request) -> None:
    # 1. 立即释放KV Cache块
    self.kv_cache_manager.free(request)
    
    # 2. 重置num_computed_tokens为0
    request.num_computed_tokens = 0
    
    # 3. 将请求重新入队，状态设为PREEMPTED
    request.status = RequestStatus.PREEMPTED
    if self.scheduling_policy == "fcfs":
        # FCFS: 插入等待队列最前端
        self.waiting_queue.appendleft(request)
    else:
        # Priority: 按(priority, arrival_time)重新插入优先级队列
        self.waiting_queue.push(request)
```



**抢占的触发条件**：



| 策略           | 被抢占对象                              |
| ------------ | ---------------------------------- |
| **FCFS**     | 最后加入的运行中请求（last-admitted）          |
| **Priority** | 优先级最低的运行中请求（`priority`值最大，然后按到达时间） |





**V1与V0的关键差异**：



* **V0**：支持将KV Cache交换（Swap）到CPU内存

* **V1**：**不交换KV Cache到CPU**。被抢占的请求重置 `num_computed_tokens=0`，重新入队后通过**前缀缓存**或**外部匹配的KV Token**快速恢复进度

> 💡 前缀缓存使得被抢占请求的重计算代价很小——如果请求与已缓存的前缀匹配，只需计算缺失的部分。
>
>



#### 1.5 异步调度（AsyncScheduler）

`AsyncScheduler` 是 `Scheduler` 的子类，位于 `vllm/v1/core/sched/async_scheduler.py`。它实现了**CPU调度与GPU执行的流水线重叠**：



* GPU执行第N步时，CPU**推测性地调度**第N+1步

* 每个调度的解码步骤为请求添加**输出占位符**（`num_output_placeholders`）

* 占位符数量 = 1（主Token）+ 推测Token数（投机解码场景）

这种设计使调度开销被GPU计算时间掩盖，进一步提升了吞吐量。







### 二、BlockManager与KV Cache内存管理

#### 2.1 核心问题与设计目标

传统静态显存分配面临三大挑战：



1. **上下文长度爆炸**：KV Cache显存占用随上下文长度线性增长

2. **动态请求模式**：请求长度、到达时间各不相同，静态分配难以适应

3) **显存碎片化**：频繁分配释放导致碎片，即使总剩余显存充足也可能OOM

vLLM通过**Paged KV Cache**机制解决这些问题。**Block Manager正是实现这一机制的核心组件**。



#### 2.2 核心源码文件

| 文件                                           | 职责                                 |
| -------------------------------------------- | ---------------------------------- |
| `vllm/core/block_manager.py`                 | 块分配逻辑：`SelfAttnBlockSpaceManager`类 |
| `vllm/v1/core/kv_cache_manager.py`           | KV Cache管理接口                       |
| `vllm/core/block/block_table.py`             | BlockTable实现                       |
| `vllm/core/block/cpu_gpu_block_allocator.py` | CPU/GPU块分配器                        |





`SelfAttnBlockSpaceManager` 的完整定义：



```python
# vllm/core/block_manager.py (约L15-L37)
class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.
    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.
    
    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float): The threshold used for memory swapping.
        enable_caching (bool): Flag indicating whether caching is enabled.
    """
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        # ...
```



#### 2.3 逻辑块与物理块

**Block的定义**：



* 每个Block存储**固定数量Token**的Key和Value数据

* 默认 `block_size=16`（即每个Block存16个Token的K/V）

* Block在GPU显存中连续存储，但不同Block之间不要求连续

**逻辑块 vs. 物理块**：



| 概念                      | 说明             | 类比        |
| ----------------------- | -------------- | --------- |
| **逻辑块（Logical Block）**  | 请求视角的连续Token序列 | 虚拟内存的"页"  |
| **物理块（Physical Block）** | GPU显存上的离散内存块   | 物理内存的"页框" |





核心思想：**逻辑上连续，物理上离散**。一个句子的Token在物理上可能分布在不相邻的Block中，但BlockTable记录了映射关系。



#### 2.4 BlockTable：逻辑到物理的映射

**BlockTable**是维护逻辑块到物理块映射的核心数据结构：



```python
# vllm/core/block/block_table.py
class BlockTable:
    """A table mapping logical block indices to physical blocks."""
    def __init__(self, block_size: int, block_allocator: BlockAllocator, ...):
        self._blocks: List[Optional[Block]] = []  # 逻辑块号 → 物理块
    
    def allocate(self, num_blocks: int) -> List[int]:
        """分配num_blocks个物理块，返回物理块ID列表"""
        ...
    
    def append_token_ids(self, token_ids: List[int]) -> None:
        """追加Token，必要时分配新块"""
        ...
```



**BlockTable在PagedAttention中的作用**：



* PagedAttention Kernel通过BlockTable寻址，将逻辑块索引转换为物理块ID

* 物理块ID进一步映射到KV Cache池中的实际内存地址

* 整个过程对上层模型透明，模型看到的是连续的Token序列

#### 2.5 Lookahead Slots：投机解码的内存预留

Block Manager支持 **"Lookahead Slots"** ——为投机解码等场景预留的KV Cache槽位：



```python
# vllm/core/block_manager.py (注释)
# Lookahead slots are slots in the KV cache that are allocated for a sequence.
# Unlike the other allocated slots, the content of these slots is undefined --
# the worker may use the memory allocations in any way.
# In practice, a worker could use these lookahead slots to run multiple forward
# passes for a single scheduler invocation. Each successive forward pass would
# write KV activations to the corresponding lookahead slot.
```



**Lookahead Slots的用途**：



* **投机解码**：存储草稿Token的KV激活值

* **低延迟场景**：在一次调度调用中运行多次前向传播，将连续批处理的调度开销分摊到多个生成Token上

#### 2.6 调度器与BlockManager的协作

调度器与BlockManager的交互流程：



```plain&#x20;text
┌─────────────────────────────────────────────────────────────────┐
│                       Scheduler.schedule()                      │
│  1. 遍历 waiting_queue，调用 can_allocate() 检查是否有足够块    │
│  2. 如果有，调用 kv_cache_manager.allocate_slots() 分配块      │
│  3. 将请求从 waiting_queue 移至 running_queue                  │
│  4. 遍历 running_queue，为每个请求分配新Token所需的额外块      │
│  5. 如果块不足，调用 _preempt_request() 抢占低优先级请求       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      KVCacheManager                             │
│  - get_computed_blocks(): 查找前缀缓存命中的块                 │
│  - allocate_slots(): 为未缓存Token分配新块                     │
│  - free(): 释放请求占用的块                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SelfAttnBlockSpaceManager                     │
│  - 管理物理块池（Block Pool）的空闲状态                        │
│  - 执行块的分配（allocate）、释放（free）和交换（swap）        │
│  - 管理BlockTable（逻辑块→物理块映射）                         │
└─────────────────────────────────────────────────────────────────┘
```



**关键调用链**：



1. 调度器调用 `kv_cache_manager` 的方法

2. `KVCacheManager` 将实际块分配委托给 `BlockSpaceManager`（如 `SelfAttnBlockSpaceManager`）

3) `BlockSpaceManager` 通过 `BlockAllocator` 管理物理内存块

### 三、核心流程图

#### 3.1 调度器核心循环

```plain&#x20;text
while True:
    # 1. 调度决策（Scheduler.schedule()）
    scheduler_output = self.scheduler.schedule()          # 
    
    # 2. 模型执行（GPU Worker）
    model_runner_output = self.model_executor.execute_model(scheduler_output)
    
    # 3. 更新状态（Scheduler.update_from_output()）
    engine_core_outputs = self.scheduler.update_from_output(...)
```



调度器在每一步**迭代级别**做出决策——每一步对应模型的一次前向传播。调度和模型执行是**异步的**，两者并非严格一一对应。



#### 3.2 请求生命周期与队列流转

```plain&#x20;text
新请求到达 → waiting_queue
                ↓
         Scheduler.schedule()
                ↓
        分配KV Cache块
                ↓
         moving → running_queue
                ↓
         每步生成一个Token
                ↓
         Token全部生成完成
                ↓
         从running_queue移除
```



**抢占流程**：



```plain&#x20;text
running_queue中的请求
        ↓
   显存不足，需要抢占
        ↓
   _preempt_request()
        ↓
   释放KV Cache块
        ↓
   重置num_computed_tokens=0
        ↓
   状态设为PREEMPTED
        ↓
   重新插入waiting_queue最前端
```



