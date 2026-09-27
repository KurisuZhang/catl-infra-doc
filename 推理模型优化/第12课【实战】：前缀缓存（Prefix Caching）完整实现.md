##

前缀缓存（Prefix Caching）是vLLM中一项重要的优化技术，它通过缓存已处理请求的KV Cache块，并在新请求带有相同前缀时直接复用，从而**跳过重复的Prefill计算**，大幅降低首Token延迟（TTFT）。



### 一、前缀缓存的原理

#### 1.1 核心思想：KV Cache的块级复用

vLLM将KV Cache管理为固定大小的**块（Block）** ，每个块存储固定数量Token（默认16个）的Key和Value数据。前缀缓存的核心思想是：**缓存已处理请求的KV Cache块，当新请求与之前请求拥有相同前缀时，直接复用这些块**。



**一个具体示例**：



假设有三个请求：



```plain&#x20;text
请求1: "A gentle breeze stirred the leaves as children laughed in the distance"
请求2: "A gentle breeze stirred the leaves as children played in the garden"
请求3: "The sun was setting over the mountains"
```



vLLM将每个请求的Token序列切分为固定大小的块（假设块大小为4个Token）：



| 请求  | Block 1                   | Block 2                  | Block 3                   | Block 4 | Block 5 |
| --- | ------------------------- | ------------------------ | ------------------------- | ------- | ------- |
| 请求1 | "A gentle breeze stirred" | "the leaves as children" | "laughed in the distance" | -       | -       |
| 请求2 | "A gentle breeze stirred" | "the leaves as children" | "played in the garden"    | -       | -       |
| 请求3 | "The sun was setting"     | "over the mountains"     | -                         | -       | -       |





请求1首次处理时，所有Block的KV被计算并缓存。请求2到达时，前两个Block（Block 1和Block 2）**命中缓存**，直接复用，只需计算Block 3。请求3的Block 1与缓存不匹配（`"The sun was setting"` ≠ `"A gentle breeze stirred"`），**完全未命中**，需要从头计算。



#### 1.2 基于哈希的块标识

vLLM采用**基于哈希**的方法实现前缀缓存。每个KV Cache块通过以下信息计算唯一哈希值：



```plain&#x20;text
Block Hash = hash(父哈希值 + 块内Token + 额外哈希)
```



**哈希组件的详细说明**：



| 组件          | 说明                                     |
| ----------- | -------------------------------------- |
| **父哈希值**    | 前一个块的哈希值，形成**链式结构**                    |
| **块内Token** | 当前块中所有Token的元组，减少哈希冲突                  |
| **额外哈希**    | LoRA ID、多模态输入哈希、**缓存盐（cache\_salt）** 等 |





**哈希的链式结构**使得**只有完全连续的前缀才能命中缓存**——Block 3的哈希依赖于Block 1和Block 2的哈希，因此如果Block 2不同，Block 3即使内容相同也无法命中。



#### 1.3 哈希算法选择

vLLM v0.11+支持通过 `--prefix-caching-hash-algo` 控制哈希算法：



| 算法            | 说明                                 |
| ------------- | ---------------------------------- |
| `sha256`（默认）  | 使用Python pickle序列化，哈希值在不同版本间可能不可重现 |
| `sha256_cbor` | 使用cbor2序列化，**可重现、跨语言兼容**，推荐生产环境使用  |
| `xxhash`      | 更快的非密码学哈希，需安装xxhash包               |
| `xxhash_cbor` | 结合CBOR序列化和xxHash，实现可重现的快速哈希        |





> ⚠️ **安全提示**：使用非密码学安全的哈希算法（如xxhash）理论上存在哈希冲突风险，在多租户环境中可能泄漏信息。
>
>



#### 1.4 缓存隔离（Cache Salting）

为提高共享环境中的隐私性，vLLM支持通过**每个请求的缓存盐（cache\_salt）** 隔离前缀缓存重用：



* 只有**具有相同盐值**的请求才能重用缓存的KV块

* 防止基于时间的攻击（攻击者通过延迟差异推断缓存内容）

* 在信任组内部实现缓存重用，同时隔离其他请求

#### 1.5 只缓存完整块

vLLM**只缓存完整的块**（即块中所有Token都已计算完成）。如果一个块尚未填满（例如请求的最后一块），则该块不会被缓存。







### 二、实现机制：数据结构与工作流程

#### 2.1 核心数据结构

vLLM v1中用于前缀缓存的核心数据结构包括：



**（1）全局哈希表（Global Hash Table）**



```plain&#x20;text
{block_hash: KVCacheBlock}
```



将逻辑KV块映射到其哈希值，维护一个**包含所有物理块的全局哈希表**。通过这个映射，任何带有相同哈希值的新请求都能找到对应的物理块并直接复用。



**（2）BlockTracker**



位于 `vllm/core/block/prefix_caching_block.py`，跟踪每个块在缓存分配器中的状态：



| 属性              | 说明              |
| --------------- | --------------- |
| `active`        | 块是否正在被使用        |
| `last_accessed` | 最后访问时间（用于LRU淘汰） |
| `computed`      | 块是否已完全计算        |





**（3）CachedBlock**



缓存的块是**完整的块**，带有块哈希，可被用于前缀缓存。可能被运行中的请求使用，也可能在`free_block_queue`中等待被淘汰。



#### 2.2 前缀缓存的工作流程

vLLM v1中前缀缓存的**四个核心操作**：



**（1）分配（Allocate）**



当新请求到达时，调度器通过KV Cache Manager查找是否有可复用的缓存块：



1. 从请求的第一个块开始，逐块计算哈希值

2. 在全局哈希表中查找匹配的块

3) 命中则复用（引用计数+1），未命中则分配新块

**（2）追加（Append）**



当请求生成新Token时：



1. 新Token被追加到当前块

2. 如果块变满（达到`block_size`），计算该块的哈希并加入全局哈希表

3) 如果块未满，暂不缓存

**（3）释放（Free）**



当请求完成时，其占用的块被释放：



1. 块的引用计数减1

2. 如果引用计数为0，块进入淘汰候选队列

**（4）淘汰（Eviction）**



当显存不足时，采用**LRU（Least Recently Used）** 策略淘汰缓存块：



1. 选择`last_accessed`最早且引用计数为0的块

2. 从全局哈希表中移除该块的条目

3) 释放物理块供新请求使用

#### 2.3 前缀缓存与PagedAttention的协同

前缀缓存建立在**PagedAttention的分页KV Cache机制**之上：



* PagedAttention将KV Cache切分为**固定大小的块**

* 前缀缓存**复用这些块**，而不是复制数据

* 多个请求可以通过**引用计数**共享同一个物理块

这种设计使得前缀缓存**几乎零成本**——复用的块不需要额外的显存拷贝，只需增加引用计数。







### 三、配置与使用

#### 3.1 离线推理中启用

在初始化vLLM引擎时设置 `enable_prefix_caching=True`：



```python
from vllm import LLM, SamplingParams

# 启用前缀缓存
prefix_cached_llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    enable_prefix_caching=True,      # 关键参数
    gpu_memory_utilization=0.9,
)
```



#### 3.2 服务部署中启用

```bash
# 通过命令行启用
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-prefix-caching \
    --port 8000 \
    --gpu-memory-utilization 0.9
```



#### 3.3 固定前缀缓存（Pinned Prefix Caching）

vLLM V1引擎支持**固定前缀缓存**，允许将特定前缀的KV Cache“钉住”，即使在显存压力下也不会被淘汰。这在RAG或固定System Prompt的场景中特别有效。



#### 3.4 多模态输入的前缀缓存

vLLM支持多模态输入（如图像）的前缀缓存：



* 图像被替换为**占位符Token**，在Prefill期间被图像嵌入替换

* 前缀缓存的哈希包含**图像哈希**，由前端图像处理器生成

* 相同图像 + 相同文本前缀 → 命中缓存；不同图像 → 缓存未命中

```latex
Block 0 哈希 = hash(
    父哈希: None,
    Token IDs: [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, ..., <P>],
    额外哈希: <image_hash>
)
```







### 四、完整示例代码

以下是vLLM官方提供的**前缀缓存完整示例**：



```python
#!/usr/bin/env python3
"""
vLLM 自动前缀缓存（APC）完整示例
来源：vLLM官方 examples/offline_inference/prefix_caching.py
"""

import time
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def main():
    # ============ 1. 定义共享前缀 ============
    # 所有请求共享相同的System Prompt（RAG场景的典型模式）
    prefix = (
        "You are an expert school principal, skilled in effectively managing "
        "faculty and staff. Draft 10-15 questions for a potential first grade "
        "Head Teacher for my K-12, all-girls', independent school that emphasizes "
        "community, joyful discovery, and life-long learning. The candidate is "
        "coming in for a first-round panel interview for a 8th grade Math "
        "teaching role. They have 5 years of previous teaching experience "
        "as an assistant teacher at a co-ed, public school with experience "
        "in middle school math teaching. Based on these information, "
        "fulfill the following paragraph: "
    )

    # 不同的后缀（每个请求不同）
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Japan?",
    ]

    # 构建完整Prompts（共享前缀 + 不同后缀）
    generating_prompts = [prefix + prompt for prompt in prompts]

    # ============ 2. 采样参数 ============
    sampling_params = SamplingParams(
        temperature=0.0,   # 确定性输出便于对比
        max_tokens=50,
    )

    # ============ 3. 基线：无前缀缓存 ============
    print("=" * 60)
    print("📊 基线测试：不启用前缀缓存")
    print("=" * 60)

    regular_llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.4,
        enable_prefix_caching=False,
    )

    start = time.time()
    outputs_no_cache = regular_llm.generate(generating_prompts, sampling_params)
    time_no_cache = time.time() - start
    print(f"⏱️  无缓存耗时: {time_no_cache:.2f}s")

    # 保存输出用于对比
    regular_generated_texts = [o.outputs[0].text for o in outputs_no_cache]

    # 清理显存
    del regular_llm
    cleanup_dist_env_and_memory()

    # ============ 4. 启用前缀缓存 ============
    print("\n" + "=" * 60)
    print("🚀 启用自动前缀缓存（APC）")
    print("=" * 60)

    prefix_cached_llm = LLM(
        model="facebook/opt-125m",
        enable_prefix_caching=True,      # 关键参数！
        gpu_memory_utilization=0.4,
    )

    # ============ 5. 预热：计算并缓存共享前缀 ============
    # 注意：前缀只会在第一次处理完成后被缓存
    print("🔥 预热：计算并缓存共享前缀...")
    prefix_cached_llm.generate([generating_prompts[0]], sampling_params)

    # ============ 6. 正式推理：后续请求复用缓存 ============
    start = time.time()
    outputs_with_cache = prefix_cached_llm.generate(
        generating_prompts, 
        sampling_params
    )
    time_with_cache = time.time() - start
    print(f"⏱️  启用缓存耗时: {time_with_cache:.2f}s")

    cached_generated_texts = [o.outputs[0].text for o in outputs_with_cache]

    # ============ 7. 结果对比 ============
    print("\n" + "=" * 60)
    print("📊 性能对比")
    print("=" * 60)
    print(f"无缓存: {time_no_cache:.2f}s")
    print(f"有缓存: {time_with_cache:.2f}s")
    print(f"🚀 加速比: {time_no_cache / time_with_cache:.2f}x")

    # 验证输出一致性（APC保证无损）
    generated_same = all(
        regular_generated_texts[i] == cached_generated_texts[i]
        for i in range(len(prompts))
    )
    print(f"✅ 输出一致性: {'通过' if generated_same else '失败'} (APC保证无损)")


if __name__ == "__main__":
    main()
```



#### 代码关键点解析

| 步骤         | 说明                                        |
| ---------- | ----------------------------------------- |
| **共享前缀定义** | 所有请求共享相同的System Prompt，这是前缀缓存生效的前提        |
| **基线测试**   | `enable_prefix_caching=False`，每次请求都完整计算KV |
| **启用缓存**   | `enable_prefix_caching=True`，一行代码启用       |
| **预热步骤**   | 首次处理会计算并缓存共享前缀，后续请求命中缓存                   |
| **无损保证**   | APC在数学上保证输出与不启用缓存时**完全一致**                |





### 五、服务端部署示例

```bash
# 启动vLLM服务，启用前缀缓存
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-prefix-caching \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --prefix-caching-hash-algo sha256_cbor  # 推荐生产环境使用
```



客户端请求时，共享相同System Prompt的请求会自动命中缓存：



```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

system_prompt = "You are a helpful assistant. Answer concisely."

# 第一个请求：计算并缓存System Prompt
response1 = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is AI?"}
    ]
)

# 第二个请求：System Prompt完全相同，自动复用缓存！
response2 = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},  # 完全相同
        {"role": "user", "content": "What is ML?"}
    ]
)
```







### 六、源码导读

如需深入理解前缀缓存的实现，建议按以下顺序阅读源码：



| 文件                                        | 核心内容                                                        |
| ----------------------------------------- | ----------------------------------------------------------- |
| `vllm/core/block/prefix_caching_block.py` | `BlockTracker`、`ComputedBlocksTracker` —— 块状态跟踪             |
| `vllm/v1/core/block_pool.py`              | 缓存块池：`{block_hash: KVCacheBlocks}` 映射                       |
| `vllm/v1/core/kv_cache_manager.py`        | KV Cache管理：`get_computed_blocks()`、`allocate_slots()`       |
| `vllm/core/block_manager.py`              | `SelfAttnBlockSpaceManager` —— 块分配、释放、交换、前缀缓存               |
| `vllm/config.py`                          | `--enable-prefix-caching`、`--prefix-caching-hash-algo` 配置解析 |





### 七、最佳实践

| 实践要点                | 说明                              |
| ------------------- | ------------------------------- |
| **固定System Prompt** | 确保共享前缀**完全一致**（标点、空格、换行都不能变）    |
| **动态内容放末尾**         | 将静态或重复内容放在Prompt开头，动态内容放在结尾     |
| **预热缓存**            | 首次请求会计算并缓存，后续请求加速               |
| **监控命中率**           | 关注`vllm:kv_cache_usage_perc`等指标 |
| **低峰期预热**           | 在低峰期发送长Prompt有助于缓存保留，高峰期缓存更易被清理 |
| **生产环境哈希算法**        | 推荐使用`sha256_cbor`确保跨环境缓存一致性     |

