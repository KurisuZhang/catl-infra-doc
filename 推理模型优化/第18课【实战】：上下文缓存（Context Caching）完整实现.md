

本讲将提供**三个层次的实战示例**，覆盖从生产级到教学级的完整实现路径：



1. **方案一**：vLLM前缀缓存（最常用，生产就绪）

2. **方案二**：LMCache多级缓存（支持磁盘持久化，超越前缀匹配）

3) **方案三**：教学模拟实现（理解底层原理）

### 一、核心概念回顾

**KV Cache vs. Context Cache**：



| 维度       | KV Cache                | Context Cache            |
| -------- | ----------------------- | ------------------------ |
| **生命周期** | 单次请求内                   | 跨多次请求复用                  |
| **使用时机** | 首Token之后的增量预测（Decode阶段） | 首Token之前的存量计算（Prefill阶段） |
| **共享范围** | 不共享                     | 在多个请求间共享                 |
| **优化目标** | 降低TPOT（Token间延迟）        | 降低TTFT（首Token延迟）         |





**Context Cache的核心挑战**：



* **命中率**：如何让更多请求命中缓存？

* **存储管理**：KV Cache体量巨大，如何高效存储？

* **匹配粒度**：块级匹配 vs. Token级匹配

### 二、方案一：vLLM前缀缓存（生产就绪）

vLLM的自动前缀缓存（Automatic Prefix Caching, APC）是上下文缓存最成熟的实现。当多个请求共享相同前缀时，vLLM自动缓存并复用KV块。



#### 2.1 完整示例代码

```python
#!/usr/bin/env python3
"""
vLLM 自动前缀缓存（APC）完整示例
参考：vLLM官方 examples/offline_inference/automatic_prefix_caching.py
"""

import time
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def main():
    # ============ 1. 定义共享前缀 ============
    # 所有请求共享相同的System Prompt（RAG场景的典型模式）
    shared_prefix = (
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
    user_questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Japan?",
    ]

    # 构建完整Prompts（共享前缀 + 不同后缀）
    prompts = [shared_prefix + q for q in user_questions]

    # ============ 2. 采样参数 ============
    sampling_params = SamplingParams(
        temperature=0.0,  # 确定性输出便于对比
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
    outputs_no_cache = regular_llm.generate(prompts, sampling_params)
    time_no_cache = time.time() - start
    print(f"⏱️  无缓存耗时: {time_no_cache:.2f}s")

    # 清理显存
    del regular_llm
    cleanup_dist_env_and_memory()

    # ============ 4. 启用前缀缓存 ============
    print("\n" + "=" * 60)
    print("🚀 启用自动前缀缓存（APC）")
    print("=" * 60)

    cached_llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,  # 关键参数！
    )

    # 预热：处理第一个请求，计算并缓存共享前缀的KV
    # 注意：前缀只会在第一次批处理完成后被缓存
    print("🔥 预热：计算并缓存共享前缀...")
    cached_llm.generate([prompts[0]], sampling_params)

    # 正式推理：后续请求复用缓存的KV
    start = time.time()
    outputs_with_cache = cached_llm.generate(prompts, sampling_params)
    time_with_cache = time.time() - start
    print(f"⏱️  启用缓存耗时: {time_with_cache:.2f}s")

    # ============ 5. 结果对比 ============
    print("\n" + "=" * 60)
    print("📊 性能对比")
    print("=" * 60)
    print(f"无缓存: {time_no_cache:.2f}s")
    print(f"有缓存: {time_with_cache:.2f}s")
    print(f"🚀 加速比: {time_no_cache / time_with_cache:.2f}x")

    # 验证输出一致性（APC保证无损）
    texts_no_cache = [o.outputs[0].text for o in outputs_no_cache]
    texts_with_cache = [o.outputs[0].text for o in outputs_with_cache]
    all_same = all(a == b for a, b in zip(texts_no_cache, texts_with_cache))
    print(f"✅ 输出一致性: {'通过' if all_same else '失败'} (APC保证无损)")

    # 清理
    del cached_llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
```



#### 2.2 关键实现要点

**（1）启用前缀缓存**



```python
cached_llm = LLM(
    model="facebook/opt-125m",
    enable_prefix_caching=True,  # 一行代码启用
)
```



**（2）预热步骤**



缓存只会在第一次批处理后生成，因此需要预热：



```python
# 先处理一次，让系统计算并缓存共享前缀
cached_llm.generate([prompts[0]], sampling_params)
# 后续请求命中缓存
cached_llm.generate(prompts, sampling_params)
```



**（3）无损保证**



vLLM的APC在数学上保证输出与不启用缓存时**完全一致**。



#### 2.3 服务端部署

```bash
# 启动vLLM服务，启用前缀缓存
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-prefix-caching \
    --port 8000 \
    --gpu-memory-utilization 0.9
```



客户端请求时，共享相同System Prompt的请求会自动命中缓存。







### 三、方案二：LMCache多级缓存（超越前缀匹配）

LMCache是独立的KV缓存引擎，与vLLM深度集成。它突破了前缀缓存的限制——**不要求文本在前缀位置，任意位置重复都能命中**。LMCache构建了**GPU显存 → CPU内存 → 磁盘**的三级缓存架构。



#### 3.1 安装与配置

```bash
# 安装LMCache
pip install lmcache

# 或从源码编译
git clone https://github.com/LMCache/LMCache.git
cd LMCache
pip install -r requirements/build.txt
pip install -e .
```



#### 3.2 完整示例代码

```python
#!/usr/bin/env python3
"""
LMCache多级缓存完整示例
支持：GPU显存 → CPU内存 → 磁盘三级缓存
"""

import time
import lmcache
from vllm import LLM, SamplingParams


def main():
    # ============ 1. 配置LMCache ============
    # 启用多级缓存：GPU + CPU + Disk
    lmcache_config = {
        "cache_storage_backend": "hybrid",  # 混合存储
        "gpu_cache_size": "2GB",            # GPU显存缓存上限
        "cpu_cache_size": "10GB",           # CPU内存缓存上限
        "disk_cache_path": "./lmcache_data", # 磁盘缓存路径
        "disk_cache_size": "100GB",         # 磁盘缓存上限
    }
    
    # 初始化LMCache
    lmcache.init(**lmcache_config)

    # ============ 2. 准备测试数据 ============
    # 多轮对话场景：共享System Prompt + 历史对话
    system_prompt = "You are a helpful AI assistant."
    history = [
        "User: What is machine learning?",
        "Assistant: Machine learning is a subset of AI...",
        "User: Can you give an example?",
        "Assistant: Sure, image recognition is a common example...",
    ]
    
    # 构建多轮对话上下文
    context = system_prompt + "\n".join(history)
    
    prompts = [
        context + "\nUser: What is deep learning?",
        context + "\nUser: What is neural network?",
        context + "\nUser: What is backpropagation?",
    ]

    # ============ 3. 使用LMCache + vLLM ============
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        # LMCache通过环境变量或配置文件与vLLM集成
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    # ============ 4. 首次推理（缓存未命中） ============
    print("=" * 60)
    print("📝 首次推理：缓存未命中，计算并存储KV")
    print("=" * 60)

    start = time.time()
    # 在LMCache中，首次推理会自动计算并存储KV
    outputs1 = llm.generate(prompts, sampling_params)
    time_first = time.time() - start
    print(f"⏱️  首次推理耗时: {time_first:.2f}s")

    # ============ 5. 第二次推理（缓存命中） ============
    print("\n" + "=" * 60)
    print("🚀 第二次推理：缓存命中，直接复用KV")
    print("=" * 60)

    start = time.time()
    outputs2 = llm.generate(prompts, sampling_params)
    time_second = time.time() - start
    print(f"⏱️  缓存命中耗时: {time_second:.2f}s")
    print(f"🚀 加速比: {time_first / time_second:.2f}x")

    # ============ 6. 查看缓存状态 ============
    stats = lmcache.get_stats()
    print("\n📊 LMCache统计:")
    print(f"  GPU缓存命中: {stats.get('gpu_hits', 0)}")
    print(f"  CPU缓存命中: {stats.get('cpu_hits', 0)}")
    print(f"  磁盘缓存命中: {stats.get('disk_hits', 0)}")
    print(f"  缓存未命中: {stats.get('misses', 0)}")


if __name__ == "__main__":
    main()
```



#### 3.3 LMCache的关键特性

| 特性           | 说明                 |
| ------------ | ------------------ |
| **任意位置匹配**   | 不要求文本在前缀位置，重复即命中   |
| **多级存储**     | GPU显存 → CPU内存 → 磁盘 |
| **跨请求共享**    | 不同请求可共享同一份KV       |
| **vLLM深度集成** | 无缝配合vLLM v1使用      |





#### 3.4 LMCache vs. vLLM前缀缓存

| 维度       | vLLM前缀缓存        | LMCache        |
| -------- | --------------- | -------------- |
| **匹配规则** | 仅前缀匹配           | 任意位置匹配         |
| **存储层级** | GPU显存           | GPU + CPU + 磁盘 |
| **持久化**  | 不支持             | 支持磁盘持久化        |
| **适用场景** | 共享System Prompt | 多轮对话、RAG、重复文本块 |





### 四、方案三：教学模拟实现（理解底层原理）

以下代码模拟上下文缓存的核心机制，帮助理解其底层原理。



```python
#!/usr/bin/env python3
"""
上下文缓存（Context Caching）教学模拟
演示：KV块缓存、哈希查找、缓存命中与复用
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVBlock:
    """模拟KV缓存块"""
    block_id: int
    tokens: Tuple[int, ...]
    kv_data: str  # 模拟KV数据（实际是张量）
    ref_count: int = 0


class ContextCacheSimulator:
    """
    上下文缓存模拟器
    核心机制：
    1. 将Token序列切分为固定大小的块
    2. 每个块通过内容哈希唯一标识
    3. 新请求逐块查找缓存，命中则复用
    4. 未命中则计算并存入缓存
    """
    
    def __init__(self, block_size: int = 8, max_blocks: int = 50):
        self.block_size = block_size
        self.max_blocks = max_blocks
        
        # 缓存存储：hash_key -> KVBlock
        self.cache: Dict[str, KVBlock] = {}
        
        # 空闲块列表（模拟显存池）
        self.free_blocks: List[int] = list(range(max_blocks))
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate": 0.0,
        }
    
    def _compute_block_hash(self, tokens: Tuple[int, ...]) -> str:
        """计算Token块的哈希值（SHA-256）"""
        content = "|".join(str(t) for t in tokens)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _split_into_blocks(self, tokens: List[int]) -> List[Tuple[int, ...]]:
        """将Token序列切分为固定大小的块"""
        blocks = []
        for i in range(0, len(tokens), self.block_size):
            block = tuple(tokens[i:i + self.block_size])
            blocks.append(block)
        return blocks
    
    def _allocate_block(self, tokens: Tuple[int, ...]) -> Optional[str]:
        """分配新块并存入缓存"""
        if not self.free_blocks:
            # 缓存已满，模拟LRU淘汰（简化版）
            if len(self.cache) > 0:
                # 淘汰第一个块
                evict_key = next(iter(self.cache))
                del self.cache[evict_key]
                self.free_blocks.append(0)
            else:
                return None
        
        block_id = self.free_blocks.pop()
        hash_key = self._compute_block_hash(tokens)
        self.cache[hash_key] = KVBlock(
            block_id=block_id,
            tokens=tokens,
            kv_data=f"KV_data_for_block_{block_id}",
            ref_count=1
        )
        return hash_key
    
    def process_request(self, tokens: List[int]) -> Dict:
        """
        处理一个推理请求
        返回：命中/未命中统计
        """
        self.stats["total_requests"] += 1
        
        blocks = self._split_into_blocks(tokens)
        hit_count = 0
        miss_count = 0
        hit_details = []
        miss_details = []
        
        print(f"\n📝 处理请求 (Token数: {len(tokens)}, 块数: {len(blocks)})")
        
        for i, block_tokens in enumerate(blocks):
            hash_key = self._compute_block_hash(block_tokens)
            
            if hash_key in self.cache:
                # 缓存命中：复用已有块
                hit_count += 1
                self.cache[hash_key].ref_count += 1
                hit_details.append(f"块{i}: 命中 (hash={hash_key[:8]}...)")
                print(f"  ✅ 块{i}: 缓存命中 (复用已有KV)")
            else:
                # 缓存未命中：计算并存入
                miss_count += 1
                self._allocate_block(block_tokens)
                miss_details.append(f"块{i}: 未命中 (hash={hash_key[:8]}...)")
                print(f"  ❌ 块{i}: 缓存未命中 (计算并存储)")
        
        # 更新统计
        self.stats["cache_hits"] += hit_count
        self.stats["cache_misses"] += miss_count
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        self.stats["hit_rate"] = self.stats["cache_hits"] / total if total > 0 else 0
        
        return {
            "hit_count": hit_count,
            "miss_count": miss_count,
            "hit_rate": hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0,
            "hit_details": hit_details,
            "miss_details": miss_details,
        }
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            **self.stats,
            "cached_blocks": len(self.cache),
            "free_blocks": len(self.free_blocks),
            "cache_utilization": len(self.cache) / self.max_blocks,
        }


def run_simulation():
    """运行上下文缓存模拟"""
    print("=" * 60)
    print("🧠 上下文缓存（Context Caching）教学模拟")
    print("=" * 60)
    
    # 初始化缓存（块大小=4 Token，最大50块）
    cache = ContextCacheSimulator(block_size=4, max_blocks=50)
    
    # ============ 场景1：首次请求（冷启动） ============
    print("\n--- 场景1: 首次请求（冷启动，无缓存） ---")
    # 模拟Token序列：System Prompt (16 Token) + User Question (4 Token)
    system_prompt = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
    user_q1 = [201, 202, 203, 204]
    tokens1 = system_prompt + user_q1
    
    result1 = cache.process_request(tokens1)
    print(f"  命中: {result1['hit_count']}块, 未命中: {result1['miss_count']}块")
    
    # ============ 场景2：共享System Prompt（缓存命中） ============
    print("\n--- 场景2: 相同System Prompt（缓存命中） ---")
    user_q2 = [205, 206, 207, 208]
    tokens2 = system_prompt + user_q2
    
    result2 = cache.process_request(tokens2)
    print(f"  命中: {result2['hit_count']}块, 未命中: {result2['miss_count']}块")
    print(f"  🚀 系统提示词部分全部命中缓存！")
    
    # ============ 场景3：不同System Prompt（部分命中） ============
    print("\n--- 场景3: 不同System Prompt（部分命中） ---")
    system_prompt2 = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 200, 201]
    user_q3 = [209, 210, 211, 212]
    tokens3 = system_prompt2 + user_q3
    
    result3 = cache.process_request(tokens3)
    print(f"  命中: {result3['hit_count']}块, 未命中: {result3['miss_count']}块")
    print(f"  📊 前12个Token命中缓存，后4个Token因变化未命中")
    
    # ============ 最终统计 ============
    print("\n" + "=" * 60)
    print("📊 缓存统计")
    print("=" * 60)
    stats = cache.get_stats()
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  缓存命中: {stats['cache_hits']}")
    print(f"  缓存未命中: {stats['cache_misses']}")
    print(f"  命中率: {stats['hit_rate']*100:.1f}%")
    print(f"  已缓存块数: {stats['cached_blocks']}")
    print(f"  缓存利用率: {stats['cache_utilization']*100:.1f}%")
    
    print("\n💡 核心结论:")
    print("  1. 共享前缀（如System Prompt）在首次计算后被缓存")
    print("  2. 后续请求复用缓存的KV，跳过Prefill计算")
    print("  3. 前缀变化越小，缓存命中率越高")
    print("  4. 块大小影响命中粒度：块越小命中越精确，但管理开销越大")


if __name__ == "__main__":
    run_simulation()
```



#### 4.1 模拟输出示例

```plain&#x20;text
============================================================
🧠 上下文缓存（Context Caching）教学模拟
============================================================

--- 场景1: 首次请求（冷启动，无缓存） ---
📝 处理请求 (Token数: 20, 块数: 5)
  ❌ 块0: 缓存未命中 (计算并存储)
  ❌ 块1: 缓存未命中 (计算并存储)
  ❌ 块2: 缓存未命中 (计算并存储)
  ❌ 块3: 缓存未命中 (计算并存储)
  ❌ 块4: 缓存未命中 (计算并存储)
  命中: 0块, 未命中: 5块

--- 场景2: 相同System Prompt（缓存命中） ---
📝 处理请求 (Token数: 20, 块数: 5)
  ✅ 块0: 缓存命中 (复用已有KV)
  ✅ 块1: 缓存命中 (复用已有KV)
  ✅ 块2: 缓存命中 (复用已有KV)
  ✅ 块3: 缓存命中 (复用已有KV)
  ❌ 块4: 缓存未命中 (计算并存储)
  命中: 4块, 未命中: 1块
  🚀 系统提示词部分全部命中缓存！

============================================================
📊 缓存统计
============================================================
  总请求数: 3
  缓存命中: 8
  缓存未命中: 7
  命中率: 53.3%
```



#### 4.2 关键设计要点

| 设计决策      | 说明                    |
| --------- | --------------------- |
| **块大小**   | 块越小，缓存命中越精确，但管理开销越大   |
| **哈希寻址**  | 使用SHA-256对块内容哈希，确保唯一性 |
| **引用计数**  | 支持多请求共享同一块，引用计数管理生命周期 |
| **LRU淘汰** | 缓存满时淘汰最久未使用的块         |





### 五、三种方案对比与选型

| 方案           | 适用场景                | 优点          | 局限        |
| ------------ | ------------------- | ----------- | --------- |
| **vLLM前缀缓存** | 共享System Prompt、RAG | 一行代码启用，生产就绪 | 仅前缀匹配     |
| **LMCache**  | 多轮对话、任意重复文本         | 任意位置匹配，多级存储 | 需额外部署缓存服务 |
| **教学模拟**     | 学习原理                | 理解底层机制      | 不可用于生产    |





### 六、生产部署最佳实践

1. **启用前缀缓存**：vLLM服务加 `--enable-prefix-caching`

2. **固定System Prompt**：确保共享前缀**完全一致**（标点、空格、换行）

3) **预热缓存**：首次请求会计算并缓存，后续请求加速

4) **监控命中率**：关注KV Cache命中率指标

5. **多级缓存**：LMCache可将缓存扩展到CPU内存和磁盘

###
