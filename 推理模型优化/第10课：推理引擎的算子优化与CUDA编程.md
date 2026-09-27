在前几课中，我们学习了推理框架（vLLM、TensorRT-LLM、SGLang）的架构设计与调度策略。这些框架通过**系统层面的优化**——PagedAttention、Continuous Batching、RadixAttention——解决了显存管理和请求调度的问题。



然而，无论调度策略多么精妙，最终每一个Token的生成都要落到**GPU上的算子（Kernel）执行**。算子优化的质量，直接决定了推理引擎的“物理极限”——在给定硬件上，推理速度能跑到多快。



如果说框架优化是“战略”，算子优化就是“战术”。一个高效的算子可以将某个操作的延迟降低数倍甚至数十倍，而一个低效的算子则可能成为整个推理管道的瓶颈。



本讲将从推理场景下的算子特点出发，深入解析FlashAttention的核心原理与演进，学习使用Triton开发自定义推理算子，并掌握CUDA Stream实现并发执行的技术。





## 一、推理场景下的算子特点：访存密集 vs. 计算密集

### 1.1 理解GPU的内存层次

要理解算子优化，首先需要理解GPU的内存层次结构：



```plain&#x20;text
寄存器 (Registers)        ← 最快，每个线程私有，KB级
    ↓
共享内存 (Shared Memory/L1 Cache) ← 次快，每个SM私有，192KB (A100)
    ↓
L2 Cache                  ← 中等，全局共享，40MB (A100)
    ↓
HBM (High Bandwidth Memory) ← 最慢，全局共享，40-80GB (A100)
```



不同层级的内存带宽差异巨大：A100的HBM带宽为1.5-2.0 TB/s，而片上SRAM的带宽估计可达19 TB/s左右，相差约**10-15倍**。



GPU编程的黄金法则是：**把数据搬到内存层次的上层，然后留在那里；除非万不得已，别回HBM**。



### 1.2 Prefill阶段：计算密集型

**Prefill阶段**处理用户的输入Prompt，需要一次性计算所有输入Token的注意力。



**特点**：



* 操作的是**大矩阵乘法**（GEMM）：`Q × K^T` 的矩阵规模为 `[seq_len, seq_len]`

* 计算量随序列长度**平方增长**：`O(seq_len²)`

* 瓶颈在**算力（FLOPS）** ，而非内存带宽

* MFU（模型浮点运算利用率）可达40%-50%

**优化方向**：使用Tensor Core加速矩阵乘法、FlashAttention减少HBM访问、更大的Batch Size。



### 1.3 Decode阶段：访存密集型

**Decode阶段**逐Token生成输出，每步只处理一个新Token。



**特点**：



* 操作的是**矩阵-向量乘法**（GEMV）：`Q × K^T` 中Q是单Token的向量

* 每生成一个Token需要**从HBM读取全部模型权重和KV Cache**

* 瓶颈在**显存带宽（HBM Bandwidth）** ，而非算力

* MFU通常只有10%-30%

**核心公式**（我们在第3课学过）：



```plain&#x20;text
Token生成速度 ≈ 显存带宽 / 模型参数大小
```



**优化方向**：KV Cache量化、FlashAttention减少HBM读写、更快的HBM（H200）、投机解码。



### 1.4 标准Attention的内存瓶颈

标准缩放点积注意力的计算过程为：



```plain&#x20;text
S = Q × K^T    → 写入HBM: N×N矩阵 (O(N²))
P = softmax(S) → 从HBM读取S，写入P
O = P × V      → 从HBM读取P，写入O
```



在这个过程中，**每个元素被访问2-4次，每次都走HBM**。当序列长度N=16K时，这个中间矩阵包含 `16,384² ≈ 2.56亿` 个元素——反复在HBM和计算单元之间搬运这些数据，是标准Attention效率低下的根本原因。



标准Attention浪费了**97%的内存流量**在中间N×N矩阵上。







## 二、FlashAttention/FlashAttention-2在推理中的应用

### 2.1 核心思想：让Attention具备“IO感知”能力

FlashAttention的核心思想是**让Attention算法“感知”到内存层次结构**，通过分块计算，避免将完整的N×N注意力矩阵写入HBM。



**具体操作**：



1. 将Q、K、V分成**小块（Tiles）** ，每块大小刚好能放入片上SRAM

2. 拿一块Q\_block，然后分块迭代K和V序列

3) 边迭代边做**在线Softmax**，同时追踪必要的统计量（最大值、和）

4) 累积输出块并在片上归一化，**只把最终结果写回HBM**

这样，注意力的**内存复杂度就从O(N²)降到了O(N)** 。



### 2.2 在线Softmax：FlashAttention最巧妙的设计

分块计算的最大挑战在于**Softmax依赖于整行的最大值和和**——而在分块计算时，我们并不知道未来块的最大值。



FlashAttention的解决方案是**在线Softmax（Online Softmax）** ：



1. 处理第一个块时，记录当前块的最大值 `m` 和和 `sum`

2. 处理后续块时，如果遇到更大的最大值，**对之前已计算的部分进行重新缩放**

3) 最终得到的结果与标准Softmax**数学上完全等价**

这种设计使得分块计算成为可能，同时保证了数值稳定性。



### 2.3 FlashAttention的演进

| 版本               | 发布时间       | 核心改进                                  | 硬件目标  |
| ---------------- | ---------- | ------------------------------------- | ----- |
| FlashAttention   | 2022年      | IO感知分块计算，O(N)内存复杂度                    | 通用GPU |
| FlashAttention-2 | 2023年7月    | 更好的并行化，减少非矩阵运算开销                      | A100  |
| FlashAttention-3 | 2025-2026年 | 针对Hopper架构优化，75% GPU利用率（740 TFLOPS/s） | H100  |
| FlashAttention-4 | 2026年3月    | Blackwell平台流水线专业化                     | B200  |





**一个反直觉的事实**：FlashAttention实际上做的**浮点运算比标准Attention更多**，但它依然更快。原因是它大幅减少了**HBM访问次数**——对于访存密集型的Attention操作，减少内存访问比减少计算量更重要。



### 2.4 FlashAttention在推理中的应用

**Prefill阶段**：FlashAttention的收益最大，因为Prefill需要处理完整的输入序列，N×N矩阵的规模最大。



**Decode阶段**：单Token解码时，注意力计算退化为 `[1, seq_len] × [seq_len, head_dim]`，FlashAttention的收益相对较小。



**实际使用**：在vLLM、TensorRT-LLM、SGLang等框架中，FlashAttention（或其变体FlashInfer）默认作为Attention后端启用。







## 三、使用Triton开发推理算子

### 3.1 为什么需要Triton？

CUDA编程虽然强大，但门槛极高。编写一个高效的CUDA Kernel需要考虑：



* 线程块（Block）和网格（Grid）的维度设计

* 共享内存（Shared Memory）的分配与使用

* Warp级别的同步与调度

* 寄存器压力的管理

Triton是OpenAI推出的**以Python为基础的GPU编程语言和编译器**，旨在**简化和优化GPU编程的复杂操作，降低高性能优化的门槛**。



Triton的核心价值：



* **门槛低**：用类Python语法编写GPU Kernel

* **高效**：编译器自动处理线程调度和内存合并

* **多平台**：从CUDA起步，现已支持AMD ROCm和Intel XPU

正如得物技术的实践总结：“Triton是起于CUDA，又不止于CUDA”。



### 3.2 Triton编程模型快速入门

**核心概念**：



* **Grid**：Kernel启动时的总线程块数，类似于CUDA的Grid

* **Block/Program**：一个线程块执行一个Triton程序实例

* **线程（Thread）** ：Triton中每个线程处理一个数据元素，通常用 `tl.arange` 生成索引

**与CUDA的对应关系**：



| CUDA概念                   | Triton概念                            |
| ------------------------ | ----------------------------------- |
| `gridDim` / `blockIdx`   | `tl.num_programs` / `tl.program_id` |
| `blockDim` / `threadIdx` | `tl.arange` + `tl.max_contiguous`   |
| 共享内存（`__shared__`）       | `tl.alloc` + `tl.zeros`/`tl.full`   |
| 内核启动配置 `<<<>>>`          | `@triton.jit` 装饰器 + 函数调用            |





### 3.3 实战：用Triton实现向量加法（Hello World）

这是Triton的入门示例，展示了一个最简单的Kernel：



```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前程序的ID（对应CUDA的blockIdx）
    pid = tl.program_id(axis=0)
    # 计算该程序负责的数据范围
    block_start = pid * BLOCK_SIZE
    # 生成该块内的索引偏移
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码，防止超出边界
    mask = offsets < n
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 计算并写回
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 确保输入在GPU上
    assert x.is_cuda and y.is_cuda
    n = x.numel()
    output = torch.empty_like(x)
    # 配置Grid大小：需要多少个Block来覆盖所有数据
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    # 启动Kernel
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output

# 测试
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
result = add(x, y)
print(torch.allclose(result, x + y))  # True
```



### 3.4 实战：用Triton实现融合Attention Kernel

FlashAttention的Triton实现是理解算子融合的最佳范例。下面展示一个简化版的融合Attention前向Kernel的核心结构：



```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim,
    stride_q_batch, stride_q_seq, stride_q_head,
    stride_k_batch, stride_k_seq, stride_k_head,
    stride_v_batch, stride_v_seq, stride_v_head,
    stride_o_batch, stride_o_seq, stride_o_head,
    BLOCK_M: tl.constexpr,   # Q的块大小
    BLOCK_N: tl.constexpr,   # K/V的块大小
):
    # 获取当前程序的ID
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    
    # 计算Q块在序列中的起始位置
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # 加载Q块到共享内存
    q_offs = (pid_batch * stride_q_batch + 
              offs_m[:, None] * stride_q_seq +
              pid_head * stride_q_head +
              tl.arange(0, BLOCK_N)[None, :] * 1)
    q = tl.load(q_ptr + q_offs)
    
    # 初始化在线Softmax的状态
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # 遍历K/V序列的所有块
    for start_n in range(0, seq_len, BLOCK_N):
        # 加载K块
        k_offs = (pid_batch * stride_k_batch +
                  (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_k_seq +
                  pid_head * stride_k_head +
                  tl.arange(0, BLOCK_N)[None, :] * 1)
        k = tl.load(k_ptr + k_offs)
        
        # 计算注意力分数: Q @ K^T
        s = tl.dot(q, tl.trans(k))
        
        # 在线Softmax更新
        m_i_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(s - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # 加载V块并累加输出
        v_offs = (pid_batch * stride_v_batch +
                  (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_seq +
                  pid_head * stride_v_head +
                  tl.arange(0, BLOCK_N)[None, :] * 1)
        v = tl.load(v_ptr + v_offs)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        m_i = m_i_new
    
    # 归一化并写回
    acc = acc / l_i[:, None]
    o_offs = (pid_batch * stride_o_batch +
              offs_m[:, None] * stride_o_seq +
              pid_head * stride_o_head +
              tl.arange(0, BLOCK_N)[None, :] * 1)
    tl.store(o_ptr + o_offs, acc.to(o_ptr.dtype.element_ty))
```



> **注意**：以上代码为教学简化版本，展示了Triton实现FlashAttention的核心结构。完整的FlashAttention-2实现包含了因果掩码、反向传播、更复杂的块大小自动调优等特性。
>
>



### 3.5 Triton的自动调优（Autotune）

Triton提供了 `@triton.autotune` 装饰器，可以自动搜索最优的块大小配置：



```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def tuned_attention_kernel(...):
    # kernel实现
    pass
```



自动调优会根据输入形状选择最优配置，无需手动调参。







## 四、CUDA Stream与并发执行

### 4.1 什么是CUDA Stream？

CUDA Stream是**GPU上按顺序执行的操作序列**。不同Stream中的操作可以**并发执行**，从而实现计算与数据传输的重叠。



**默认Stream**：所有操作在默认Stream中按顺序执行，CPU会等待每个操作完成（同步）。



**非默认Stream**：操作可以异步执行，CPU不需要等待GPU完成即可继续执行后续代码。



### 4.2 CUDA Stream的核心价值

**（1）计算与数据传输重叠**



```python
import torch

# 创建两个Stream
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 在Stream1中：将数据从CPU复制到GPU + 执行计算
with torch.cuda.stream(stream1):
    data1 = torch.randn(1000, 1000, device='cuda')
    result1 = model(data1)

# 在Stream2中：同时执行另一个独立操作
with torch.cuda.stream(stream2):
    data2 = torch.randn(1000, 1000, device='cuda')
    result2 = another_model(data2)

# 等待两个Stream完成
torch.cuda.synchronize()
```



**（2）多模型并行推理**



在同一个GPU上运行多个独立模型时，可以为每个模型分配不同的Stream，实现指令级并行。



**（3）流水线并行**



将一个大模型的不同层分配到不同的Stream，实现层间并行执行。



### 4.3 实战：双Stream并行执行

以下示例展示了如何在两个CUDA Stream中并行执行两个独立的矩阵乘法：



```python
import torch
import time

def benchmark_parallel_streams():
    # 准备数据
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    c = torch.randn(size, size, device='cuda')
    d = torch.randn(size, size, device='cuda')
    
    # ---- 串行执行 ----
    torch.cuda.synchronize()
    start = time.time()
    
    result1 = torch.matmul(a, b)
    result2 = torch.matmul(c, d)
    
    torch.cuda.synchronize()
    serial_time = time.time() - start
    print(f"串行执行时间: {serial_time:.4f}s")
    
    # ---- 并行执行（双Stream） ----
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.cuda.stream(stream1):
        result1_parallel = torch.matmul(a, b)
    
    with torch.cuda.stream(stream2):
        result2_parallel = torch.matmul(c, d)
    
    # 等待两个Stream完成
    torch.cuda.synchronize()
    parallel_time = time.time() - start
    print(f"并行执行时间: {parallel_time:.4f}s")
    print(f"加速比: {serial_time / parallel_time:.2f}x")

if __name__ == "__main__":
    benchmark_parallel_streams()
```



### 4.4 CUDA Stream在大模型推理中的应用

**vLLM中的Stream使用**：vLLM在MoE（混合专家）模型推理中，使用并行CUDA Stream同时执行共享专家和路由专家的计算。



**TensorRT-LLM中的多Stream**：TensorRT-LLM提供了多Stream工具函数，用于在CUDA Graph启用时并行执行两个函数。这种设计主要面向**低延迟场景**。



**注意力与SSM并行**：在最新的研究工作中，通过双CUDA Stream将Attention计算和SSM（状态空间模型）计算并行化，可实现端到端10-30%的速度提升。







## 五、综合实战：自定义Triton算子 + CUDA Stream

以下是一个完整的实战示例，展示如何将Triton编写的自定义算子与CUDA Stream结合使用：



```python
import torch
import triton
import triton.language as tl
import time

# ============ 1. 用Triton实现融合GeLU + 矩阵乘法 ============
@triton.jit
def fused_gelu_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 遍历K维度
    for k in range(0, K, BLOCK_K):
        # 加载A和B的块
        a_offs = offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_offs = (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptr + a_offs)
        b = tl.load(b_ptr + b_offs)
        acc += tl.dot(a, b)
    
    # 应用GeLU激活函数（融合）
    # GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.5 * acc * (1.0 + tl.tanh(0.79788456 * (acc + 0.044715 * acc * acc * acc)))
    
    # 写回结果
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + c_offs, c)

def fused_gelu_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fused_gelu_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c

# ============ 2. 使用CUDA Stream实现并行推理 ============
class ParallelInferenceEngine:
    def __init__(self, model):
        self.model = model
        self.streams = [torch.cuda.Stream() for _ in range(4)]
    
    def infer_parallel(self, inputs):
        """在多个Stream上并行推理多个输入"""
        outputs = [None] * len(inputs)
        
        # 将输入分配到不同的Stream
        for i, inp in enumerate(inputs):
            stream_idx = i % len(self.streams)
            stream = self.streams[stream_idx]
            
            with torch.cuda.stream(stream):
                # 异步地将数据移到GPU
                inp_gpu = inp.cuda(non_blocking=True)
                # 执行推理
                outputs[i] = self.model(inp_gpu)
        
        # 等待所有Stream完成
        torch.cuda.synchronize()
        return outputs

# ============ 3. 性能对比测试 ============
def benchmark():
    print("=" * 60)
    print("【性能对比：PyTorch vs Triton融合算子】")
    print("=" * 60)
    
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # PyTorch实现 (MatMul + GeLU分开)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        result_pytorch = torch.matmul(a, b)
        result_pytorch = torch.nn.functional.gelu(result_pytorch)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    print(f"PyTorch (分离执行): {pytorch_time:.4f}s")
    
    # Triton融合算子
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        result_triton = fused_gelu_matmul(a, b)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    print(f"Triton (融合执行): {triton_time:.4f}s")
    print(f"加速比: {pytorch_time / triton_time:.2f}x")
    
    # 验证正确性
    print(f"结果一致: {torch.allclose(result_pytorch, result_triton, atol=1e-3)}")

if __name__ == "__main__":
    benchmark()
```

##
