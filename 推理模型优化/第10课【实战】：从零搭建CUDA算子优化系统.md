**Prefill阶段是计算密集型的**，瓶颈在算力；而**Decode阶段是访存密集型的**，瓶颈在显存带宽。矩阵乘法（GEMM）是推理引擎中最核心的算子——它在Attention和FFN中占比超过70%的总计算量。优化GEMM算子，就是优化推理引擎的“心脏”。



> ⚠️ **环境要求**：NVIDIA GPU（本教程数据基于RTX 3090测试），CUDA Toolkit 11.0+，NVIDIA驱动。完整的7阶段代码可参考 [CUDA Kernel Academy](https://github.com/AICL-Lab/cuda-kernel-academy) 和 [mini-inference-engine](https://github.com/AICL-Lab/mini-inference-engine)。
>
>



### 一、整体架构：7个阶段的优化阶梯

矩阵乘法 `C = A × B`（维度 M×K 乘 K×N 得到 M×N），我们将用**9个增量式核函数**展示优化演进。所有数据均来自NVIDIA GeForce RTX 3090（CUDA 10.2），矩阵尺寸 **5120×5120×5120**：



| 核函数           | 描述             | GFLOPS    | 相对cuBLAS (%) |
| ------------- | -------------- | --------- | ------------ |
| **cuBLAS**    | NVIDIA官方库（基准）  | 14449     | 100%         |
| **kernel\_1** | 朴素实现           | 2262      | 15.7%        |
| **kernel\_2** | 共享内存缓存         | 4217      | 29.2%        |
| **kernel\_3** | 一维Thread Tile  | 7810      | 54.1%        |
| **kernel\_4** | 二维Thread Tile  | 12251     | 84.8%        |
| **kernel\_5** | 寄存器缓存          | 12178     | 84.3%        |
| **kernel\_6** | **FLOAT4向量访存** | **13161** | **91.1%**    |
| **kernel\_7** | 双缓存预取          | 13635     | 94.4%        |





> 数据来源：CUDA SGEMM优化实践（RTX 3090，矩阵5120×5120）
>
>



**加速路径关键节点**：朴素实现（15.7%）→ 共享内存（29.2%）→ 一维Tile（54.1%）→ **二维Tile（84.8%）** → 向量化访存（91.1%）→ 双缓存预取（94.4%）。最核心的飞跃来自**二维Thread Tile优化**。



***



### 二、Kernel 1：朴素实现——性能起点

每个线程计算C矩阵中的一个元素，直接从全局内存读取A和B的对应值。



```c++
__global__ void mysgemm_v1(int M, int N, int K, float alpha, 
                           float *A, float *B, float beta, float *C) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    
    float tmp = 0.0f;
    // 每次迭代访问两次全局内存（A[gy*K+i]和B[i*N+gx]），进行一次乘加
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx];
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}
```



**性能分析**：GFLOPS约2262，仅为cuBLAS的15.7%。



**主要瓶颈**：



1. **全局内存重复访问**：同一行A的每个元素，被该行所有线程重复读取；同一列B类似。

2. **计算访存比极低**：每次迭代1次浮点乘加，对应2次全局内存读取，计算远慢于访存。

***



### 三、Kernel 2：共享内存缓存——突破内存墙

将小块A和B加载到片上共享内存（Shared Memory），然后在块内重复使用，大幅减少全局内存访问量。



```c++
template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha,
                           float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x % BLOCK_SIZE;
    int ty = threadIdx.x / BLOCK_SIZE;
    
    // 共享内存缓存A和B的小块（BLOCK_SIZE × BLOCK_SIZE）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    A += by * BLOCK_SIZE * K;
    B += bx * BLOCK_SIZE;
    C += by * BLOCK_SIZE * N + bx * BLOCK_SIZE;
    
    float tmp = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // 加载A和B的小块到共享内存
        As[ty][tx] = A[ty * K + tx];
        Bs[ty][tx] = B[ty * N + tx];
        __syncthreads();  // 等待所有线程加载完成
        
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
        
        // 在共享内存上计算
        for (int i = 0; i < BLOCK_SIZE; i++) {
            tmp += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();  // 确保计算完成后再写入新块
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
```



**关键操作**：



* `__shared__` 声明共享内存，访问延迟仅几十cycle，远低于全局内存

* `__syncthreads()` 在块内同步所有线程，确保数据加载完成

* 每个A/B元素从全局内存**只读一次**，然后在共享内存中被重复使用

**性能提升**：GFLOPS达到4217（29.2%），全局内存访存量降至原来的约1/32。



***



### 四、Kernel 3\~4：Thread Tile——让每个线程做更多计算

**核心思想**：不是每个线程只算一个元素，而是**一个线程负责多个元素的计算**，从而减少共享内存加载开销与线程同步次数。



**一维Thread Tile（kernel\_3）** ：每个线程计算一行或一列中的多个元素。



**二维Thread Tile（kernel\_4）** ：每个线程计算一个 **TM×TN** 的矩形区域（如4×4 = 16个元素）。这是**最关键的优化步骤**，将性能从54%提升到84.8%。



```c++
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha,
                           float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x, by = blockIdx.y;
    
    // 一个线程负责 TM×TN 个元素（二维tile）
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;
    
    float reg_C[TM][TN] = {0.0f};  // 寄存器中缓存输出
    float reg_A[TM];
    float reg_B[TK];
    
    for (int k = 0; k < K; k += BK) {
        // 加载A/B到共享内存（每个线程负责多个元素）
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            As[ty + i][tx] = A[(ty + i) * K + tx];
        }
        #pragma unroll
        for (int i = 0; i < TN; i++) {
            Bs[ty][tx + i] = B[ty * N + (tx + i)];
        }
        __syncthreads();
        
        // 在共享内存上计算TM×TN个结果
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_A[i] = As[ty + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_B[j] = Bs[kk][tx + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
        
        A += BK;
        B += BK * N;
    }
    
    // 写回全局内存（16个元素一起写）
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            C[(ty + i) * N + (tx + j)] = 
                alpha * reg_C[i][j] + beta * C[(ty + i) * N + (tx + j)];
        }
    }
}
```



**为什么二维Thread Tile如此有效？**



1. **指令级并行（ILP）** ：每个线程执行16次独立的乘加运算，让GPU流水线填满

2. **寄存器复用**：`reg_A[i]` 和 `reg_B[j]` 在寄存器中保存，被多次使用

3) **共享内存访问减少**：原来需要16次共享内存读取才能完成16个元素的计算，现在只需2BK次读入寄存器

**性能**：GFLOPS突破12251，达到cuBLAS的84.8%。



***



### 五、Kernel 6：FLOAT4向量化访存——榨干内存总线

在kernel\_4的基础上，利用 **FLOAT4（128位）向量类型**一次性加载4个float，将指令数量减少4倍。



```c++
__global__ void mysgemm_v6(...) {
    // 使用FLOAT4从全局内存加载
    float4 *A4 = reinterpret_cast<float4*>(A);
    float4 *B4 = reinterpret_cast<float4*>(B);
    float4 *C4 = reinterpret_cast<float4*>(C);
    
    // 一次加载4个float
    float4 a4 = A4[...];
    float4 b4 = B4[...];
    
    // 展开为单独寄存器
    float a0 = a4.x, a1 = a4.y, a2 = a4.z, a3 = a4.w;
    // ... 计算 ...
}
```



**性能**：GFLOPS达到13161，**达到cuBLAS的91.1%**。



**注意事项**：



* 向量加载要求**内存地址对齐到16字节**（FLOAT4的整数倍）

* 会增加寄存器压力，寄存器受限的内核可能不适用

***



### 六、Kernel 7：双缓存预取——让计算永不等待

用两套共享内存缓冲区实现**加载与计算的流水线重叠**：



```c++
// 双缓冲：一套用于当前计算，一套用于异步加载下一块
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];

int write_stage = 0, read_stage = 0;

// 预加载第一块
load_tile(As[0], Bs[0]);
__syncthreads();

for (int k = 0; k < K; k += BK) {
    // 异步加载下一块到后台缓冲区（不阻塞计算）
    if (k + BK < K) {
        load_tile(As[1 - read_stage], Bs[1 - read_stage]);
    }
    
    // 使用当前缓冲区计算
    compute_tile(As[read_stage], Bs[read_stage]);
    
    // 切换缓冲区
    read_stage = 1 - read_stage;
    __syncthreads();
}
```



**性能**：GFLOPS达到13635，**达到cuBLAS的94.4%**。这是**无需Tensor Core的纯CUDA核心优化上可达到的顶级水平**。



***



### 七、完整测试框架与运行

将上述所有核函数集成到主程序中测试：



```c++
// sgemm.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 声明所有核函数
extern void mysgemm_v1(...);
// ... 其他核函数

// 性能测试函数
template<typename Kernel>
void benchmark_kernel(const char* name, Kernel kernel, int M, int N, int K,
                      float *A, float *B, float *C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10;  // 平均
    
    float gflops = 2.0f * M * N * K / (ms * 1e6);
    printf("%s: %.2f ms, %.2f GFLOPS\n", name, ms, gflops);
}

int main() {
    int M = 5120, N = 5120, K = 5120;
    float *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));
    
    // 1. cuBLAS基准
    benchmark_kernel("cuBLAS", cublas_launch, M, N, K, A, B, C);
    
    // 2. 各优化阶段
    benchmark_kernel("Kernel 1 (Naive)", mysgemm_v1, M, N, K, A, B, C);
    benchmark_kernel("Kernel 2 (Shared Mem)", mysgemm_v2, M, N, K, A, B, C);
    benchmark_kernel("Kernel 4 (2D Tiling)", mysgemm_v4, M, N, K, A, B, C);
    benchmark_kernel("Kernel 6 (Float4)", mysgemm_v6, M, N, K, A, B, C);
    
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```



**编译与运行**：



```bash
nvcc -o sgemm sgemm.cu -lcublas
./sgemm
```



***



### 八、本代码与真实LLM推理引擎的关系

| 本教程组件        | 真实推理引擎中的对应物                           |
| ------------ | ------------------------------------- |
| 矩阵乘法优化（GEMM） | FlashAttention中的Q×K^T和P×V计算、FFN层全部线性层 |
| 共享内存缓存       | TensorRT-LLM的算子融合kernel               |
| FLOAT4向量化    | vLLM FlashAttention Kernel的向量内存加载     |
| 双缓存预取        | TensorRT-LLM的CUDA Graph流水线执行          |
| 自动调优         | TensorRT-LLM的kernel autotuning机制      |

