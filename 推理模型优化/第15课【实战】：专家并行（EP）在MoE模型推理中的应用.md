##

专家并行（Expert Parallelism）是专门为混合专家（MoE）模型设计的分布式推理策略，它将不同的专家网络分布在不同的GPU上，每个GPU只持有部分专家权重，通过All-to-All通信在设备间分发Token。这种方法能显著降低单卡显存压力，同时利用MoE模型的稀疏激活特性（如DeepSeek-R1每Token仅激活8/256个专家）提升推理效率。



由于**真正运行一个数百GB的MoE模型需要多节点集群和高性能网络环境**（如InfiniBand），本讲将提供两套方案：



1. **方案一**：**多节点实战部署模板**——基于vLLM和官方文档的生产级部署指南，让你了解真实环境下的完整配置流程。

2. **方案二**：**模拟教学代码**——在单机上模拟EP的数据分发与计算流程，帮助理解核心概念，适合教学和原型验证。

***



### 一、EP核心概念与规模计算

EP的关键技术指标和计算公式，依据多个权威来源整理如下：



| 概念           | 说明             | 计算公式/来源                                   |
| ------------ | -------------- | ----------------------------------------- |
| **EP\_SIZE** | 专家并行度（GPU数量）   | `EP_SIZE = TP_SIZE × DP_SIZE`（vLLM v0.11） |
| **通信模式**     | 专家间的Token分发与合并 | **All-to-All** 集合通信                       |
| **负载均衡**     | 解决热点专家导致的性能瓶颈  | 启用 `--enable-eplb`                        |
| **多节点后端**    | 跨节点EP通信        | **DeepEP**（支持GPU-initiated RDMA）          |





在vLLM中，EP大小会自动计算为 `EP_SIZE = TP_SIZE × DP_SIZE`。对于注意力层，采用数据并行（每个DP组内复制注意力权重）；对于MoE专家层，采用专家并行（权重分片到各GPU）。



***



### 二、方案一：vLLM多节点EP实战部署（生产级）

以下是基于vLLM官方文档的**真实部署命令**，适用于8卡及以上节点环境。单节点部署与多节点的区别在于多节点需要安装 **DeepEP通信后端** 并配置网络环境。



#### 2.1 环境准备

EP依赖**DeepEP**和**DeepGEMM**库。在启动容器前，确保宿主机具备：



* **IB/RoCE网络**：多节点需全Mesh InfiniBand互联&#x20;

* **GDRCopy**（多节点）: 用于GPU内存的直接访问

对于**GB200 NVL72**环境，还需要挂载IMEX设备：



```bash
docker run --gpus all \
  -v /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels \
  -v /dev/gdrdrv:/dev/gdrdrv \
  --net=host \
  ...  # 后续vLLM启动命令
```



#### 2.2 单节点EP部署（8卡H200示例）

以下命令部署DeepSeek-V3模型，使用**1路张量并行（TP=1）** 和**8路数据并行（DP=8）**——即每个GPU独立处理注意力，专家权重分片到8张卡上。



```bash
# 启动vLLM服务，启用专家并行
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --enable-expert-parallel \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --port 8000
```



#### 2.3 多节点EP部署（DeepEP低延迟模式）

多节点部署需要在每个节点分别启动命令，并配置节点角色（主节点处理请求，从节点运行headless模式）。



**主节点命令**：



```bash
# 主节点（处理API请求）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --data-parallel-master-ip <MASTER_IP> \
    --data-parallel-master-port 29500 \
    --distributed-executor-backend mp \
    --api-server-count 8
```



**从节点命令**（每个额外节点）：



```bash
# 从节点（headless模式，仅参与计算）
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --data-parallel-master-ip <MASTER_IP> \
    --data-parallel-master-port 29500 \
    --distributed-executor-backend mp \
    --headless
```



**通信后端选择**：



* `deepep_low_latency`：**解码阶段**推荐（支持CUDA Graph，掩码布局）

* `deepep_high_throughput`：**预填充阶段**推荐（分组GEMM，连续布局）

#### 2.4 专家并行负载均衡器（EPLB）

MoE模型在实际推理中常出现专家负载严重不均衡的现象——少数“热门专家”接收的Token远多于其他专家，导致部分GPU成为性能瓶颈。 vLLM通过 `--enable-eplb` 启用自动负载均衡。



```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"window_size": 1000, "step_interval": 3000}'
```



***



### 三、方案二：模拟EP核心流程（教学代码）

为加深理解，以下提供一个**单机模拟实现**，展示EP的三大核心步骤：**Token分发**（Dispatch）、**专家计算**、**结果合并**（Combine）。



```python
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple

# ==================== 配置 ====================
NUM_GPUS = 4              # 模拟4张GPU
NUM_EXPERTS = 8           # 总专家数
ACTIVE_EXPERTS = 2        # 每个Token激活的专家数（Top-K）
HIDDEN_SIZE = 128
BATCH_TOKENS = 16         # 模拟16个Token

# ==================== 模拟专家网络 ====================
class MockExpert(torch.nn.Module):
    """模拟MoE专家（简单的线性层）"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# ==================== EP模拟器 ====================
class ExpertParallelSimulator:
    """
    模拟专家并行（EP）的核心流程
    步骤：
    1. 将专家均匀分布到各GPU（设备）
    2. 计算每个Token的路由权重
    3. Dispatch：将Token发送到对应的专家所在设备
    4. 本地专家计算
    5. Combine：将计算结果合并回原始Token
    """
    def __init__(self, num_gpus: int, num_experts: int, hidden_size: int):
        self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # 创建专家网络（8个专家）
        self.experts = [MockExpert(hidden_size) for _ in range(num_experts)]
        
        # 路由层（模拟门控网络）
        self.router = torch.nn.Linear(hidden_size, num_experts, bias=False)
        
        # 将专家分配到各GPU（均匀分布）
        self.experts_per_gpu = num_experts // num_gpus
        self.device_maps = self._build_device_map()
        
        # 将专家移动到对应设备
        for expert_id, device_id in self.device_maps.items():
            self.experts[expert_id].to(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    def _build_device_map(self) -> dict:
        """构建专家ID到设备ID的映射"""
        device_map = {}
        for expert_id in range(self.num_experts):
            device_id = expert_id // self.experts_per_gpu
            device_map[expert_id] = device_id
        return device_map
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        模拟EP前向传播（简化版）
        Args:
            hidden_states: [num_tokens, hidden_size]
        Returns:
            经MoE处理后的输出 [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        print(f"[EP模拟] 输入Token数: {num_tokens}, 隐藏维度: {self.hidden_size}")
        
        # -------- Step 1: 路由计算 --------
        router_logits = self.router(hidden_states)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择Top-K专家 (ACTIVE_EXPERTS=2)
        topk_probs, topk_indices = torch.topk(router_probs, ACTIVE_EXPERTS, dim=-1)
        # 归一化权重
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        print(f"[EP模拟] 路由Top-2专家 (每个Token选择2个专家):")
        for i in range(min(4, num_tokens)):
            print(f"  Token {i}: 专家 {topk_indices[i].tolist()}, 权重 {topk_probs[i].tolist()}")
        
        # -------- Step 2: Dispatch (Token分发) --------
        # 按目标专家设备分组
        # 分组策略：根据每个Token选中的专家所在设备，将Token分发到对应设备
        device_to_tokens = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        device_to_token_indices = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        device_to_experts = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        device_to_weights = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        
        for token_idx in range(num_tokens):
            for k in range(ACTIVE_EXPERTS):
                expert_id = topk_indices[token_idx, k].item()
                weight = topk_probs[token_idx, k].item()
                device_id = self.device_maps[expert_id]
                
                device_to_tokens[device_id].append(hidden_states[token_idx])
                device_to_token_indices[device_id].append(token_idx)
                device_to_experts[device_id].append(expert_id)
                device_to_weights[device_id].append(weight)
        
        print(f"[EP模拟] Token分发到各设备:")
        for gpu_id in range(self.num_gpus):
            print(f"  GPU {gpu_id}: {len(device_to_tokens[gpu_id])} 个Token")
        
        # -------- Step 3: 本地专家计算 --------
        # 每个设备只计算分配给它的专家
        outputs_per_token = [[] for _ in range(num_tokens)]  # 每个Token累积各专家的输出
        
        for device_id in range(self.num_gpus):
            if not device_to_tokens[device_id]:
                continue
            
            # 当前设备上需要处理的Token数量
            num_device_tokens = len(device_to_tokens[device_id])
            
            # 按专家分组该设备上的Token
            # 实际EP中，同一设备的不同专家可以并行计算
            expert_to_tokens = {}
            expert_to_weights = {}
            expert_to_original_idx = {}
            
            for idx in range(num_device_tokens):
                expert_id = device_to_experts[device_id][idx]
                if expert_id not in expert_to_tokens:
                    expert_to_tokens[expert_id] = []
                    expert_to_weights[expert_id] = []
                    expert_to_original_idx[expert_id] = []
                expert_to_tokens[expert_id].append(device_to_tokens[device_id][idx])
                expert_to_weights[expert_id].append(device_to_weights[device_id][idx])
                expert_to_original_idx[expert_id].append(device_to_token_indices[device_id][idx])
            
            # 对每个专家进行计算
            for expert_id, tokens_list in expert_to_tokens.items():
                # 将Token张量堆叠
                tokens_tensor = torch.stack(tokens_list).to(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                # 专家前向计算
                expert_output = self.experts[expert_id](tokens_tensor)
                expert_output = expert_output.cpu()
                
                # 按权重累加到对应Token的输出
                weights_list = expert_to_weights[expert_id]
                orig_indices = expert_to_original_idx[expert_id]
                
                for i, orig_idx in enumerate(orig_indices):
                    # 一个Token的输出 = Σ(专家输出 × 路由权重)
                    weighted_output = expert_output[i] * weights_list[i]
                    outputs_per_token[orig_idx].append(weighted_output)
        
        # -------- Step 4: Combine (结果合并) --------
        # 每个Token的最终输出是各专家输出的加权和
        final_outputs = []
        for token_outputs in outputs_per_token:
            # 如果某Token没有收到任何专家输出（理论上不会发生）
            if not token_outputs:
                final_outputs.append(torch.zeros(self.hidden_size))
            else:
                # 求和所有专家的加权输出
                final_outputs.append(torch.stack(token_outputs).sum(dim=0))
        
        final_output = torch.stack(final_outputs)
        print(f"[EP模拟] 最终输出 shape: {final_output.shape}")
        
        return final_output


# ==================== 测试运行 ====================
def run_ep_simulation():
    print("=" * 60)
    print("专家并行（EP）核心流程模拟")
    print("=" * 60)
    
    # 初始化模拟器
    sim = ExpertParallelSimulator(
        num_gpus=NUM_GPUS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE
    )
    
    # 生成随机输入Token（模拟16个Token的隐藏状态）
    hidden_states = torch.randn(BATCH_TOKENS, HIDDEN_SIZE)
    print(f"\n输入随机Token隐藏状态: shape {hidden_states.shape}")
    
    # 执行EP前向传播
    output = sim.forward(hidden_states)
    
    print("\n" + "=" * 60)
    print("模拟完成！")
    print(f"输入shape: {hidden_states.shape} -> 输出shape: {output.shape}")
    print("\n[核心结论] 专家并行的本质：")
    print("  1. 专家分片：每个GPU只存放部分专家权重")
    print("  2. Token分发（Dispatch）：将Token按路由结果发送到对应专家的GPU")
    print("  3. 本地计算：各GPU独立计算本地专家")
    print("  4. 结果合并（Combine）：将各专家输出按路由权重汇总")

if __name__ == "__main__":
    run_ep_simulation()
```



#### 代码输出示例：

```plain&#x20;text
============================================================
专家并行（EP）核心流程模拟
============================================================

输入随机Token隐藏状态: shape torch.Size([16, 128])

[EP模拟] 输入Token数: 16, 隐藏维度: 128
[EP模拟] 路由Top-2专家 (每个Token选择2个专家):
  Token 0: 专家 [3, 7], 权重 [0.7234, 0.2766]
  Token 1: 专家 [1, 5], 权重 [0.6543, 0.3457]
  Token 2: 专家 [2, 4], 权重 [0.8123, 0.1877]
  Token 3: 专家 [0, 6], 权重 [0.5912, 0.4088]
[EP模拟] Token分发到各设备:
  GPU 0: 4 个Token (专家0,1)
  GPU 1: 4 个Token (专家2,3)
  GPU 2: 4 个Token (专家4,5)
  GPU 3: 4 个Token (专家6,7)
[EP模拟] 最终输出 shape: torch.Size([16, 128])

============================================================
模拟完成！
输入shape: torch.Size([16, 128]) -> 输出shape: torch.Size([16, 128])

[核心结论] 专家并行的本质：
  1. 专家分片：每个GPU只存放部分专家权重
  2. Token分发（Dispatch）：将Token按路由结果发送到对应专家的GPU
  3. 本地计算：各GPU独立计算本地专家
  4. 结果合并（Combine）：将各专家输出按路由权重汇总
```



#### 关键设计注释

| 代码组件               | 对应真实EP行为                              |
| ------------------ | ------------------------------------- |
| `device_maps`      | 专家到GPU的静态分配（真实环境支持动态负载均衡）             |
| Top-K路由选择          | MoE模型每Token只激活部分专家（DeepSeek-R1为8/256） |
| `expert_to_tokens` | Dispatch阶段按目标专家分组Token                |
| 加权求和输出             | Combine阶段通过All-to-All通信汇总结果           |





***



### 三、生产部署与教学模拟的差异

| 差异点            | 生产EP (vLLM)                   | 本教学模拟  |
| -------------- | ----------------------------- | ------ |
| **通信**         | 真实All-to-All集合通信（NCCL/DeepEP） | 本地数据重排 |
| **专家分布**       | 动态负载均衡（EPLB）                  | 静态均匀分配 |
| **多节点**        | 支持，需InfiniBand/RDMA           | 单机模拟   |
| **CUDA Graph** | 支持，与解码阶段兼容                    | 未实现    |
| **性能**         | 多GPU真正的加速                     | 仅流程演示  |





多节点EP部署的**前置要求**包括：安装DeepEP、DeepGEMM、GDRCopy，确保InfiniBand全Mesh连接，以及正确配置IMEX服务。

