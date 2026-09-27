

> **硬件说明**：DeepSeek-V3（671B总参数，37B激活参数）需多卡A100/H100集群部署，Mixtral 8x7B（8×7B，每Token激活2个专家）可在单机8卡环境部署。以下命令均基于vLLM。
>
>



### 一、DeepSeek-V3推理部署

DeepSeek-V3采用MoE架构（671B总参数，每Token激活37B），推理优化的核心是**PD分离**和**专家并行（EP）的联合配置**。



#### 1.1 基础部署命令

DeepSeek-V3官方推荐使用vLLM进行部署：



```bash
# 安装vLLM
pip install vllm

# 启动vLLM服务（基础版，适合有足够GPU资源的场景）
vllm serve "deepseek-ai/DeepSeek-V3" \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --trust-remote-code
```



#### 1.2 高级部署：EP + PD分离

对于生产级部署，火山引擎提供了基于xLLM的PD分离部署方案，将Prefill和Decode分离到不同实例：



| 组件            | 推荐配置          | 说明                     |
| ------------- | ------------- | ---------------------- |
| **Prefill实例** | TP=8，计算密集     | 快速处理输入Prompt生成KV Cache |
| **Decode实例**  | EP=16-32，DP=8 | 高并发Token生成，优化访存        |





**部署架构**：



```plain&#x20;text
用户请求 → API网关 → Prefill实例(TP=8) → KV传输 → Decode实例(EP=16-32) → 响应
```



性能对比数据：



| 指标   | 开源SGLang  | xLLM（优化后） |
| ---- | --------- | --------- |
| TTFT | ≤5000ms   | ≤150ms    |
| 单卡吞吐 | 167.4 TPS | 669 TPS   |





#### 1.3 Transformers直接加载

对于小规模测试或研究场景，可直接使用Hugging Face Transformers加载：



```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V3", 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    trust_remote_code=True,
    device_map="auto",  # 自动分配到多GPU
    torch_dtype="auto"
)

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```







### 二、Mixtral 8x7B推理部署

Mixtral 8x7B包含8个专家模型，每个Token仅激活2个专家（约13B参数），是MoE推理的经典基准模型。



#### 2.1 vLLM基础部署

```bash
# 安装vLLM
pip install vllm

# 启动服务
vllm serve "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --dtype float16 \
    --gpu-memory-utilization 0.7
```



Mixtral使用特定指令格式：



```python
# 调用示例
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "[INST] What is the capital of France? [/INST]"}
    ],
    max_tokens=128
)
```



#### 2.2 Transformers加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# FP16精度加载（需GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```



#### 2.3 负载均衡优化

Mixtral部署时，启用vLLM的均衡调度可提升吞吐：



```bash
# 启用v1调度器的均衡调度
VLLM_ASCEND_BALANCE_SCHEDULING=1 \
vllm serve "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.7
```



**关键参数说明**：



| 参数                               | 作用       | 建议值      |
| -------------------------------- | -------- | -------- |
| `--tensor-parallel-size`         | 张量并行度    | 与GPU数量一致 |
| `--enable-expert-parallel`       | 启用专家并行   | MoE模型必开  |
| `--gpu-memory-utilization`       | GPU显存使用率 | 0.7-0.9  |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 均衡调度     | 1（启用）    |





### 三、性能对比参考

基于Nano-vLLM-MS在RTX 3090上的MoE推理测试数据：



| 模型               | 引擎           | 吞吐量 (tokens/s) |
| ---------------- | ------------ | -------------- |
| Qwen3-MoE (随机权重) | vLLM         | 24,242         |
| Qwen3-MoE (随机权重) | Nano-vLLM-MS | 23,214         |





实际生产环境中，DeepSeek-V3在PD分离优化后，单卡吞吐可达**669 TPS**（vs 开源SGLang的167.4 TPS）。







### 四、完整部署流程检查清单

| 步骤 | 检查项    | 命令/方法                                              |
| -- | ------ | -------------------------------------------------- |
| 1  | GPU环境  | `nvidia-smi`                                       |
| 2  | vLLM安装 | `pip install vllm`                                 |
| 3  | 模型下载   | `huggingface-cli download deepseek-ai/DeepSeek-V3` |
| 4  | 启动服务   | `vllm serve "model_path" --tensor-parallel-size 8` |
| 5  | 验证服务   | `curl http://localhost:8000/v1/models`             |
| 6  | 发送请求   | OpenAI API curl调用                                  |
| 7  | 性能压测   | `vllm bench` 命令                                    |

