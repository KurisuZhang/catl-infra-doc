TensorRT-LLM 提供了一个高级 `LLM` API。它能自动处理模型下载、格式转换、引擎优化和推理等全部流程。



以下是一个使用 `LLM` API 的完整示例，全程只需不到 20 行代码。



```python
# 1. 导入必要的库
from tensorrt_llm import LLM, SamplingParams  # 
from tensorrt_llm import BuildConfig           # 

def main():
    # 2. (可选) 配置引擎构建参数
    build_config = BuildConfig()
    build_config.max_batch_size = 256          # 最大批次大小
    build_config.max_num_tokens = 1024         # 单批次最大Token数

    # 3. 创建LLM实例：指定模型，传入构建配置
    # 直接传入Hugging Face模型ID即可
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        build_config=build_config
    )

    # 4. 准备输入 prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 5. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95
    )

    # 6. 执行推理
    outputs = llm.generate(prompts, sampling_params)

    # 7. 打印输出
    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

if __name__ == '__main__':
    # 使用多GPU时，此入口保护是必需的
    main()
```



**代码逻辑拆解：**



* **导入模块**：从 `tensorrt_llm` 包中导入核心类 `LLM` 和配置类 `SamplingParams`、`BuildConfig`。

* **构建配置 (**`BuildConfig`**)**：这是一个关键的可选项，用于控制 TensorRT 引擎的构建参数。

  * `max_batch_size`: 定义了引擎一次能处理的最大请求数。

  * `max_num_tokens`: 限制了单个批次中所有请求的总 Token 数，用于控制显存和计算开销。

  * 你可以在这里进行更多高级配置，如量化等。

* **创建 `LLM` 实例**：这是最核心的一步。`LLM(model="...")` 这行代码在后台自动完成了：

  1. 从 Hugging Face 下载模型权重。

  2. 将模型转换为 TensorRT-LLM 的内部格式。

  3) 根据你的配置（如 `BuildConfig`）和硬件环境，构建一个高度优化的 TensorRT 引擎。

  4) 加载引擎，准备进行推理。

* **采样参数 (**`SamplingParams`**)**：控制文本生成策略。`temperature` 控制随机性，`top_p` 控制采样池的大小。

* **执行推理 (**`llm.generate`**)**：将 prompts 和采样参数传入，触发推理过程。支持传入字符串列表进行批量推理。

* **处理输出**：遍历 `outputs` 列表，从每个 `output` 对象中提取生成的文本。

> **注意**：当你第一次运行这个脚本时，TensorRT-LLM 需要下载模型并构建引擎，这个过程可能需要几分钟到十几分钟，取决于模型大小和硬件性能。引擎构建完成后会被缓存，后续运行会很快。
>
>



### 进阶：部署为在线服务 (`trtllm-serve`)

除了 Python API，TensorRT-LLM 还提供了命令行工具 `trtllm-serve`，可以一键启动一个与 OpenAI API 兼容的 HTTP 服务。



1. **启动服务**： &#x20;

在命令行中执行以下命令，即可为 `TinyLlama` 模型启动一个服务端。



```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```



2. **调用服务**： &#x20;

服务启动后，你可以使用 `curl` 命令或任何 OpenAI 客户端库来发送请求。



```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Where is New York? Tell me in a single sentence."}],
    "max_tokens": 32,
    "temperature": 0
  }'
```



通过本课的例子，可以看到 TensorRT-LLM 的 `LLM` API 极大地简化了使用流程：



1. **化繁为简**：将“模型转换”、“引擎构建”、“模型加载”和“推理”这四个原本独立的步骤，融合到一行 `LLM(model=...)` 代码中。

2. **性能优化**：在“一键式”操作的背后，TensorRT-LLM 自动应用了**图优化、算子融合、内核自动调优**等硬件级优化技术，确保了在 NVIDIA GPU 上的极致性能。

3) **生产就绪**：`trtllm-serve` 工具让从开发到部署的过渡变得非常平滑，可以快速启动一个生产级的推理服务。

下面通过一个完整的示例，带你从零开始体验 TensorRT-LLM 的“一键式”引擎构建与推理。

###
