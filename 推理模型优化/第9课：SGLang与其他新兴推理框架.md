## 引言：推理框架的“春秋战国”走向“三国鼎立”



在前两课中，我们深入学习了vLLM和TensorRT-LLM——两个分别代表**系统架构创新**和**硬件极致优化**的推理引擎。然而，推理框架的版图远不止于此。



2024年到2025年，大模型推理框架经历了一场“春秋战国”式的爆发——vLLM、TensorRT-LLM、TGI、SGLang、LMDeploy、llama.cpp等框架相继涌现，各有拥趸。但到了2026年，格局已经明显收敛：**通用生产部署看vLLM，Agent与结构化输出看SGLang，NVIDIA硬件极致性能看TensorRT-LLM**。



在这场收敛中，SGLang凭借其**结构化生成语言（Structured Generation Language）** 的独特定位异军突起，而曾经的“老牌劲旅”TGI则因进入维护模式而逐渐淡出新项目的选型视野。



## 一、SGLang的设计理念：结构化生成语言

### 1.1 起源与定位：不只是“又一个推理引擎”

SGLang起源于加州大学伯克利分校LMSYS.org团队——也就是大名鼎鼎的Chatbot Arena（大模型竞技场）的创造者。它于2024年1月首次公开发布，定位为**通用的LLM/VLM服务引擎，支持结构化生成和复杂推理任务**。



与vLLM“通过内存管理优化吞吐”的定位不同，SGLang的核心设计理念是：**通过前后端协同设计（co-designing the frontend language and the runtime system），让模型交互更快、更可控**。



SGLang不是简单地“在vLLM上加了一层API”——它是从头为**结构化LLM工作流**设计的全栈式编程与执行框架。截至2026年，SGLang已在全球超过**40万张GPU**上部署，日处理数万亿Token，被xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS等一线公司采用。



### 1.2 编程式前端：让LLM应用开发像写普通程序一样自然

SGLang最独特的创新在于它的**前端语言（Frontend Language）** ——一个专为大语言模型设计的领域特定语言（DSL）。



传统的LLM应用开发中，开发者需要手动管理Prompt拼接、多次模型调用的串并联、条件分支、循环等逻辑——这些在普通编程中理所当然的控制流，在LLM场景中却需要复杂的工程实现。



SGLang的前端语言改变了这一点。开发者可以用类似写普通Python程序的方式，定义复杂的LLM工作流：



```python
import sglang as sgl

@sgl.function
def multi_turn_dialogue(s, user_question, system_prompt):
    # 系统提示
    s += system_prompt
    # 多轮对话的循环
    for i in range(3):
        s += f"User: {user_question[i]}\n"
        s += "Assistant: " + sgl.gen("response_" + str(i), max_tokens=256)
        s += "\n"
```



这种“编程式”的接口设计，使得**链式生成调用、高级提示工程、控制流、多模态输入、并行处理和外部交互**都可以用统一的语法表达。



### 1.3 RadixAttention：让“公共前缀”只算一次

如果说vLLM的PagedAttention解决了“KV Cache怎么存”的问题，那么SGLang的RadixAttention解决的是 **“KV Cache怎么复用”** 的问题。



**核心原理**：RadixAttention使用**基数树（Radix Tree，即压缩前缀树）** 数据结构来索引已缓存的Token序列。当新请求到来时，调度器通过基数树进行**最长前缀匹配（Longest Prefix Matching）** ——如果新请求的输入与已缓存的某个请求共享相同前缀，就直接复用对应的KV Cache，完全跳过Prefill阶段的计算。



**一个具体的例子**：



假设你的RAG应用有固定的System Prompt（500 Token），每个用户问题不同但都跟在同一个System Prompt后面。没有RadixAttention时，每个请求都要重新计算这500 Token的KV Cache——浪费巨大。有了RadixAttention：



1. 第一个请求到来时，System Prompt的KV被存入基数树

2. 后续请求到达时，RadixAttention检测到前缀匹配，**直接复用**已缓存的KV

3) 只有用户问题部分需要新计算

**性能数据**：RadixAttention相比基于哈希的缓存方案，可将前缀缓存命中率提升**5倍**。在多轮对话和RAG场景中，这种复用可以**显著降低首Token延迟并提升吞吐**。



**与vLLM前缀缓存的对比**：



| 维度   | vLLM前缀缓存     | SGLang RadixAttention |
| ---- | ------------ | --------------------- |
| 数据结构 | 哈希表（Block级别） | 基数树（Token级别）          |
| 匹配粒度 | 固定16-Token块  | 可变长度                  |
| 匹配精度 | 块级（可能浪费部分匹配） | Token级（精确最长匹配）        |





### 1.4 结构化输出：xGrammar的“强制合规”机制

SGLang的另一大核心能力是**结构化输出（Structured Output）** ——通过其xGrammar后端，确保模型输出严格符合JSON Schema、正则表达式或EBNF语法约束。



**传统方法的痛点**：



常规做法是“先生成，再校验”——让模型自由生成文本，然后用正则或JSON解析器去检查是否符合格式。如果不符合，就重试。这种方法的问题是：



* **不确定性**：无法保证输出一定符合格式

* **效率低**：无效生成浪费计算资源

* **延迟不可控**：重试次数不确定

**xGrammar的解法：约束解码（Constrained Decoding）**



xGrammar在生成的**每一步**都动态计算当前状态下合法的Token集合。如果JSON Schema规定某个字段必须是数字，xGrammar会在解码时**屏蔽所有非数字Token**，确保模型“不得不”生成合法内容。



更重要的是，xGrammar将**语法掩码生成与LLM前向计算重叠（Overlap）** ——掩码生成在GPU上并行执行，不额外增加延迟。



**性能数据**：xGrammar相比标准引导解码（Guided Decoding）可快**3倍**。在JSON Schema工作负载上，比现有结构化生成方案快**3.5倍**；在CFG（上下文无关文法）工作负载上，快**超过10倍**。



**实战示例**：强制生成合法JSON



```python
import sglang as sgl

@sgl.function
def generate_product_json(s, description):
    s += f"Generate a JSON for a product based on this description: {description}\n"
    s += sgl.gen(
        "product",
        max_tokens=128,
        temperature=0.1,
        # 通过JSON Schema约束输出格式
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "category": {"type": "string"}
            },
            "required": ["name", "price", "category"]
        }
    )

# 输出一定是合法的JSON，例如：
# {"name": "Wireless Headphones", "price": 79.99, "category": "electronics"}
```



这种“**必须生成JSON，而不是能生成JSON**”的确定性，在工程落地中比“高准确率”更珍贵——**错一个括号、少一个逗号、类型不匹配，请求直接失败**，而不是默默返回一个无法解析的字符串。



### 1.5 mini-SGLang：5000行代码的“推理核心”

2026年6月，SGLang团队发布了**mini-SGLang**——将原本30万行的代码库压缩到仅5000行。这个迷你版本保留了所有核心优化技术，包括**重叠调度（Overlap Scheduling）、FlashAttention-3、RadixAttention**等，在在线服务场景下性能与完整版几乎相同。



mini-SGLang的价值在于：



* **极低的学习门槛**：5000行代码即可理解LLM推理引擎的核心机制

* **轻量部署**：适合资源受限的边缘场景

* **快速原型**：开发者可以基于mini版本快速定制自己的推理方案

## 二、TGI（Text Generation Inference）的架构特点

### 2.1 历史定位：Hugging Face生态的“官方推理引擎”

TGI（Text Generation Inference）是Hugging Face于2022年推出的生产级推理引擎，专为开源LLM（LLaMA、Falcon、StarCoder、BLOOM等）的高性能服务而设计。



它的核心优势在于**与Hugging Face生态的深度整合**——从Hugging Face Hub拉取模型、使用Transformers库的配置格式、通过Hugging Face Inference Toolkit部署，整个链路无缝衔接。



### 2.2 架构设计：Rust + Python的“双引擎”策略

TGI最独特的设计是**Rust与Python联用**：



* **Rust层**：负责HTTP服务和调度层（Scheduler）。Rust的强类型系统和内存安全保证，使得TGI能够**最大化并发性能，同时绕过Python的GIL（全局解释器锁）限制**。Rust的零成本抽象和 fearless concurrency 让TGI在处理高并发请求时表现出色。

* **Python层**：负责模型加载和前向计算。Python在AI生态中的主导地位使得TGI可以无缝集成Hugging Face的Transformers库和各类量化工具。

TGI采用**三层架构**：



1. **Launcher（启动器）** ：编排部署，计算最优配置，管理进程生命周期

2. **Router（路由器）** ：基于Rust的高性能HTTP服务器，负责请求处理、验证和批处理

3) **Worker（工作器）** ：实际执行模型推理的Python进程

### 2.3 核心功能

TGI在功能上与vLLM、SGLang高度重叠，包括：



* **连续批处理 + 流式输出**：动态分组飞行中的请求，通过SSE流式返回Token

* **优化的Attention与解码**：Flash Attention、Paged Attention、KV Cache

* **量化支持**：bitsandbytes、GPTQ等

* **生产就绪**：OpenAI兼容API、Prometheus指标、OpenTelemetry追踪

* **多后端支持**：2025年1月，TGI引入了**多后端（Multi-Backend）** 架构，可以通过统一的TGI前端层集成vLLM、TensorRT-LLM、llama.cpp等不同后端

### 2.4 重要转折：2025年12月进入维护模式

2025年12月11日，Hugging Face官方宣布TGI进入**维护模式（Maintenance Mode）** 。这意味着：



* 只接受**小Bug修复、文档改进和轻量级维护任务**的PR

* 不再添加新功能

* 对于Inference Endpoints，官方推荐使用**vLLM或SGLang作为替代方案**

这一决定的原因是多方面的：随着vLLM、SGLang等框架的成熟和生态的壮大，TGI的差异化优势逐渐缩小。Hugging Face选择将精力集中在**作为统一前端层整合各后端**的战略方向上，而非继续维护一个独立的推理引擎。



**对选型的影响**：对于2026年及以后的新项目，TGI已不再是推荐的默认选项。虽然现有TGI部署可以继续运行，但新项目应优先考虑vLLM或SGLang。







## 三、框架选型指南：不同场景下的最优选择

### 3.1 2026年推理框架格局

经过2024-2025年的激烈竞争，2026年的推理框架格局已基本清晰：



| 框架               | 核心定位             | 最佳场景                    |
| ---------------- | ---------------- | ----------------------- |
| **vLLM**         | 通用生产部署的默认首选      | 大多数在线服务、模型兼容性优先         |
| **SGLang**       | Agent与结构化输出专家    | 多轮对话、RAG、Agent、强制JSON输出 |
| **TensorRT-LLM** | NVIDIA硬件极致性能     | 纯NVIDIA环境、追求峰值吞吐        |
| **TGI**          | Hugging Face生态整合 | **已进入维护模式，不推荐新项目**      |
| **llama.cpp**    | CPU/边缘端推理        | 无GPU环境、本地部署             |
| **LMDeploy**     | 轻量级高性能方案         | 快速部署、资源受限场景             |





### 3.2 选型决策树

```plain&#x20;text
第一步：硬件环境
├── CPU only → llama.cpp
├── AMD GPU / 国产NPU → vLLM（ROCm支持）或 LMDeploy
└── NVIDIA GPU（A100/H100/L40S）
    ├── 追求极致吞吐（>2000 Tokens/s）→ TensorRT-LLM
    └── 平衡性能与易用性 → 进入第二步

第二步：业务场景
├── 通用在线服务、模型快速迭代 → vLLM
├── 多轮对话 / RAG（固定System Prompt）→ SGLang（RadixAttention）
├── Agent / Tool Use / 强制JSON输出 → SGLang（xGrammar）
├── 长上下文、高并发 → vLLM 或 SGLang（两者均支持PagedAttention）
└── 已有Hugging Face深度投资 → 迁移至vLLM或SGLang
```



### 3.3 场景化选型建议

**场景一：通用在线对话服务**



如果业务是标准的Chatbot，模型需要频繁更换（今天LLaMA、明天Qwen），团队希望快速上线、运维简单→ **选vLLM**。vLLM是目前开源社区的事实标准，硬件覆盖广（NVIDIA、AMD、TPU等），新模型Day 0支持最快。



**场景二：多轮对话 + RAG**



如果业务是多轮对话（如客服系统），有固定的System Prompt，每个请求共享大量公共前缀→ **选SGLang**。RadixAttention的基数树缓存可以让多轮对话的吞吐量**远超vLLM**。



**场景三：Agent + 结构化输出**



如果业务需要Agent调用Tool、强制输出JSON Schema（如API编排、数据抽取）→ **选SGLang**。xGrammar的约束解码确保输出100%合规，无需重试。



**场景四：极致性能 + NVIDIA独占**



如果业务在纯NVIDIA环境（A100/H100/Blackwell），追求每一毫秒的极致性能，团队有足够的工程能力处理TensorRT引擎的编译和调试→ **选TensorRT-LLM**。在dense模型上，TensorRT-LLM的吞吐可比vLLM高出一到两成。



**场景五：CPU / 边缘端**



如果没有GPU或只有CPU→ **选llama.cpp**。llama.cpp是目前CPU上最快的推理引擎，支持最广泛的模型格式（GGUF），对量化级别和线程数有完全的控制。



### 3.4 选型中的常见误区

**误区一：“选最快的框架就行了”**



“最快”取决于场景。在7B模型上，vLLM、SGLang和TensorRT-LLM的性能差距可能只有10-20%；但在70B模型上，错误的选型可能导致**GPU算力浪费50%以上**。**先明确业务场景，再谈性能**。



**误区二：“SGLang只是vLLM的上层封装”**



SGLang确实复用了vLLM的部分架构，但它的RadixAttention和xGrammar是**从底层重新设计的**，不是简单的API封装。在结构化生成和多轮对话场景中，SGLang的能力远超vLLM。



**误区三：“TGI还能用，先凑合着”**



TGI已于2025年12月进入维护模式。虽然现有部署可以继续运行，但新功能不会再有，长期来看存在技术债务风险。**新项目应避免选择TGI**。

##
