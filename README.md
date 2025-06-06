# Catalogue
* [LLM](#llm)
* [LLM & Knowledge Graph](#llm--kg)
# LLM
## 数据准备
以下内容参考自: https://wangrongsheng.github.io/awesome-LLM-resourses/

### **数据标注与合成工具**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **AotoLabel** | 使用LLM自动清洗和标注数据，提升效率和一致性 | 依赖LLM，可能有偏差；对多语言或领域特定数据效果不一 |
| **LabelLLM** | 开源、可自托管，适用于各种数据标注任务 | 复杂项目可能需要额外定制；社区支持尚在成长中 |
| **Distilabel** | 基于研究论文构建的可信合成数据与反馈框架 | 有一定学习曲线；对初学者不够友好 |
| **llm-swarm** | 可大规模生成合成数据，适合Cosmopedia这类任务 | 对数据质量控制要求高，容易产生冗余或偏差 |
| **Promptwright** | 本地部署，生成合成数据更安全可控 | 对本地LLM资源要求较高，硬件门槛可能较大 |
| **Curator** / **Bespoke Curator** | 用于后训练和结构化数据提取的合成数据策展 | 仍依赖于上游模型质量；对于语义理解较复杂的文本仍有局限 |

---

### **PDF与文档解析工具**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **PDF-Extract-Kit** | 提供高质量PDF内容提取工具，功能全面 | 文档结构复杂时可能提取不完整或失真 |
| **pdf-extract-api** | 集成现代OCR+Ollama模型，支持API接入 | 可能有延迟或费用问题；精度依赖OCR模型 |
| **Zerox** | 基于GPT-4o-mini进行零样本PDF OCR，非常现代 | 模型能力有限时提取准确性受限 |
| **pdf2htmlEX** | 将PDF转为HTML，保留格式优秀 | 对图片、复杂表格支持有限；非LLM友好格式 |
| **olmOCR** | 为训练LLM设计的OCR工具，适配“野外”文档 | 相对较新，社区支持和成熟度待验证 |

---

### **网页与多格式解析**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **Parsera** | 轻量级网页爬虫，结合LLM进行内容提取 | 不适合结构极其复杂或频繁变动的网站 |
| **OmniParser** | 支持多种格式，Go语言高效流式处理 | 非Python生态，集成到某些项目可能有门槛 |
| **Sparrow** | 支持从图像与文档中提取信息，适合复杂数据源 | 可能需要配合训练或自定义规则使用 |
| **MinerU** | 多源抽取工具，支持PDF/网页/电子书等 | 初始设置稍复杂；对格式适配要求较高 |

---

### **表格处理与结构化提取**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **Tabled** | 自动识别表格并导出为Markdown/CSV | 对结构模糊的表格识别率下降 |
| **DocLayout-YOLO** | 强化文档布局识别的YOLO方案，支持合成训练数据 | 需额外训练；硬件需求高 |
| **MegaParse** | 为LLM优化设计，防止信息丢失的高质量解析器 | 功能强但可能过于复杂，需调试集成 |
| **ReaderLM-v2** | 将HTML转为美观的Markdown或JSON，非常适合训练用数据 | 处理非标准HTML时可能失败或混乱 |

---

### **数据清洗、去重与优化**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **DataTrove** | 大规模文本处理、去重、过滤的强大工具 | 对系统资源有一定消耗，需优化配置 |
| **semhash** | 使用语义相似度去重，更智能 | 算法参数调优不当易出现误杀或漏判 |
| **datasketch** | 支持处理极大规模数据的概率数据结构工具 | 存在一定信息丢失的风险，不适合所有场景 |
| **LLM Decontaminator** | 为Benchmark去除污染样本，保证公平性 | 对普通用户应用场景有限，更偏研究用途 |

---

### **模型辅助与监控**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **TensorZero** | 支持通过“经验”提升LLM，强化学习方向 | 仍处于新阶段，使用门槛高 |
| **LangKit** | 监控LLM响应，保证安全性和一致性 | 多用于企业/研究环境；设置稍复杂 |

---

### **通用处理工具**
| 工具 | 优点 | 缺点 |
|------|------|------|
| **MarkItDown** | 将Office文档/文件转换为Markdown，方便清洗 | 格式可能有损，特别是复杂排版 |
| **Common-Crawl-Pipeline-Creator** | 创建公共抓取管道，适合大规模网络数据收集 | 门槛较高，适合工程团队使用 |
| **Easy Dataset** | 快速创建微调数据集，适配LLM训练 | 需一定理解数据结构和标签需求 |

---

## 模型训练（微调）

### 一体化高效微调框架（支持多模型、多策略）

| 工具 | 优点 | 缺点 |
|------|------|------|
| **LLaMA-Factory** | 支持100+模型、各种微调策略（LoRA、QLoRA、DPO等），社区活跃 | 配置稍复杂，对新手不够友好 |
| **360-LLaMA-Factory** | 在LLaMA-Factory基础上增加了Sequence Parallelism，适合长上下文训练 | 更依赖分布式资源，配置复杂 |
| **unsloth** | 2-5倍加速，减少80%内存消耗，极度高效 | 当前集中于QLoRA策略，通用性有限 |
| **Xtuner** | 灵活、高效、全功能，面向工业微调 | 文档相对分散，学习成本略高 |
| **torchtune** | 原生 PyTorch 实现，兼容性好，透明度高 | 模板少，功能较为基础，需要自己搭建训练逻辑 |
| **Swift** | 支持200+LLM和15+多模态模型（MLLMs），支持PEFT与全量微调 | 更新较频繁，可能存在兼容性问题 |
| **LitGPT** | 强调高性能（Flash Attention, LoRA, FSDP等），训练+部署全包 | 当前更适合研究人员使用 |

---

### RLHF、DPO、对齐训练框架

| 工具 | 优点 | 缺点 |
|------|------|------|
| **OpenRLHF** | 支持70B全量调优、LoRA、Mixtral、DPO等策略，功能超全 | 上手需要一定RLHF知识 |
| **ChatLearn** | 灵活高效的对齐训练框架，支持大规模数据训练 | 对初学者不太友好，需要较强工程能力 |
| **Online-RLHF** | 支持在线DPO与RLHF，适合实时优化 | 依赖自定义反馈系统 |
| **Effective LLM Alignment** | 涵盖各种对齐技术（如RLAIF、DPO等） | 研究性较强，实用性依赖定制化能力 |
| **Proxy Tuning** | 使用代理任务提升训练效果，降低高成本任务的依赖 | 适用范围有局限，需配合自定义任务 |

---

### 低门槛微调（自动化/低代码/GUI）

| 工具 | 优点 | 缺点 |
|------|------|------|
| **AutoTrain / autotrain-advanced** | 自动化训练、部署，易用性高 | 灵活性和控制力稍弱 |
| **H2O-LLMStudio** | 提供图形界面，支持微调多个开源模型 | 模型适配性略有限，适合中小项目 |
| **Simplifine** | 一行代码即可开始微调，极简入门 | 仅适合基础需求；不支持复杂训练逻辑 |
| **Kiln** | 微调 + 合成数据 + 协作平台一体化 | 云平台可能有限制；尚不成熟 |

---

### 通用训练/评估工具

| 工具 | 优点 | 缺点 |
|------|------|------|
| **LLMBox** | 全流程训练+评估一体化框架 | 功能全面但文档偏复杂 |
| **Ludwig** | 低代码，支持多种模型类型，不局限于LLM | 更适用于通用ML任务，LLM支持较新 |
| **Transformer Lab** | 本地交互式训练与测试平台，适合研究型使用 | 文档有限，需摸索使用方式 |
| **aikit** | 一站式部署开源LLM，适合快速上手 | 新项目，社区生态尚未成熟 |
| **LLM-Foundry** | Databricks官方支持，训练大模型的工业级流程 | 偏向Databricks生态，迁移成本高 |

---

### 多模态训练支持（文本+视觉）

| 工具 | 优点 | 缺点 |
|------|------|------|
| **TinyLLaVA Factory** | 支持轻量多模态模型微调，面向边缘部署 | 模型选择有限，非通用型方案 |
| **lmms-finetune** | 支持LLaVA、Qwen-VL等多个多模态模型 | 对硬件配置有一定要求 |
| **MLX-VLM** | 支持Mac上本地运行与微调VLM模型 | 仅限于Apple Silicon生态 |
| **Vision-LLM Alignment** | 针对视觉语言对齐（SFT/RLHF/DPO）全流程 | 使用门槛高，适合研究人员 |
| **finetune-Qwen2-VL** | 快速开始Qwen2-VL的训练 | 限于Qwen2-VL模型，适配性不强 |

---

### 系统/底层优化工具

| 工具 | 优点 | 缺点 |
|------|------|------|
| **Liger-Kernel** | Triton内核优化训练效率 | 需要底层开发经验 |
| **nanotron** | 极简多卡并行训练框架（3D并行） | 功能不多，适合工程/研究用 |
| **veRL** | Volcano Engine支持的高性能强化学习平台 | 商业闭环生态，开源文档较少 |
| **InternEvo** | 开源轻量训练框架，无需大量依赖 | 适合轻量训练，不适合超大模型 |

---

### 实验性 / 研究友好型项目

| 工具 | 优点 | 缺点 |
|------|------|------|
| **Meta Lingua** | 面向LLM研究的极简高效代码库 | 更适合开发者/研究者 |
| **DeepSeek-671B-SFT-Guide** | 支持DeepSeek-V3/R1 671B的完整SFT流程 | 针对特定模型，通用性弱 |
| **Oumi** | 从预处理、训练到推理全流程支持 | 仍在发展中，资源和文档较少 |

---



# LLM & KG
## Preface
大语言模型已经能很好的实现问答、文本输出，但其输出结果可能有时候并没有足够的依据，因此利用 **RAG(Retrieval-Augmented Generation)** 技术，通过在生成回答之前先检索相关知识，从而使生成的内容更加可靠，适用于问答系统、文档生成、客户支持等任务。这样的相关知识可以是文档，也可以是知识图谱。

本文旨在从0开始研究LLM和知识图谱的融合: LLM赋能知识图谱的构建，知识图谱赋能LLM的准确性。

在研究开始前，需了解RAG，NER，LLM， Knowledge graph。

## 当前待解决问题
* 如何实现模型并行？vLLM似乎只支持数据并行，当单卡显存不足以推理模型时即使多卡也无法推理
* 如何微调LLM？ 参考: https://qwen.readthedocs.io/zh-cn/latest/training/SFT/llama_factory.html

## RAG
**常见工具**

[RAGFlow](https://infiniflow.cn)  更侧重于通过检索增强生成的方式，专注于利用外部知识来增强语言模型的推理能力。

[Langchain](https://www.langchain.com)  提供更灵活的框架，支持多种模块集成和复杂的推理任务，适用于多模态交互和不同类型的任务管理。

### RAG的优势
* 增强的知识覆盖：即使生成模型本身没有直接学习到特定信息，通过检索可以查找到相关信息。
* 动态更新：由于信息可以实时检索，知识库或数据库中的更新信息可以即时被使用，模型无需重新训练。
* 提升生成的准确性和可信度：在生成阶段引入检索信息，使生成内容更具参考价值，有助于回答事实性问题。
### RAG的应用
* 问答系统：提供更精准的回答，尤其适合需要基于外部知识库回答的问题。
* 文档生成：将多个段落或文档中的内容组织成流畅的文本。
* 客户支持和推荐系统：根据问题或查询检索相关信息，并生成响应，以便提供个性化的回答。
### RAG的实现
* RAG通常通过两步分离的管道实现：首先检索模型（如DPR）获取相关文档，然后生成模型（如BERT、T5）生成最终回答。
* 在实践中，可以使用诸如Hugging Face Transformers库中的实现，将RAG模型部署在本地或云端。
在neo4j的官网中有该文档介绍了关于Graph RAG: https://neo4j.com/blog/graphrag-manifesto/
## NER
NER（Named Entity Recognition，命名实体识别）是一种自然语言处理任务，用于识别文本中的命名实体。命名实体可以是人名、地名、组织名、时间、日期、货币、百分比等具有特定意义的词或短语。NER在信息抽取、知识图谱构建、问答系统等领域具有广泛的应用。
### NER的主要步骤
文本预处理：对输入文本进行预处理，包括分词、去停用词、词性标注等，为后续实体识别做准备。
特征提取：对每个词或短语提取特征，如词本身、词的上下文、词性、词的形态特征等。
识别和分类：使用机器学习或深度学习模型将文本中的每个词标注为实体或非实体。对于实体，还需进一步分类（如人名、地名、组织名等）。
### NER的主要模型方法
基于规则的NER：通过预定义的规则和词典识别实体。例如，通过正则表达式匹配日期格式来识别日期实体。
机器学习模型：HMM（隐马尔可夫模型）和CRF（条件随机场）：传统的序列标注算法，利用统计特征预测每个词的标签。
深度学习模型：
BiLSTM-CRF：利用双向LSTM捕捉文本的上下文信息，再用CRF层对序列标签进行全局优化。
Transformer-based模型（如BERT、RoBERTa、GPT等）：基于Transformer架构的预训练模型在NER任务中表现优异，尤其在细粒度和上下文依赖性较强的任务中。这些模型可以通过微调，直接应用于NER任务。
### NER的常见应用
信息抽取：从非结构化文本中提取出有用的结构化信息（如人名、时间、地点）。
知识图谱构建：识别实体后，可以将其与关系抽取结合，构建知识图谱。
问答系统：识别用户问题中的关键信息（如实体和意图）以提升问答的准确性。
情感分析：识别情感倾向的对象，如产品名或人物名，以提供更细致的情感分析。
### NER的常见挑战
同名问题：一些实体具有同样的名称（如“苹果”可以指公司或水果），需要通过上下文判断具体的实体类别。
语言和领域的多样性：不同语言和领域的实体类型和命名方式差异很大，通用模型可能无法适应特定领域。
多义性：一个词在不同上下文中的实体类别可能不同，这对模型理解和分类带来了挑战。

## LLM
在Hugging Face上测试, 以下模型显存占用情况, 以及使用模型进行三元组抽取的表现:
[实验内容 Colab](https://colab.research.google.com/drive/1scsACHDW_1hjFq3KDSfsOuLgPJGx8ox9?usp=sharing)
由于显存问题，白嫖modelscope的24GB显存进行部分实验；vLLM加载模型使用默认参数。
| 模型                           | HF模型显存占用    | vLLM显存占用   | 可靠性           | 主观评分 |
|--------------------------------|---------|---------|------------------|------|
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                   | 2.5GB   |       | 几乎不可靠        | 0.1  |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                   | 4GB     |       | 偶尔可靠         | 0.2  |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                       | 7GB     |       | 有些可靠         | 0.3  |
| [Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)   | 2.3GB   |  9.2GB| 有些可靠         | 0.3  |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)                       | 15GB    |       | 有些可靠         | 0.5  |
| [Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)   | 5.4GB   | 16.5GB| 有些可靠         | 0.5  |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)                     | 29GB    |       |                |      |
| [Qwen2.5-14BInstruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)  |         | 18.3GB|                |      |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)                     | 65GB    |       |                |      |
| [Qwen2.5-32B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4)  |    |21.2GB (max_model_len=48)|   |     |
| [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)                     | 145GB   |       |                |      |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)   |        |    20GB   |  比Qwen7B强 | 0.6  |
### 部署优化
直接使用Hugging Face上的demo推理模型存在严重效率低效的问题，尝试如下方案:
* vLLM, 直接，大致需要配置以下环境:
  ```
  # 环境配置
  conda create -n vllm python==3.10
  conda activate vllm
  pip install vllm==0.6.1 -i https://mirrors.aliyun.com/pypi/simple # 好像在Qwen2.5介绍页看到不支持当前最新版本(v0.6.3)
  pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
  pip install -U accelerate bitsandbytes datasets peft transformers -i https://mirrors.aliyun.com/pypi/simple
  pip install auto_gptq -i https://mirrors.aliyun.com/pypi/simple
  pip install optimum -i https://mirrors.aliyun.com/pypi/simple
  mkdir Qwen2.5 && cd Qwen2.5

  # 下载模型
  modelscope download --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --local_dir ./3B/int4
  modelscope download --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --local_dir ./3B/int4
  modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./3B/fp16
  modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./3B/fp16

  # 使用
  vllm serve 3B/int4 --dtype auto --api-key 123 --port 8008 --max-model-len 32768 --gpu-memory-utilization 0.8
  或python代码:
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  sampling_params = SamplingParams(temperature=0.2, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
  llm = LLM(model=model_name, gpu_memory_utilization=0.9, enforce_eager=True, max_model_len=32768, tensor_parallel_size=1)
  
  # gpu_memory_utilization  enforce_eager  memory
  # 0.9                     False          11GB  
  # 0.9                     True           9.7GB
  # 0.7                     True           7.2GB
  # 0.6                     True           6GB
  # 0.5                     True           报错, 提示需要降低max_model_len 默认32768: Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
  # 0.4 (不能再低了)         True    max_model_len=16       5.6G
  # 初始化对话历史
  ```
## Knowledge Graph
### 什么是三元组信息
三元组可以表示为以下形式: (主体, 关系, 客体)

**示例**

句子：“巴菲特是投资大师。” 主体：巴菲特 关系：是 客体：投资大师

三元组：(巴菲特, 是, 投资大师)

句子：“马云创办了阿里巴巴。” 主体：马云 关系：创办了 客体：阿里巴巴

三元组：(马云, 创办了, 阿里巴巴)

### 为什么要三元组信息
由于知识图谱的**有向图数据结构**由点和带方向的边构成，其恰好需要一个source(src，对应于三元组的主体sub）, relation（rel，表示主体客体关系），target（tgt，对应三元组的客体obj），因此在构建知识图谱过程中需要三元组数据。

### 如何获得三元组信息
一个简单的英文三元组抽取例子: [colab](https://colab.research.google.com/drive/1scsACHDW_1hjFq3KDSfsOuLgPJGx8ox9?usp=sharing)

**最简单可靠的方案:** 利用强大的GPT类模型(但较为昂贵, 无论是时间还是金钱成本)

<img width="500" alt="image" src="https://github.com/user-attachments/assets/97c5c1c1-fa2b-4295-918e-41322eaa8e78">

基于Qwen2.5模型的实验参考: [实验内容 Colab](https://colab.research.google.com/drive/1scsACHDW_1hjFq3KDSfsOuLgPJGx8ox9?usp=sharing)

**基于NER的方案:** 利用NER技术 + /规则/机器学习/Transformer模型/_GPT(又贵又好)_

例如我有句子:

```
“巴菲特是著名的投资者，他投资了公司A和公司B。公司A收购了公司C。”
```

Step 1 假设通过NER技术得到结果
```
- 实体：巴菲特，标签：PERSON
- 实体：公司A，标签：ORG
- 实体：公司B，标签：ORG
- 实体：公司C，标签：ORG
```

Step 2 对NER的到的实体进行交叉配对:

```
pairs = [    ("巴菲特", "公司A"),    ("巴菲特", "公司B"),    ("公司A", "公司C")]
```

Step 3 构建提问内容:

```
"巴菲特 和 公司A 之间的关系是什么？"
"巴菲特 和 公司B 之间的关系是什么？"
"公司A 和 公司B 之间的关系是什么？"
```

Step 4 利用Transformer模型/GPT:

将原句子、提问内容给到Transformer模型，得到以下结果

```
"巴菲特 和 公司A 之间的关系是：投资"
"巴菲特 和 公司A 之间的关系是：投资"
"公司A 和 公司B 之间的关系是：收购"
```

Step 5 利用Transformer/GPT的结果搭建三元组

```
("巴菲特", "投资", "公司A")
("巴菲特", "投资", "公司B")
("公司A", "收购", "公司B")
```

### 存储数据结构
最初设计的方案是仅包含`主体`, `关系`, `客体` 三个部分，形如: 

<img width="255" alt="image" src="https://github.com/user-attachments/assets/8b57182b-4a9c-43e6-9a3a-b975df2a609e">

但针对信息溯源、属性缺失等问题，优化数据结构:

<img width="573" alt="image" src="https://github.com/user-attachments/assets/72d99328-0402-4569-9958-c64c44965b26">


### Neo4j安装
将 Neo4j 仓库添加到系统的apt源列表
尝试了很多民间方案都安装失败了, 还是找的官方教程:

https://neo4j.com/docs/operations-manual/current/installation/linux/debian/

关于Neo4j的更多搭建部署细节参见本人知乎文章: https://zhuanlan.zhihu.com/p/6404202845
## LLM x Knowledge Graph
### 知识图谱作为检索源
检索节点和关系：可以将知识图谱作为检索来源，将用户的查询与知识图谱中的实体和关系匹配，并检索出相关的子图。检索到的结构化信息可以直接传递给生成模型，以提供有根据的生成内容。
结合文本检索：如果查询较为复杂，RAG系统可以先进行文档检索，再在匹配到的文档中提取出相关实体和关系。生成模型可以在文档和知识图谱的双重支持下生成更加丰富的回答。
### 知识图谱与生成模型的融合
结构化信息输入：生成模型可以利用知识图谱中的节点属性和关系，作为额外的上下文信息。例如，在问答过程中，生成模型可以参考实体的相关属性值，并基于这些属性生成答案。
关系路径增强：如果知识图谱中存在查询实体的多层关系（如“公司—产品—功能”路径），生成模型可以利用这种路径信息，生成更具逻辑性的回答。例如，模型可以沿着关系路径“推理”出答案，而不仅仅依赖无结构的文档信息。
### 推理与动态更新
推理：知识图谱可以通过规则和逻辑推理扩展已有知识。例如，对于具有关系推理功能的知识图谱（如有推理引擎的图数据库），可以根据现有的关系推导出新的知识，并将这些推理结果提供给生成模型。
动态更新：RAG系统可以从知识图谱中实时检索和生成最新的信息，无需重新训练模型，从而实现动态知识的更新和准确性保持。
### RAG与知识图谱的混合架构
图谱嵌入：利用图嵌入技术将知识图谱的结构信息转换为数值向量。将知识图谱中的实体和关系向量化后，可以直接与查询向量匹配，以此检索相关知识。之后再将匹配到的知识作为生成模型的输入，生成符合语境的内容。
多模态融合：结合文档检索和知识图谱检索的结果，通过多模态数据增强生成效果。例如，首先检索到的文本段落和知识图谱内容可以一起作为生成模型的输入，帮助模型从多个角度生成回答。

## 知识图谱搭建Pipeline
### 1. 准备好txt格式文档数据
当前存在待优化问题: txt质量不高, 涉及到标题、段落的时候问题较大。
### 2. 使用Qwen模型抽取格式化三元组，存储到csv中
当前存在待优化问题: 以多大的长度单位作为句子; 适当的overlap滑动窗口; 什么样的提示词更合适
#### 方案1:
直接用LLM抽取三元组, Qwen2.5-32B-Int4提示词:
```
user:
现在你是一个知识图谱抽取器，我发给你文本你帮我抽取成格式化的三元组信息。要求:
- 返回格式为（主体，关系，客体），要求严格按照格式回复，用括号括起，并且主体、关系和客体用词精简，不要回复任何无关信息。
- 不允许使用模糊的词语作为主体和客体，例如'设备'必须具体写什么设备，'它'必须用实际的主体或客体代替。
- 确保主体和客体仅包含名词，如果有动词将动词部分放在关系里。
- 确保关系用词简练，尽可能用一个词表示关系。
```
#### 方案2:
使用LLM先进行NER, Qwen2.5-32B-Int4提示词:
```
user:
现在你是一个知识图谱抽取器，我发给你文本你帮我进行NER, 要求NER返回的标签在"运动项目", "身体部位", "器械设备", "身体素质", "人物",中选择一个, 并按如下格式返回: - 实体 单杆训练 - 标签 运动项目
assistant:
...
user:
根据我给你的原句子和你提供的NER结果, 为他们两两配对并返回关系, 例如：主体为单杆，关系为增强，客体为上肢力量，为我返回格式化结果(单杆,增强,上肢力量)，使用英文的括号和逗号
```
### 3. 三元组数据清洗、去重、合并
当前存在待优化问题: 现有的word2vec开源模型词库里根本没有垂类任务的词汇, 训练本地word2vec模型由受到数据量, 数据质量的影响导效果较差; ~~是否考虑设置一个节点词库, 最终制作的图谱节点必须是词库内的?~~ 考虑设置三元组中的主体和客体的限定属性来局限, 例如主体只能是人物、运动类型、名称等。

暂定方案:
* 对csv文件的主体、客体和关系进行完全匹配度的去重, 去重后留下的词汇全部是独一无二的
* TODO: 可以考虑下载WIKI数据集重新用word2vec训练一个模型，并使用本地数据集微调，如果本地数据集足够多了的话
* 对留下的词汇使用[WIKI预训练的word2vec模型](https://github.com/Embedding/Chinese-Word-Vectors/tree/master)转化特征向量; 由于垂类任务词汇存在大量OOV问题，使用jieba先对词汇分割成词组，然后对词组的每个词向量求均值作为词组向量; 更有甚者连jieba都分不出，那就一个字一个字计算向量然后求均值
* 节点融合: 计算相似度, 将相似度大于阈值的词向量记录下来, 之后人工标注，例如:  下肢 - 下肢肌肉  - 相似度 0.9046， 人工选择是否要合并成一个词，如果合并的话保留哪个词作为最终合并词。
* 根据节点融合的处理结果, 修改csv中的主体和客体名
* 关系消歧: 在节点融合之后，查找主体和客体相同的部分，进行关系融合或消歧, 例如(运动，提高，素质） （运动，增强，素质） （运动，降低，素质），针对关系先进行word2vec的相似度计算，查找到异常结果后记录异常值，并人工标注检查异常值。
### 4. 导入Neo4j
https://zhuanlan.zhihu.com/p/6404202845

### 5. 前端可视化
http://www.graphvis.cn/graphvis/

https://www.relation-graph.com/#/index

### 6. Graph RAG
