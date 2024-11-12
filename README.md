# LLM & KG
## Preface
大语言模型已经能很好的实现问答、文本输出，但其输出结果可能有时候并没有足够的依据，因此利用 **RAG(Retrieval-Augmented Generation)** 技术，通过在生成回答之前先检索相关知识，从而使生成的内容更加可靠，适用于问答系统、文档生成、客户支持等任务。这样的相关知识可以是文档，也可以是知识图谱。

本文旨在从0开始研究LLM和知识图谱的融合: LLM赋能知识图谱的构建，知识图谱赋能LLM的准确性。

在研究开始前，需了解RAG，NER，LLM， Knowledge graph。

## 当前待解决问题
* 如何实现模型并行？vLLM似乎只支持数据并行，当单卡显存不足以推理模型时即使多卡也无法推理
* 如何微调LLM？ 参考: https://qwen.readthedocs.io/zh-cn/latest/training/SFT/llama_factory.html

## RAG
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
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                   | 4GB     |       | 偶尔可靠         | 0.3  |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                       | 7GB     |       | 有些可靠         | 0.5  |
| [Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)   | 2.3GB   |  9.2GB| 有些可靠         | 0.5  |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)                       | 15GB    |       | 有些可靠         | 0.5  |
| [Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)   | 5.4GB   | 16.5GB| 有些可靠         | 0.5  |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)                     | 29GB    |       |                |      |
| [Qwen2.5-14BInstruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)  |         | 18.3GB|                |      |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)                     | 65GB    |       |                |      |
| [Qwen2.5-32B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4)                     |         |21.2GB (max_model_len=48)|                |      |
| [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)                     | 145GB   |       |                |      |
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

## Knowledge Graph
将 Neo4j 仓库添加到系统的apt源列表
尝试了很多民间方案都安装失败了, 还是找的官方教程:
https://neo4j.com/docs/operations-manual/current/installation/linux/debian/
具体来说：
```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg

echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list

sudo apt-get update

sudo apt-get install neo4j=1:5.25.1
```

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
