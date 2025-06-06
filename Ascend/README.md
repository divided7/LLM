# 目录
* [工具简介](#工具简介)
* [8卡910B部署Qwen1.5-14B](#昇腾910B部署Qwen1.5-14B)
* [8卡910B部署DeepSeek_R1_distill_llama70b](#8卡910B部署DeepSeek_R1_distill_llama70b)
* [服务部署](#服务部署)
* [模型训练](#模型训练)

# 工具简介
* [CANN](https://www.hiascend.com/software/cann): CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，对上支持多种AI框架，对下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。同时针对多样化应用场景，提供高效易用的编程接口，支持用户快速构建基于昇腾平台的AI应用和业务。可近似对标NVIDIA CUDA + cuDNN。
* [MindIE](https://www.hiascend.com/software/mindie): 昇腾推理引擎，基于昇腾硬件的运行加速、调试调优、快速迁移部署的高性能深度学习推理框架，分层开放满足各类需求，统一接口使能极简开发，沉淀能力构筑极致性能。对标TRT、ONNX Runtime。
* [MindSpore](https://www.mindspore.cn): 昇思MindSpore，全场景AI框架，对标Pytorch、Tensorflow、PaddlePaddle。
    * [msModelSlim](https://gitee.com/ascend/msit/tree/dev/msmodelslim): MindSpore生态中用于模型压缩的组件，实现量化、剪枝等操作。
* [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM/tree/master): MindSpeed-LLM是基于昇腾生态的大语言模型分布式训练框架，旨在为华为昇腾芯片生态合作伙伴提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调、分布式偏好对齐以及对应的开发工具链，如：数据预处理、权重转换、在线推理、基线评估。

# 昇腾910B部署Qwen1.5-14B
参考该文档: [昇腾复现笔记](https://github.com/divided7/Ascend_Study/blob/main/昇腾复现笔记.md)

# 昇腾910B部署DeepSeek_R1_distill_llama70b
## 前期准备:
* 多卡910B服务器
* 华为昇腾对接人（必要）

## 环境准备:
参考自 [昇腾ModelZoo](https://www.hiascend.com/software/modelzoo/big-models?activeTab=language)

**驱动准备:**

一般华为给你的服务器都会配置好了，使用下列指令验证
```bash
npu-smi info # 若有返回结果则可看到驱动信息如下图所示，类似nvidia-smi
```
<img width="300" alt="image" src="https://github.com/user-attachments/assets/74f30b48-7cf2-4dd8-86c4-515daf7c534d" />

若无返回信息，参考[宿主机（服务器）驱动下载](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0006.html)

**DeepSeek-R1-Distill-Llama-70B模型准备**
模型下载到`Deepseek_R1/R1_distill_llama70b/deepseek-ai/DeepSeek-R1-Distill-Llama-70B`文件夹内
```python
# -*- coding: utf-8 -*-
# 使用modelscope国内镜像加速下载
from modelscope import snapshot_download
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', cache_dir='R1_distill_llama70b')
```


**Docker镜像准备:**
根据昇腾官方提供的`mindie`镜像（需要向华为对接人申请权限才能下载）；该镜像中内置了ATB-Models压缩包，并放置于/opt/package之下，如需使用，可从镜像中获取。
<img width="1276" alt="image" src="https://github.com/user-attachments/assets/068f2997-bce0-4228-abbc-16ca8c965eb1" />

当docker镜像下载好后可以看到:
```bash
docker images
```
<img width="1176" alt="image" src="https://github.com/user-attachments/assets/383261b3-4b55-466d-aa57-adb38ada4d3b" />

此时我们需要用该镜像运行容器，使用指令
```bash
# 这里--name可以自己修改
# 注意 -v /模型路径:/容器内路径:rw 
# 最后一行的结果是根据docker image ls查看到的`{REPOSITORY:TAG} bash`
docker run -it -d --net=host --shm-size=1g \
    --privileged \
    --name MindIE800I \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /mnt/nvme0/luyuxi/Deepseek_R1/R1_distill_llama70b/deepseek-ai:/deepseek-ai:rw \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts bash
```
此时使用指令`docker ps`可以看到：
<img width="1515" alt="image" src="https://github.com/user-attachments/assets/0a70e237-54c6-4faa-aea5-c69501b0a893" />

进入容器：
```bash
docker exec -it ${容器名称} bash # 例如我这里容器名字是MindIE800I， 使用指令： docker exec -it MindIE800I bash
```

查看容器内是否正确存在模型(确认运行容器时指定的路径是否在容器内可以正常访问):
```bash
ls /deepseek-ai/DeepSeek-R1-Distill-Llama-70B # 如果有正常显示一堆.safetensors文件则说明正常
```

安装msModelSlim工具([参考链接](https://gitee.com/ascend/msit/tree/dev/msmodelslim)):
```
dnf install git
git clone https://gitee.com/ascend/msit.git
cd msit/msmodelslim
bash install.sh
cd /
```
<img width="1185" alt="image" src="https://github.com/user-attachments/assets/99c57675-a76d-4de9-8368-2f161b960a5b" />

模型量化:
```bash
# 进入atb路径
cd /opt/package # 镜像中内置了ATB-Models压缩包，并放置于/opt/package之下
mkdir atb
tar -xf Ascend-mindie-atb-models_2.0.T3_linux-aarch64_py311_torch2.3.1-abi1.tar.gz -C atb # 这里根据python版本和torch版本选型，解压到atb文件夹下
cd atb
# 设置CANN包的环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 关闭虚拟内存
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
sed -i '167s/m3/m4/' examples/models/llama3/generate_quant_weight.sh
# DeepSeek-R1-Distill-Llama-70B量化 bf16，有回退层，antioutlier使用m4算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在NPU上进行运算
# -src指定hf下载的模型路径, -dst指定W8A8量化权重路径 （这个过程有点慢 要半个多小时）
bash examples/models/llama3/generate_quant_weight.sh -src /deepseek-ai/DeepSeek-R1-Distill-Llama-70B -dst /deepseek-ai/DeepSeek-R1-Distill-Llama-70B-W8A8 -type llama3.1_70b_instruct_bf16_w8a8
```
在模型量化的时候最终报错, 但似乎不影响模型生成, 详见[issues](https://gitee.com/ascend/msit/issues/IBQOGM?from=project-issue): 
```
  File "/usr/local/Ascend/atb-models/atb_llm/utils/file_utils.py", line 141, in check_file_safety
    raise FileExistsError("The file is expected not to exist, but it already does. "
FileExistsError: The file is expected not to exist, but it already does. Please check the input path:/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-W8A8/config.json.
[ERROR] 2025-03-04-15:54:22 (PID:951, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```

## 简单推理验证
```bash
# --model_path选择刚才量化模型的路径
torchrun --nproc_per_node 8 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path /deepseek-ai/DeepSeek-R1-Distill-Llama-70B-W8A8 \
         --input_texts 'What is deep learning?' \
         --max_output_length 20
```
<img width="1512" alt="image" src="https://github.com/user-attachments/assets/9b11c692-7aea-455c-9799-3c3fa278d8b5" />
从上图可以看到模型正常输出了，但是由于限制了输出长度所以被提前截断

## 性能测试
略 详见[原文档](https://www.hiascend.com/software/modelzoo/models/detail/ee3f9897743a4341b43710f8d204733a)
```bash
cd atb/tests/modeltest/
# 测试未量化模型（没什么问题）
bash run.sh pa_bf16 performance [[256,256]] 1 llama ${未量化模型路径: /deepseek-ai/DeepSeek-R1-Distill-Llama-70B} 8
# 测试量化模型 (这里有问题，第一个参数{model_type}_{data_type}里，model_type只支持basic, pa, fa; data_type里只支持fp16和bf16, 但该量化模型为8位的)
bash run.sh pa_bf16 performance [[256,256]] 1 llama ${量化模型路径: /deepseek-ai/DeepSeek-R1-Distill-Llama-70B-W8A8} 8
```

## 服务
### 启用非量化模型服务
```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
chmod -R 750 /PATH/TO/非量化deepseek模型路径 # 确保路径权限正确，因为这个文件夹是容器外部来的
```
修改下面参数
```
...
"ipAddress" : "127.0.0.1",
"managementIpAddress" : "127.0.0.2",
"port" : 1025, # 自定义 也可以不改
"managementPort" : 1026, # 自定义 也可以不改
"metricsPort" : 1027, # 自定义 也可以不改
...
"httpsEnabled" : false,
...
},
...
"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"modelName" : "llama",
"modelWeightPath" : "/data/datasets/DeepSeek-R1-Distill-Llama-70B", # 选择非量化模型路径
"worldSize" : 8,
...
```
启动服务
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

测试服务
```bash
curl 127.0.0.1:1025/generate -d '{
"prompt": "介绍一下什么是LLM",
"max_tokens": 65536,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "llama"
}'
```
结果如下
<img width="1489" alt="image" src="https://github.com/user-attachments/assets/affc96c2-4a0d-4237-8c60-92439dbceb65" />
### 启用量化模型服务
```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```
修改里面的内容，和非量化模型服务的部分都一样，只有“modelWeightPath”要修改成量化模型路径,然后重新启动服务
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```
结果报错:
```
[root@host-5jl98c bin]# ./mindieservice_daemon
LLMInferEngine failed to init LLMInferModels
ERR: Failed to init endpoint! Please check the service log or console output.
Killed
```
怀疑是在量化模型过程中的报错中断导致的问题，等华为那边给[issues](https://gitee.com/ascend/msit/issues/IBQOGM?from=project-issue)结果再看看，先挂起

# 服务部署
**TODO**
```
临时笔记
参考:
https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0004.html
https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0026.html
```
# 模型训练
**TODO**
使用**MindSpeed-LLM**（原名ModelLink）框架进行模型训练。
```
临时笔记
参考:
https://gitee.com/ascend/MindSpeed-LLM/tree/e77800f8c654c4eb89f1012774a829e3e6f1e7f4
https://www.hiascend.com/software/modelzoo/models/detail/52226fe32f7c472084aec868ce55f00c
```
## MindSpeed-LLM安装
由于这里使用的是华为给的镜像，像是CANN、torch、torch-npu已经自带，完整安装步骤参考[该链接](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/install_guide.md)，这里只记录基于该镜像的安装步骤
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # source ascend-toolkit环境变量

# 安装MindSpeed加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # checkout commit from MindSpeed core_r0.8.0 in 2025.02.26
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 准备MindSpeed-LLM及Megatron-LM源码
git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
pip install -r requirements.txt  # 安装其余依赖库

yum install patch gcc gcc-c++ # 安装patch, gcc, g++

# 安装apex, 参考: https://gitee.com/ascend/apex
git clone -b master https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.11 # 这里官方是些的只支持到3.10，但是下方又写的支持3.11; 该指令从github拉取仓库，注意网络问题，如果在#include <torch/extension.h>报错参考pr: https://gitee.com/ascend/apex/pulls/129

# 如果上面bash scripts/build.sh --python=3.11报错#include <torch/extension.h>相关则进行下面操作 否则跳过:
# -- 报错#include <torch/extension.h> --
cd ..
rm -rf apex
pip show torch # 查看torch的Location，例如我的是 /usr/local/lib64/python3.11/site-packages
vim patch/npu.patch
# 修改 patch/npu.patch 直接修改第2649行，把 package_dir 手工指定为上面torch的位置，即我这个例子中的 /usr/local/lib64/python3.11/site-packages
bash scripts/build.sh --python=3.11
# -- 报错#include <torch/extension.h> --

cd apex/dist
pip install *.whl # 安装当前路径的whl文件
```

## 数据准备和处理
[Alpaca风格数据集](https://gitee.com/ascend/MindSpeed-LLM/blob/e77800f8c654c4eb89f1012774a829e3e6f1e7f4/docs/features/alpaca_dataset.md)

这里用Alpaca数据集`train-00000-of-00001-a09b74b3ef9c3b56.parquet`，[下载链接](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), 需要梯子（24.2MB）。

将`train-00000-of-00001-a09b74b3ef9c3b56.parquet`放到容器MindSpeed-LLM路径的`MindSpeed-LLM/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet`, `MindSpeed-LLM/dataset`路径需要手工创建。


数据处理参见MindSpeed-LLM中的`MindSpeed-LLM/examples/mcore/deepseek_r1_distill_llama`路径下的[data_convert_distill_llama_instruction.sh](https://gitee.com/ascend/MindSpeed-LLM/blob/e77800f8c654c4eb89f1012774a829e3e6f1e7f4/examples/mcore/deepseek_r1_distill_llama/data_convert_distill_llama_instruction.sh)
为了方便更新参数，这里直接把sh脚本内容掏出来手工运行，下面的代码均来自bash脚本

```bash
cd /PATH/TO/MindSpeed-LLM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

ls dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # 检查alpaca数据集位置是否正确

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path /deepseek-ai/DeepSeek-R1-Distill-Llama-70B/ \ # 替换成模型tokenizer.json所在的路径，即一般是hf模型权重所在路径
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type deepseek3 \
    --seq-length 8192 \
```
此时处理好的数据会保存在`MindSpeed-LLM/finetune_dataset`
<img width="1049" alt="image" src="https://github.com/user-attachments/assets/b2126a48-22c2-44e0-a735-35a096fb6f48" />

## 模型 hf -> mcore
[参考例子](https://gitee.com/ascend/MindSpeed-LLM/blob/e77800f8c654c4eb89f1012774a829e3e6f1e7f4/examples/mcore/deepseek_r1_distill_llama/ckpt_convert_distill_llama_hf2mcore.sh)
本质代码：
```bash
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py --use-mcore-models --model-type-hf llama2 --model-type GPT --load-model-type hf --save-model-type mg --params-dtype bf16 --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --load-dir /deepseek-ai/DeepSeek-R1-Distill-Llama-70B/ --save-dir /deepseek-ai/DeepSeek-R1-Distill-Llama-70B-mcore/ --tokenizer-model /deepseek-ai/DeepSeek-R1-Distill-Llama-70B/tokenizer.json
```
报错Bus error (core dumped)，已提[issues](https://gitee.com/ascend/MindSpeed-LLM/issues/IBRCA4?from=project-issue&search_text=Bus+error)，下次再看看
