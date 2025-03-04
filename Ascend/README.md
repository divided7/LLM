# 昇腾910B部署Qwen1.5-14B
https://github.com/divided7/Ascend_Study/blob/main/昇腾复现笔记.md

# 昇腾910B部署DeepSeek
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

## 服务
```bash
vi /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```
<img width="979" alt="image" src="https://github.com/user-attachments/assets/91a596c0-65ed-49c0-b812-4240620cc995" />
