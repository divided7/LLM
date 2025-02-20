<img width="1176" alt="image" src="https://github.com/user-attachments/assets/a06cd07b-d337-4fa7-bb77-7e836c7a2aa5" /># 昇腾910B部署Qwen1.5-14B
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
docker run -it -d --net=host --shm-size=1g \
    --privileged \
    --name MindIE800I \ # name可以自己改
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path-to-weights:/path-to-weights:ro \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts bash # swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts 是根据上面`docker images`指令看到的镜像名字设置的
```
此时使用指令`docker ps`可以看到：
<img width="1515" alt="image" src="https://github.com/user-attachments/assets/0a70e237-54c6-4faa-aea5-c69501b0a893" />

进入容器：
```bash
docker exec -it ${容器名称} bash # 例如我这里容器名字是MindIE800I， 使用指令： docker exec -it MindIE800I bash
```
模型量化:
```bash
cd /opt/package # 镜像中内置了ATB-Models压缩包，并放置于/opt/package之下
tar -xf Ascend-mindie-atb-models_2.0.T3_linux-aarch64_py311_torch2.3.1-abi1.tar.gz # 这里根据python版本和torch版本选型

```
