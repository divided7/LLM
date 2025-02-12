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
根据昇腾官方提供的`mindie`镜像（需要向华为对接人申请权限才能下载）
<img width="1276" alt="image" src="https://github.com/user-attachments/assets/068f2997-bce0-4228-abbc-16ca8c965eb1" />
