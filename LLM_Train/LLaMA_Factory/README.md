# LLaMA Factory 模型训练
参考内容: https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md

## 环境准备、安装
```bash
conda create -n llama python==3.10 -y
conda activate llama
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # 根据自己的cuda版本去torch官网选择合适的pytorch版本
pip install rouge_chinese
pip install jieba
pip install nltk
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```