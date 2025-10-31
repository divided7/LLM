# 环境配置
```bash
conda create -n vllm python==3.10
conda activate vllm
pip install vllm==0.6.1 -i https://mirrors.aliyun.com/pypi/simple # 好像在Qwen2.5介绍页看到不支持当前最新版本(v0.6.3)
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
pip install -U accelerate bitsandbytes datasets peft transformers -i https://mirrors.aliyun.com/pypi/simple
pip install auto_gptq -i https://mirrors.aliyun.com/pypi/simple
pip install optimum -i https://mirrors.aliyun.com/pypi/simple
mkdir Qwen2.5 && cd Qwen2.5
```

# 下载模型
```
modelscope download --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --local_dir ./3B/int4
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --local_dir ./3B/int4
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./3B/fp16
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./3B/fp16
```

# 使用
```bash
# gpu_memory_utilization  enforce_eager  memory
# 0.9                     False          11GB  
# 0.9                     True           9.7GB
# 0.7                     True           7.2GB
# 0.6                     True           6GB
# 0.5                     True           报错, 提示需要降低max_model_len 默认32768: Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
# 0.4 (不能再低了)         True    max_model_len=16       5.6G
vllm serve 3B/int4 --dtype auto --api-key 123 --port 8008 --max-model-len 32768 --gpu-memory-utilization 0.8
```
如果要多卡并行:
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve DeepSeek_R1_Distill_Qwen_7B --dtype auto --api-key 123 --port 3003 --max_model_len 2048 --tensor_parallel_size 2
```


# 流水式推理
参考py代码文件夹`流式调用`

# KV-cache加速固定前缀提示词
参考文件夹`固定前缀提示词`
* vllm不支持使用CLI的kv-cache, 要使用kv-cache的话必须用python启动VLLM
  