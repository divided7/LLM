import requests
import sseclient
from datetime import datetime
import json

base_url = "http://172.16.200.150:3003/v1"
api_key = "123"
model_name = "DeepSeek_R1_Distill_Qwen_7B"

prompt = "背诵一下赤壁赋"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": model_name,
    "prompt": prompt,
    "max_tokens": 1000,
    "temperature": 0.7,
    "stream": True  # 开启流式输出
}

start_time = datetime.now()
print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

# 使用 requests + sseclient 监听流式返回
with requests.post(f"{base_url}/completions", headers=headers, json=data, stream=True) as response:
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.data == "[DONE]":
            break
        json_data = json.loads(event.data)
        print(json_data["choices"][0]["text"], end="", flush=True)  # 实时输出生成内容

end_time = datetime.now()
print("\n结束时间:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
