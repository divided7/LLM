import requests
from datetime import datetime

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
    "temperature": 0.7
}

start_time = datetime.now()
print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

response = requests.post(f"{base_url}/completions", headers=headers, json=data)
result = response.json()

end_time = datetime.now()
print("结束时间:", end_time.strftime("%Y-%m-%d %H:%M:%S"))

print("生成结果:")
print(result["choices"][0]["text"])

