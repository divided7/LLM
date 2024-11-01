import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import argparse
import warnings
from datetime import datetime

# 屏蔽所有警告
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

parser=argparse.ArgumentParser()
parser.add_argument('--p',default="int4",type=str, help='precision: fp16(default) / int4')
parser.add_argument('--m',default="3B",type=str, help='model: 3B(default) / 7B')
args=parser.parse_args()


model_name = args.m + "/" + args.p
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化对话历史
conversation_history = [
    {"role": f"system", "content": "You are Qwen2.5-{args.m}-{args.p}, created by Yuxi Lu. You are a helpful assistant."}
]

def generate_response(prompt):
    # 将用户消息添加到对话历史
    conversation_history.append({"role": "user", "content": prompt})

    # 应用聊天模板并进行分词
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 使用 `generate` 方法生成文本
    output_sequence = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        return_dict_in_generate=True,
        return_legacy_cache=True,
        output_scores=False,  # 输出每一步的分数
    )

    # 提取生成的 token
    generated_ids = output_sequence.sequences[0]
    generated_ids = generated_ids[:-1] # 删掉截止符号
    # 只解码最后的 token，避免重复
    response = ""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\033[92m({current_time}) Qwen2.5-{args.m}-{args.p}: \033[0m")
    for token in generated_ids[len(model_inputs.input_ids[0]):]:
        token_str = tokenizer.decode(token)
        response += token_str
        print(token_str, end='', flush=True)  # 实时输出，确保字符及时显示
        time.sleep(0.05)  # 控制每个字符的输出速度
    print("")
    # 将最终响应添加到对话历史
    conversation_history.append({"role": "assistant", "content": response})
    
    return response

print(f"\033[90m欢迎与 Qwen2.5-{args.m}-{args.p} 聊天！输入 'exit' 退出。\033[97m ")
while True:
    current_time = datetime.now().strftime("%H:%M:%S")
    user_input = input(f"\033[91m({current_time}) 你: \033[0m \n")
    if user_input.lower() == 'exit':
        break
    
    # 生成并打印助手的响应
    response = generate_response(user_input)

