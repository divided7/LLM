# LoRA训练
LoRA的思想: 

### Step 1: 创建基础模型

```python
import torch
import torch.nn as nn

# 基础模型：一个简单的前馈网络
class BaseModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return self.output(x)
```

---

### Step 2: 手动添加 LoRA 模块

LoRA 的基本思想是：  
在原始权重 `W` 的基础上加一个低秩矩阵分解：`W_delta = A @ B`

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear  # 不改变原始权重

        # 冻结原始权重（可选）
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # 添加 LoRA 的 A, B 层（低秩分解）
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # 缩放因子
        self.scaling = alpha / r

    def forward(self, x):
        return self.original_linear(x) + self.scaling * self.lora_B(self.lora_A(x))
```

---

### Step 3: 替换原始网络中的某一层为 LoRA 层

```python
# 初始化模型
model = BaseModel()

# 替换 model.linear 为带 LoRA 的版本
model.linear = LoRALinear(model.linear, r=4, alpha=8)

# 测试
x = torch.randn(2, 10)
output = model(x)
print(output)
```

---

### Training Tip

- 现在只有 `lora_A` 和 `lora_B` 是可训练的，原始的 `linear.weight` 是被冻结的。
- 你只需要优化这两个 LoRA 层的参数，就能“微调”模型了。

---

### 优点总结

| 优点                 | 说明                                         |
|----------------------|----------------------------------------------|
| 显存占用少            | 只训练小矩阵 A 和 B                          |
| 快速训练              | 减少参数更新量                               |
| 好迁移                | 预训练模型保持不变，适合多任务多语言场景     |
| 可选合并              | 推理前可以合并为普通 Linear 层                |

---

如果你需要我帮你扩展这个例子，比如训练、保存、合并 LoRA 权重等，也可以继续说 😄