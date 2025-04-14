# LoRA训练
LoRA（Low-Rank Adaptation）的思想: 

不修改原始预训练模型的权重, 而是在特定层上添加 可训练的低秩权重（LoRA Adapter）。在训练完成之后，模型实际上由 原始模型 + LoRA adapter 组成。

LoRA的特点注定了它是一个微调模型，相比动辄几亿的参数，LoRA 微调引入的参数 非常少（通常小于1%）

LoRA 本质上是一个模块扩展，可以：
* 加载预训练模型时不加载 LoRA；
* 训练多个 LoRA 分支用于不同任务；
* 在推理阶段把 LoRA 合并（merge）到原模型里，提升性能。

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
- 只需要优化这两个 LoRA 层的参数，就能“微调”模型了。

---

---

### Step 4.1: 使用方式一：推理时保留 LoRA（推荐调试或多任务场景）

#### 1. 保留 `LoRALinear` 结构
如果没有合并 LoRA 到原模型，只需像普通模型一样调用即可：

```python
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

#### 2. 加载保存的权重
假设保存了模型：

```python
# 保存
torch.save(model.state_dict(), "lora_model.pth")

# 加载
model.load_state_dict(torch.load("lora_model.pth"))
```

如果训练了多个 LoRA 分支，也可以加载不同的权重来快速适配任务。

---

### Step 4.1: 使用方式二：合并 LoRA 到原始模型（部署更高效）

也可以选择把 LoRA 的权重“合并回”原始 `Linear` 层中，然后去掉 `lora_A/B`，这样变成一个纯 `Linear`，推理更快、更省资源。

#### 手动合并 LoRA 到原始权重

```python
def merge_lora_weights(lora_layer):
    # W + alpha / r * B @ A
    W = lora_layer.original_linear.weight.data
    A = lora_layer.lora_A.weight.data
    B = lora_layer.lora_B.weight.data
    merged = W + lora_layer.scaling * torch.matmul(B, A)
    lora_layer.original_linear.weight.data = merged

    # 替换成原始 Linear 层
    return lora_layer.original_linear

model.linear = merge_lora_weights(model.linear)
```
这之后就得到了一个**标准 Linear 模型**，不再依赖 LoRA 结构，适合部署。

---

### 优点总结

| 优点                 | 说明                                         |
|----------------------|----------------------------------------------|
| 显存占用少            | 只训练小矩阵 A 和 B                          |
| 快速训练              | 减少参数更新量                               |
| 好迁移                | 预训练模型保持不变，适合多任务多语言场景     |
| 可选合并              | 推理前可以合并为普通 Linear 层                |

---