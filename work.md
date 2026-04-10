# ConvNeXt V2 学习笔记

## 1. ConvNeXtV2 类详解 (models/convnextv2.py)

### 1.1 类定义和初始化参数

```python
class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): 输入图像通道数，默认3 (RGB)
        num_classes (int): 分类头输出类别数，默认1000 (ImageNet)
        depths (tuple): 每个stage的Block数量，默认[3, 3, 9, 3]
        dims (int): 每个stage的特征维度，默认[96, 192, 384, 768]
        drop_path_rate (float): 随机深度率（正则化），默认0
        head_init_scale (float): 分类头权重初始化缩放，默认1
    """
```

**参数详解：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_chans` | int | 3 | 输入图像通道数，RGB图像为3 |
| `num_classes` | int | 1000 | 分类类别数，ImageNet为1000 |
| `depths` | list | [3,3,9,3] | 4个stage各自的Block数量 |
| `dims` | list | [96,192,384,768] | 4个stage各自的特征维度 |
| `drop_path_rate` | float | 0.0 | 随机深度丢弃率，用于正则化 |
| `head_init_scale` | float | 1.0 | 分类头权重初始化缩放因子 |

---

### 1.2 下采样层 (downsample_layers)

#### Stem（第1层 - Patch嵌入）

```python
stem = nn.Sequential(
    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
)
```

**问题：为什么用 4×4 卷积，stride=4？**

**回答：**
- **作用**：将图像切分为 4×4 的非重叠patches，实现类似ViT的patch嵌入
- **效果**：输入 (B, 3, 224, 224) → 输出 (B, 96, 56, 56)
- **计算**：224 / 4 = 56，空间分辨率降为1/4，通道升为96
- **优势**：
  - 相比ViT的线性投影，卷积实现保留了局部空间关系
  - 可学习的投影方式，比手工设计的patch extraction更灵活
  - 非重叠patches减少计算冗余

#### 阶段间下采样（第2-4层）

```python
for i in range(3):
    downsample_layer = nn.Sequential(
        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
    )
```

**问题：为什么用 2×2 卷积而不是池化？**

**回答：**
- **可学习下采样**：卷积参数可学习，池化是固定的
- **通道数扩展**：同时实现空间下采样和通道扩展
- **现代设计趋势**：从ResNet的池化到ConvNeXt的卷积，提升表达能力

---

### 1.3 特征提取 Stages

```python
self.stages = nn.ModuleList()
dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
```

**Stage配置表：**

| Stage | 输入尺寸 | 输出尺寸 | Block数 | 特征维度 | 下采样 |
|-------|----------|----------|---------|----------|--------|
| 1 | (B,3,224,224) | (B,96,56,56) | depths[0] | dims[0]=96 | 4× (stem) |
| 2 | (B,96,56,56) | (B,192,28,28) | depths[1] | dims[1]=192 | 2× |
| 3 | (B,192,28,28) | (B,384,14,14) | depths[2] | dims[2]=384 | 2× |
| 4 | (B,384,14,14) | (B,768,7,7) | depths[3] | dims[3]=768 | 2× |

**问题：什么是 drop_path_rate？如何计算每个Block的rate？**

**回答：**
- **随机深度 (Stochastic Depth)**：训练时随机丢弃某些Block，用于正则化
- **线性增长**：`torch.linspace(0, drop_path_rate, sum(depths))`
  - 浅层Block丢弃率低，深层丢弃率高
  - 例如 drop_path_rate=0.1，18个Block：rates = [0, 0.006, 0.011, ..., 0.1]
- **效果**：防止深层网络过拟合，提升泛化能力
- **推理时**：自动关闭，所有Block都参与计算

---

### 1.4 分类头

```python
self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
self.head = nn.Linear(dims[-1], num_classes)
```

**问题：为什么用 LayerNorm 而不是 BatchNorm？**

**回答：**
- **Transformer传统**：LayerNorm在Transformer中表现更好
- **批量无关**：不依赖batch size，对训练和推理更稳定
- **通道归一化**：对每个样本的所有通道进行归一化
- **现代趋势**：从CNN的BN向Transformer的LN转变

**head_init_scale 的作用：**
```python
self.head.weight.data.mul_(head_init_scale)
self.head.bias.data.mul_(head_init_scale)
```
- 微调时通常设为0.001，分类头初始输出接近0
- 训练初期更稳定，避免大的梯度更新

---

### 1.5 权重初始化

```python
def _init_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)
```

**问题：什么是 trunc_normal_？为什么用 std=0.02？**

**回答：**
- **截断正态初始化**：从正态分布采样，但截断在 ±2 标准差之外
- **std=0.02**：较小的标准差保证初始权重不会太大
- **Transformer传统**：ViT、BERT等都使用类似初始化
- **稳定性**：防止训练初期梯度爆炸或消失

---

### 1.6 前向传播流程

#### forward_features

```python
def forward_features(self, x):
    for i in range(4):
        x = self.downsample_layers[i](x)
        x = self.stages[i](x)
    return self.norm(x.mean([-2, -1]))
```

**详细流程：**

```
输入: x (B, 3, 224, 224)
    ↓ downsample_layers[0] (Stem: 4×4 Conv, stride=4)
x: (B, 96, 56, 56)
    ↓ stages[0] (3个Block)
x: (B, 96, 56, 56)
    ↓ downsample_layers[1] (2×2 Conv, stride=2)
x: (B, 192, 28, 28)
    ↓ stages[1] (3个Block)
x: (B, 192, 28, 28)
    ↓ downsample_layers[2] (2×2 Conv, stride=2)
x: (B, 384, 14, 14)
    ↓ stages[2] (9个Block)  ← 主要计算量在这里
:x: (B, 384, 14, 14)
    ↓ downsample_layers[3] (2×2 Conv, stride=2)
x: (B, 768, 7, 7)
    ↓ stages[3] (3个Block)
x: (B, 768, 7, 7)
    ↓ x.mean([-2, -1]) 全局平均池化 (H,W维度)
x: (B, 768)
    ↓ LayerNorm
输出: (B, 768)
```

#### forward

```python
def forward(self, x):
    x = self.forward_features(x)
    x = self.head(x)  # (B, 768) → (B, num_classes)
    return x
```

---

## 2. Block 类详解

```python
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```

### 2.1 深度可分离卷积 (Depthwise Separable Convolution)

```python
self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
```

**问题：groups=dim 是什么意思？**

**回答：**
- **深度可分离卷积**：每个通道单独做卷积，不混合通道信息
- **groups=dim**：输入通道分组，每组1个通道，共dim组
- **参数量**：7×7×dim = 49×dim（标准卷积是 7×7×dim×dim）
- **作用**：空间特征提取，每个通道独立学习空间模式
- **感受野**：7×7，较大感受野捕获更多上下文

### 2.2 Pointwise Convolution (1×1 Conv)

```python
self.pwconv1 = nn.Linear(dim, 4 * dim)  # 扩展
self.pwconv2 = nn.Linear(4 * dim, dim)  # 压缩
```

**问题：为什么用 Linear 而不是 Conv2d？**

**回答：**
- **等价性**：1×1 Conv2d 等价于作用于每个像素的 Linear
- **实现简洁**：Linear更直观表达"逐点"操作
- **先permute**：(B,C,H,W) → (B,H,W,C)，通道变最后一维

**流程：**
```python
x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
x = self.pwconv1(x)        # (B, H, W, C) → (B, H, W, 4C)
x = self.act(x)            # GELU激活
x = self.grn(x)            # GRN归一化
x = self.pwconv2(x)        # (B, H, W, 4C) → (B, H, W, C)
x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
```

### 2.3 GRN (Global Response Normalization)

```python
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)  # 计算每个通道的全局L2范数
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  # 归一化
        return self.gamma * (x * Nx) + self.beta + x
```

**问题：GRN的作用是什么？与LayerNorm/ BatchNorm有什么区别？**

**回答：**

| 归一化方法 | 归一化维度 | 作用 |
|-----------|-----------|------|
| BatchNorm | (N,H,W) | 对batch内所有样本的同一通道归一化 |
| LayerNorm | (C,H,W) | 对每个样本的所有通道和空间归一化 |
| GRN | 通道间竞争 | 增强通道间的特征竞争 |

**GRN独特之处：**
1. **跨通道竞争**：通过 `Gx.mean(dim=-1)` 计算所有通道的平均响应
2. **自适应加权**：响应大的通道被抑制，响应小的通道被增强
3. **可学习参数**：gamma和beta学习如何调整这种竞争
4. **V2核心创新**：替代了V1的Layer Scale，效果更好

**数学公式：**
```
Gx = ||x||_2  # 每个通道的L2范数 (N,1,1,C)
Nx = Gx / mean(Gx)  # 相对于平均响应的归一化
output = γ * (x * Nx) + β + x  # 可学习的残差连接
```

---

## 3. 不同尺寸模型配置

```python
def convnextv2_atto(**kwargs):   # 3.7M 参数
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):  # 5.2M 参数
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):   # 9.1M 参数
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):   # 15.6M 参数
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):   # 28.6M 参数
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):   # 89M 参数
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):  # 198M 参数
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):   # 660M 参数
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
```

**配置对比：**

| 模型 | depths | dims | 参数量 | FLOPs | ImageNet精度 |
|------|--------|------|--------|-------|-------------|
| Atto | [2,2,6,2] | [40,80,160,320] | 3.7M | 0.55G | 76.7% |
| Tiny | [3,3,9,3] | [96,192,384,768] | 28.6M | 4.47G | 83.0% |
| Base | [3,3,27,3] | [128,256,512,1024] | 89M | 15.4G | 84.9% |
| Huge | [3,3,27,3] | [352,704,1408,2816] | 660M | 115G | 86.3% |

**规律：**
- **小模型**：depths用[2,2,6,2]，dims较小
- **大模型**：depths用[3,3,27,3]，dims较大
- **主要差异在Stage 3**：小模型6个Block，大模型27个Block

---

## 4. 与 ResNet 的对比

| 特性 | ResNet-50 | ConvNeXt V2-T |
|------|-----------|---------------|
| Stem | 7×7 Conv, stride=2 + MaxPool | 4×4 Conv, stride=4 |
| 下采样 | 3×3 Conv, stride=2 | 2×2 Conv, stride=2 |
| 基本单元 | Bottleneck Block (1×1→3×3→1×1) | ConvNeXt Block (7×7 dw → 1×1 → 1×1) |
| 激活函数 | ReLU | GELU |
| 归一化 | BatchNorm | LayerNorm |
| 特殊层 | SE模块 (部分) | GRN (所有Block) |
| 输出 | Global Pool + FC | Global Pool + LayerNorm + FC |

---

## 5. 关键问题总结

### Q1: 为什么 Patch 大小是 4×4？

**A:** 
- 平衡效率和精度：太小计算量大，太大信息损失多
- 4×4是ConvNeXt系列的标准选择，与Swin Transformer一致
- 224×224图像 → 56×56特征图，保留足够的空间细节

### Q2: depths 为什么是 [3,3,9,3] 而不是 [3,3,3,9]？

**A:**
- **Stage 3** 空间分辨率14×14，计算量适中
- 分辨率太高（56×56）Block太多计算爆炸
- 分辨率太低（7×7）空间信息不足
- 类似ResNet设计，更多层在中间分辨率

### Q3: 为什么用 GELU 而不是 ReLU？

**A:**
- **Transformer传统**：GELU在Transformer中表现更好
- **平滑性**：GELU是平滑激活函数，梯度更稳定
- **自监督训练**：GELU在MAE等自监督任务中表现更好
- **现代趋势**：从ReLU向GELU/Mish/Swish转变

### Q4: Drop Path 和 Dropout 有什么区别？

**A:**

| 特性 | Dropout | Drop Path (Stochastic Depth) |
|------|---------|------------------------------|
| 作用位置 | 特征图/权重 | 整个Block |
| 训练时 | 随机置零部分神经元 | 随机跳过整个残差Block |
| 推理时 | 缩放权重 | 使用所有Block |
| 主要用途 | 全连接层正则化 | 深度网络正则化 |
| 效果 | 防止特征共适应 | 防止深层网络过拟合 |

**Drop Path公式：**
- 训练：以概率 `drop_path` 跳过Block，`output = input + drop_path(block(input))`
- 推理：`output = input + block(input)`

### Q5: 为什么 LayerNorm 用 channels_first 格式？

**A:**
- **性能优化**：PyTorch中channels_first格式(NCHW)卷积更快
- **历史原因**：CNN传统使用NCHW，Transformer使用NHWC
- **代码简洁**：ConvNeXt的Block内部才permute到NHWC，外部保持NCHW

---

## 6. 学习建议

1. **动手画图**：画出数据流图，标注每层的形状变化
2. **对比实验**：对比ResNet、ViT、Swin Transformer的设计差异
3. **阅读源码**：理解 `Block` 类的每个操作
4. **可视化特征**：加载预训练权重，可视化中间层特征
5. **消融实验**：修改depths/dims，观察参数量和计算量变化
