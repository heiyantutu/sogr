# SOGR：面向 LLM 的语义对象图运行时

**语言：** [English](README.md) · **简体中文**

**论文：** [英文 Markdown](paper/SOGR.md) · [简体中文 Markdown](paper/SOGR.zh.md) · [PDF](paper/SOGR.pdf)

这是一份 **LLM**（大语言模型 / large language model）架构草案：用结构化增量计算做长上下文 Transformer，作为稠密 token 级 KV cache 的替代假设。  
独立研究草案 · 版本 0.3 · 2026 年 9 月

| | |
|---|---|
| **名称** | SOGR（Semantic Object Graph Runtime，语义对象图运行时） |
| **领域** | LLM · 大语言模型 · Transformer · 长上下文 · KV cache |
| **作者** | Zeping Tu，前端工程师 |
| **类型** | LLM 研究提案 + CPU 笔记本基线实验 |
| **不是** | GPU 规模训练结果，也不是可上线的大模型产品 |
| **仓库** | [github.com/heiyantutu/sogr](https://github.com/heiyantutu/sogr) |

**SOGR** 是一套 **LLM 运行时**假设：把语义结构当成**可缓存的中间表示**，而不是在每个 Transformer 层上对 token 序列重新发现关系。大语言模型先编译出 **语义对象图（SOG）**，沿图执行计算；局部变化时只重算**脏依赖子图**。

研究方向来自**前端工程**。**React** 和 **Vue** 会编译结构化视图，只弄脏依赖子树，持久化的是**对象**而不是像素。SOGR 问的是：语言模型能否沿用同一套契约——结构编译一次、增量执行、持久化对象，而不是 token 级 KV 缓存。

> 当前实验上限是无独显笔记本上的 DistilGPT-2 和两个 0.5B 指令模型。这是实验地板，不能当成 7B 级模型或 16K–128K 上下文上的证据。

---

## 常见问题

### SOGR 是什么？

SOGR（语义对象图运行时）是面向长上下文 **LLM / 大语言模型** 的架构假设。低频全局注意力负责构造或修复语义对象图；高频、受图约束的线性算子沿图传播状态。结构化缓存按依赖失效下游节点，类似于响应式 UI 弄脏组件子树。

### 这是 LLM / 大语言模型项目吗？

是。本仓库是 LLM 研究：Transformer 语言模型、注意力、KV cache、长上下文，以及所提的对象图运行时。不是聊天应用。本机实验用 DistilGPT-2 和 0.5B 指令 LLM，跑在 CPU 上。

### 和 KV cache 有什么不同？

常规 KV cache 按「每个 token、每一层」存放隐藏状态，坐标在**当前模型私有**基里。体积大，通常也无法交给另一个模型。SOGR 把缓存拆成 **S**（可移植结构）、**C**（共享编码器空间里的规范语义）、**H**（绑定当前模型的激活）。跨模型复用是 **lift** \(H=\mathrm{Lift}_\phi(C,S)\)，不是翻译原始 KV 张量。

### 是不是受 React、Vue 启发？

是。作者是前端工程师。并不是说自然语言等于 DOM。可迁移的是运行时契约：虚拟 DOM / 响应式依赖图、脏子树更新、跳过稳定区域、持久化对象而不是像素。

### 有没有 GPU 规模证明？

没有。本稿是提案加上笔记本 CPU 基线。DistilGPT-2 上，156 个 token 收成 54 个节点、115 条边；该玩具上序列化 \(S\cup C\) 大约比 KV 快照小 184 倍。两个 0.5B 模型：全文 6/6，只给序列化 \(S\) 则 3/6。这是地板；把图当文本喂给模型，对可移植性的支持很弱。

### 有没有声称 Transformer 变成 \(O(L)\)？

没有。摊销代价里每隔 \(K\) 步仍有全局 \(L^2\) 项。只有图足够稀疏、修复足够少、脏集足够小，才可能划算。

---

## 1. 问题

Transformer 用 token 状态表示上下文，用注意力推断交互。它几乎不预设哪些关系重要，所以表达力强。代价是**关系发现一直和信息传播缠在一起**。即便真实依赖稀疏，稠密注意力也不会把稀疏性当成一等计算对象。

线性注意力、状态空间模型、稀疏/分块注意力、混合全局层、分页/前缀 KV 都能让注意力更便宜。它们仍然把**token 序列**或**模型私有张量**当持久状态。它们本身不会把语言变成一套可按版本失效的**依赖图**，也就做不到 UI 框架那样的组件树失效。

SOGR 要问的是：

1. 语言能否编译成稳定的 SOG，让后续层主要**执行**这张图？
2. 改一个实体或事实时，能否只重算脏子图？
3. 若图足够小，用户能否本地持有可移植部分，服务器只持有权重？
4. 新模型能否 lift \((S,C)\)，而不是继承一份 KV 张量？

---

## 2. 来源：React、Vue 与运行时契约

类比不是「自然语言 = DOM」。可迁移的是响应式 UI 的**运行时契约**。

| 前端运行时（React / Vue） | SOGR 运行时 |
|---|---|
| 源码 / JSX / 模板 | token 序列 / 文档 |
| 编译为虚拟 DOM 或响应式图 | 全局构图器（GGC）生成 SOG |
| 组件树 + 带类型的 props / 边 | 节点 \(V\) + 带类型关系 \(E\) 与权重 \(A\) |
| `setState` / 响应式写入弄脏依赖 | 版本号 + \(\mathrm{Desc}_G(\Delta)\) 失效 |
| 跳过未变化子树 | 分层块路由跳过稳定块 |
| 只重绘脏区域 | 图约束线性执行器在脏集上计算 |
| 持久化组件状态，不是像素 | 持久化结构 \(S\) 与规范向量 \(C\)，不是原始 KV |
| 渲染器可换，对象还在 | 模型 \(\phi\) 可换；\(H = \mathrm{Lift}_\phi(C,S)\) |

编译器 IR / AST 是同一分裂：昂贵分析少做，执行复用已编译结构。源码和 IR 可以跟用户走。Token 级 KV 做不到：又大，又绑死一套权重。

---

## 3. 语义对象图（中间表示）

设 \(X = \{x_1,\ldots,x_L\}\) 为 token 嵌入。SOGR 引入

\[
G = (V, E, A)
\]

- \(V\)：token、短语、实体、事件或潜在语义对象
- \(E\)：带类型关系（共指、因果、处所、论元、父子……）
- \(A\)：置信或路由权重

图可以分层。最小节点为

\[
N_i = (h_i,\, c_i,\, \mathrm{type}_i,\, \mathrm{span}_i,\, \mathrm{parent}_i,\, \mathrm{version}_i)
\]

\(h_i\) 绑定当前模型；\(c_i\) 是可选规范向量；type/span/parent 是与模型无关的骨架。边为 \(E_{ij} = (r_{ij}, w_{ij}, \mathrm{version}_{ij})\)。后续全局步可以改图。

稀疏注意力掩码通常只是单层路由。SOGR 里的图是**持久计算状态**。跨会话、跨模型持久化是更强、单独陈述的假设（第 5 节）。

**例子。** *Alice bought a laptop in Hangzhou because her old computer failed.* 纯 token 模型必须反复推断 *her* 是 Alice、*old computer* 不是 *laptop*。SOG 可以给出对象（`person:Alice`、`event:purchase`、`object:laptop`、`location:Hangzhou`、`event:failure`）和带类型边（`agent`、`theme`、`loc`、`cause`）。图可以带权、可以错。工程目标是**失效局部性**和**摊销复用**，不是完美句法分析。

---

## 4. 核心运行时（四个算子 + 缓存）

这是循环，不是一次性解析。

```
tokens X
   │
   ▼
GGC  全局构图器          贵、低频
   │
   ▼
G = (V,E,A)
   │
   ▼
GLE  图约束线性执行器    便宜、高频 × K
                     只沿 G 传播
   │
   ▼
HBR  分层块路由器        跳过稳定连续块
   │
   ▼
缓存 M = (S, C, H)      版本号；脏集 I
   │  每 K 步，或 |I| 过大
   ▼
PGR  周期图修复          增 / 删 / 拆 / 合 / 确认
```

**GGC** 用高容量（常为全注意力）模块提出节点和带权边。必须低频，否则假设失败。

**GLE** 执行图，例如

\[
h'_i = U h_i + \sum_{j \in N(i)} w_{ij}\, V_{r_{ij}} h_j
\]

实现可以是 gated linear attention、delta-rule、SSM。SOGR 不绑定某一种公式。高频工作是**执行** \(G\)，不是重新发现 \(G\)。

**PGR** 在 \(K\) 步后增删拆合或确认边，用来补纯线性模型的有限状态 / 检索短板。

**HBR** 把 token 分成块，块摘要为 \(z_b\)。当稳定性 \(s_b < \tau\) **且**依赖版本未变时跳过。用块是因为 GPU 更吃连续计算，而不是任意 token 稀疏。

**脏集**按结构算：

\[
I_{t+1} = \mathrm{Desc}_G(\Delta_t) \cup \Delta_t
\]

若 \(|I|\) 过大，强制走 PGR。

---

## 5. 三层缓存 \(M = (S, C, H)\)

这是**部署假设**，不是四阶段运行时的必要条件。

| 层 | 含义 | 存放位置 | 可移植？ |
|---|---|---|---|
| **S** 结构 | id、类型、span、父节点、带类型边、版本 | 客户端 / 磁盘 | 是（类似 AST 骨架） |
| **C** 规范语义 | 共享编码器里的紧凑 \(c_i\)，不属于某一个 LLM | 与 S 一起在客户端 | 是 |
| **H** 模型状态 | 隐藏态 \(h_i\)、当前模型路由、块摘要 | 当前服务模型 | 否 |

Token KV 搬不走：布局由层数、头、位置和该模型权重决定。体积小也不够：压缩后的 \(H\) 仍绑定模型。

跨模型复用是 lift，不是 KV 翻译：

\[
H = \mathrm{Lift}_\phi(C, S)
\]

Lift 不够时回退到全文 prefill。

一次请求写成 \((q, S, C, \Delta)\)：lift、扩展脏集、在 \(I\) 上跑 GLE/PGR、返回 token 和补丁 \(\Delta S,\Delta C\)。客户端合并。新模型 \(\phi'\) 只继承 \(S\) 和 \(C\)：\(H'=\mathrm{Lift}_{\phi'}(C,T(S))\)。

不声称：原始 KV 可在模型间线性对齐；关系模式总是对齐；客户端边默认可信。

---

## 6. 有条件的复杂度

\[
C_{\mathrm{avg}} \approx \frac{c_g L^2}{K} + c_e E + \frac{c_m I}{K}
\]

只有质量保住，且 \(\rho=E/L^2 \ll 1\)、\(K\) 够大、平均 \(I\) 够小，才可能优于稠密注意力。拆分执行还要算 \(S\cup C\) 的通信。Lift 必须在同等质量下打败 prefill。这些是工作区间，不是恒等式。

---

## 7. CPU 笔记本基线

硬件：非游戏本，**CPU，无 CUDA**。

### token 序列 → 对象图（`experiments/llm_to_object`）

DistilGPT-2（约 82M）把 156 token 段落收成 54 个节点、115 条边。

| 量 | 值 |
|---|---|
| token 数 \(L\) | 156 |
| 节点 / 边 | 54 / 115 |
| 序列化 \(S \cup C\) | 约 31 KB |
| KV 快照 | 约 5.6 MB |
| 该玩具上的体积比 | 约小 184 倍 |
| 编辑 `Shanghai → Beijing`，1-hop 脏节点 | 3.7% |
| 同一编辑，传递闭包脏节点 | 40.7% |
| 全文 prefill 脏量 | 100% token |

在这个地板上，对象化和体积成立。一跳局部性成立。传递局部性只是部分成立：注意力图比语言学依存更密。抽取有误报（如 “Meanwhile”、“When Bob”）。

### 同一份 \(S\) 当可读上下文（`experiments/compiler_lite`）

| 模型 | 全文 | 仅图 \(S\) |
|---|---|---|
| Qwen2-0.5B-Instruct | 6/6 | 3/6 |
| Qwen2.5-0.5B-Instruct | 6/6 | 3/6 |

这是 compiler-lite（把 S 当文本），不是 KV 编译器。对可移植性的支持很弱；当前抽取/序列化会丢掉答题关键结构。

```bash
pip install -r experiments/llm_to_object/requirements.txt
python experiments/llm_to_object/run.py

pip install -r experiments/compiler_lite/requirements.txt
python experiments/compiler_lite/run.py
```

---

## 8. 怎样证伪

若图很密、修复太勤、失效几乎全局，或稀疏核打不过稠密 GEMM，则模型内效率假设失败。

若 lift 不读回大部分 token 就接近不了 prefill 质量，或序列化体积并不明显小于 KV，则可移植缓存假设失败。

自然语言不是确定性 AST。代词、语篇、否定、反讽、世界知识都会打局部图。安全系统需要带不确定度的边，以及强制全局修复。

---

## 9. 文稿与引用

| 路径 | 作用 |
|---|---|
| [`README.md`](README.md) / [`README.zh.md`](README.zh.md) | 本页（英文 / 简体中文） |
| [`paper/SOGR.md`](paper/SOGR.md) / [`paper/SOGR.zh.md`](paper/SOGR.zh.md) | 完整草案 |
| [`paper/SOGR.pdf`](paper/SOGR.pdf) | 排版 PDF |
| [`paper/build_pdf.py`](paper/build_pdf.py) | `python paper/build_pdf.py` |

```bibtex
@misc{tu2026sogr,
  title        = {SOGR: Semantic Object Graph Runtime — A Structured Incremental Architecture for Long-Context Language Modeling},
  author       = {Tu, Zeping},
  year         = {2026},
  note         = {Independent Research Draft, Version 0.3},
  howpublished = {\url{https://github.com/heiyantutu/sogr}}
}
```

本稿主张的是架构假设、可证伪实验方案，以及笔记本地板。并不主张线性注意力、稀疏注意力、图、缓存或表示对齐各自都是新发明，也未在系统现有技术检索之前主张优先权。
