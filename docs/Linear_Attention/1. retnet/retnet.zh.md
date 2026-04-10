---
date: 2026-04-07
categories:
    - Linear Attention
---

# 阅读笔记：RetNet

> **原文链接**：[https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621)   
> **笔记链接**：[https://mzeromiko.github.io/blogs](https://mzeromiko.github.io/blogs)    
> **声明**：本文为个人阅读笔记。所有来自我自己的推导都可能存在错误，如有发现请指正。  



## 一、动机

![](assets/Pasted%20image%2020260324103636.png)
![](assets/Pasted%20image%2020260324103719.png)


## 二、Retention 机制设计

### 2.1 架构动机

输入：

$$\begin{aligned}
X^0 &= [x_1, \cdots, x_{|x|}] \in \mathbb{R}^{|x| \times d_{\text{model}}}
\\
\\
s_n &= As_{n-1} + K_n^\top v_n
, \quad 
A \in \mathbb{R}^{d \times d}, K_n 
\in \mathbb{R}^{1 \times d}
\\
\\
o_n &= Q_n s_n = \sum_{m=1}^n Q_n A^{n-m} K_m^\top v_m
, \quad 
Q_n 
\in \mathbb{R}^{1 \times d}
\\
\\
Q &= X W_Q
, \quad 
K = X W_K
, \quad 
V = X W_V
\end{aligned}$$

对矩阵 $A$ 做对角化：

$$\begin{aligned}
A = \Lambda(\gamma e^{i\theta})\Lambda^{-1}
,\quad
\gamma, \theta \in \mathbb{R}^d
,\quad
A^{n-m} = \Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}
\end{aligned}$$

进一步将 $\Lambda$ 吸收到 $W_Q, W_K$ 中，可以得到：

$$\begin{aligned} 
o_n = \sum_{m=1}^n Q_n (\gamma e^{i\theta})^{n-m} K_m^\top v_m 
= 
\sum_{m=1}^n (Q_n(\gamma e^{i\theta})^n)(K_m(\gamma e^{i\theta})^{-m})^\top v_m 
\end{aligned}$$

进一步，如果 $\gamma$ 是标量，则：

$$\begin{aligned} 
o_n = \sum_{m=1}^n \gamma^{n-m}(Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m
\end{aligned}$$

这样就非常容易并行化了。

> **评论**：矩阵 $A$ 需要可对角化；在 RNN 中还需要特征值模长小于 1；需要时不变。还有什么隐藏假设？

### 2.2 递推模式（Recurrent Mode）

$$\begin{aligned} 
S_n &= \gamma S_{n-1} + K_n^\top V_n 
\\ 
\\
o_n &= Q_n S_n, \quad n = 1, \cdots, |x| 
\end{aligned}$$

### 2.3 并行模式（Parallel Mode）

$$\begin{aligned}
Q &= (XW_Q) \odot \Theta
, \quad 
K = (XW_K) \odot \overline{\Theta}
, \quad 
V = XW_V
\\
\\
\Theta_n &= e^{in\theta}
, \quad 
D_{nm} = \begin{cases} \gamma^{n-m}, & n \ge m \\ 0, & n < m \end{cases}
\\
\\
O &= (QK^\top \odot D)V
\end{aligned}$$

其中 $\overline{\Theta}$ 是 $\Theta$ 的共轭转置。

### 2.4 分块递推模式（Chunkwise Recurrent Mode）

$$\begin{aligned} 
Q_{[i]} &= Q_{Bi:B(i+1)}
, \quad 
K_{[i]} = K_{Bi:B(i+1)}
, \quad 
V_{[i]} = V_{Bi:B(i+1)}
\\
\\
R_i &= K_{[i]}^\top (V_{[i]} \odot \zeta) + \gamma^B R_{i-1}
, \quad 
\zeta_{ij} = \gamma^{B-i-1} 
\\
\\
O_{[i]} &= \underbrace{(Q_{[i]}K_{[i]}^\top \odot D)V_{[i]}}_{\text{Inner-Chunk}} + \underbrace{(Q_{[i]}R_{i-1}) \odot \xi}_{\text{Cross-Chunk}}
, \quad
\xi_{ij} = \gamma^{i+1} 
\end{aligned}$$

![](assets/Pasted%20image%2020260324105422.png)


## 三、模型设计

**1. 多头设计**

头数量 $h = d_{\text{model}} / d$，其中 $d$ 是单个头的维度。

**2. Multi-Scale Retention（MSR）模块结构**

$$\begin{aligned} 
\gamma &= 1 - 2^{-5-\text{arange}(0, h)} \in \mathbb{R}^h 
\\
\\
\text{head}_i &= \text{Retention}(X, \gamma_i) 
\\
\\
Y &= \text{GroupNorm}_h(\text{Concat}(\text{head}_1, \cdots, \text{head}_h)) 
\\
\\
\text{MSR}(X) &= (\text{swish}(XW_G) \odot Y)W_O 
\end{aligned}$$

**3. 归一化（Normalization）**

- **原因一**：利用 GroupNorm 的缩放不变性 $\text{GroupNorm}(\alpha \cdot \text{head}_i) = \text{GroupNorm}(\text{head}_i)$，将并行模式乘上若干系数后，通过数学技巧确保模型在深层堆叠时的数值稳定性。
- **原因二**：不同 head 之间存在不同的方差。

最终归一化公式：

$$\begin{aligned}
QK^\top &\to QK^\top / \sqrt{d}
\\
\\
D_{nm} &\to \tilde{D}_{nm} = D_{nm}/\sqrt{\sum_{i=1}^n D_{ni}}
\\
\\
R_{nm} &\to \tilde{R}_{nm} = R_{nm}/\max(|\sum_{i=1}^n R_{ni}|, 1)
\end{aligned}$$

> **评论**：是不是要 detach 那些归一化系数？

**4. 单层网络结构**

$$\begin{aligned}
Y^l &= \text{MSR}(\text{LN}(X^l)) + X^l 
\\
\\
X^{l+1} &= \text{FFN}(\text{LN}(Y^l)) + Y^l 
\\
\\
\text{FFN}(X) &= \text{gelu}(XW_1)W_2 
\end{aligned}$$

**5. 训练与推理模式**

- 训练：采用并行模式和分块递推模式
- 推理：采用递推模式

**6. 模型参数分配**

![](assets/Pasted%20image%2020260324112030.png)
![](assets/Pasted%20image%2020260324112050.png)
![](assets/Pasted%20image%2020260324112118.png)


## 四、训练与评估

![](assets/Pasted%20image%2020260324112059.png)


**1. 模型训练**

![](assets/Pasted%20image%2020260324112225.png)

**2. 性能**

![](assets/Pasted%20image%2020260324112253.png)
![](assets/Pasted%20image%2020260324112328.png)

  

**3. 训练效率**

RetNet 采用 PyTorch 实现，训练使用 chunkwise recurrent mode，chunk size = 512，硬件为 8×A100 80G。对于 6.7B 和 13B 模型，采用了 Tensor Parallel。

![](assets/Pasted%20image%2020260324112457.png)


**4. 推理效率**

![](assets/Pasted%20image%2020260324112530.png)


**5. 消融实验**

采用 200M 参数的模型，16 层，hidden dim = 1024。H3 的 head dim = 8。对于 RWKV，采用 TimeMix 模块替代 attention，FFN 与其他模型相同。训练 10K steps，batch size = 0.5M tokens。训练数据集为 RetNet 的训练数据集。

![](assets/Pasted%20image%2020260324112554.png)
![](assets/Pasted%20image%2020260324112601.png)
