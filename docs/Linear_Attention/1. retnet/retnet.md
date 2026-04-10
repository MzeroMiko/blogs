---
date: 2026-04-07
categories:
  - Linear Attention
---

# Reading Notes: RetNet

> **Paper**: [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621)  
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect, so please let me know if you spot any mistakes.

## 1. Motivation

![](assets/Pasted%20image%2020260324103636.png)
![](assets/Pasted%20image%2020260324103719.png)

## 2. The Retention Mechanism

### 2.1 Architectural Motivation

The starting point is a recurrent state-space-like formulation:


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


The paper then diagonalizes the transition matrix `A`:

$$\begin{aligned}
A = \Lambda(\gamma e^{i\theta})\Lambda^{-1}
,\quad
\gamma, \theta \in \mathbb{R}^d
,\quad
A^{n-m} = \Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}
\end{aligned}$$


By absorbing the change-of-basis matrices into `W_Q` and `W_K`, the output can be rewritten into a much cleaner form:

$$\begin{aligned} 
o_n = \sum_{m=1}^n Q_n (\gamma e^{i\theta})^{n-m} K_m^\top v_m 
= 
\sum_{m=1}^n (Q_n(\gamma e^{i\theta})^n)(K_m(\gamma e^{i\theta})^{-m})^\top v_m 
\end{aligned}$$

If we further assume that `gamma` is a scalar, the expression becomes even simpler:


$$\begin{aligned} 
o_n = \sum_{m=1}^n \gamma^{n-m}(Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m
\end{aligned}$$

This form is much easier to parallelize.

> **Comment**: This derivation seems to rely on several hidden assumptions:  
> `A` needs to be diagonalizable; in an RNN-style setting its eigenvalues usually need magnitude smaller than 1 for stability; and the transition must be time-invariant. There may be other assumptions hiding in the background as well.

### 2.2 Recurrent Mode

In recurrent mode, the retention mechanism is written as:

$$\begin{aligned} 
S_n &= \gamma S_{n-1} + K_n^\top V_n 
\\ 
\\
o_n &= Q_n S_n, \quad n = 1, \cdots, |x| 
\end{aligned}$$

This is the form used during autoregressive inference.  
The state is updated step by step, which makes decoding efficient.

### 2.3 Parallel Mode

For training, the same mechanism can be rewritten into a fully parallel form:


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

where `overline_Theta` denotes the complex conjugate of `\Theta`.


### 2.4 Chunkwise Recurrent Mode

RetNet also supports a chunkwise recurrent formulation, which sits between the fully parallel and fully recurrent views.


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


## 3. Model Design

### 3.1 Multi-Head Design

The model uses a multi-head setup, where the number of heads is:

- `h = d_model / d`

with `d` being the per-head hidden dimension.

This is conceptually similar to multi-head attention, except each head uses a different retention scale.

### 3.2 Multi-Scale Retention (MSR)

The core module is called **Multi-Scale Retention (MSR)**.


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


### 3.3 Normalization

The paper discusses two motivations for normalization.

- **Reason 1**: GroupNorm has a useful scale-invariance property, which helps stabilize the model numerically when stacking many layers.
- **Reason 2**: different heads can have very different variances, so normalization helps align them.

The final normalization scheme is:


$$\begin{aligned}
QK^\top &\to QK^\top / \sqrt{d}
\\
\\
D_{nm} &\to \tilde{D}_{nm} = D_{nm}/\sqrt{\sum_{i=1}^n D_{ni}}
\\
\\
R_{nm} &\to \tilde{R}_{nm} = R_{nm}/\max(|\sum_{i=1}^n R_{ni}|, 1)
\end{aligned}$$

> **Comment**: I wonder whether those normalization coefficients should be detached during training.

### 3.4 A Single Layer

A single RetNet block is structured as follows:


$$\begin{aligned}
Y^l &= \text{MSR}(\text{LN}(X^l)) + X^l 
\\
\\
X^{l+1} &= \text{FFN}(\text{LN}(Y^l)) + Y^l 
\\
\\
\text{FFN}(X) &= \text{gelu}(XW_1)W_2 
\end{aligned}$$

### 3.5 Training and Inference Modes

RetNet uses different computational forms in training and inference:

- **Training**: parallel mode and chunkwise recurrent mode
- **Inference**: recurrent mode

This is one of the main selling points of the paper:  
the same mechanism admits multiple equivalent implementations depending on the use case.

### 3.6 Parameter Allocation

![](assets/Pasted%20image%2020260324112030.png)
![](assets/Pasted%20image%2020260324112050.png)
![](assets/Pasted%20image%2020260324112118.png)



## 4. Training and Evaluation

![](assets/Pasted%20image%2020260324112059.png)

### 4.1 Model Training

![](assets/Pasted%20image%2020260324112225.png)


### 4.2 Performance

![](assets/Pasted%20image%2020260324112253.png)
![](assets/Pasted%20image%2020260324112328.png)


### 4.3 Training Efficiency

RetNet is implemented in PyTorch.  
Training uses the **chunkwise recurrent mode** with:

- chunk size = `512`
- hardware = `8 × A100 80G`

For the **6.7B** and **13B** models, the paper also uses Tensor Parallelism.

![](assets/Pasted%20image%2020260324112457.png)

### 4.4 Inference Efficiency

![](assets/Pasted%20image%2020260324112530.png)


### 4.5 Ablation Study

The ablation study uses a **200M-parameter** model with:

- **16 layers**
- **hidden dimension = 1024**

For **H3**, the head dimension is set to `8`.  
For **RWKV**, the TimeMix module is used in place of attention, while keeping the FFN the same as in the other models.

Training setup:

- **10K steps**
- **batch size = 0.5M tokens**
- training data = the same dataset used for RetNet

![](assets/Pasted%20image%2020260324112554.png)
![](assets/Pasted%20image%2020260324112601.png)

