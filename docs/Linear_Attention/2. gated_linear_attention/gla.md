---
date: 2026-04-07
categories:
  - Linear Attention
---

# Reading Notes: Gated Linear Attention

> **Paper**: [https://arxiv.org/pdf/2312.06635](https://arxiv.org/pdf/2312.06635)  
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)  
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect, so they should be cross-checked with the official implementation later.

## 1. Motivation

The main goal of this work is to design a hardware-efficient training algorithm for linear attention.

## 2. Notation

1. Bold uppercase letters such as `S` and `Q` denote matrices.
2. Symbols like `q_t` and `k_t` denote column vectors with shape `[d, 1]`, while matrices are written in shape `[L, d]`. Because of this convention, some transpose operations will appear.
3. Symbols like `W_t` denote learnable parameters.
4. `q_t` refers to the `t`-th row of `Q`.
5. $\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad\text{for}\quad \square \in \{ \mathbf{Q, K, V,...} \}$

> **Note**: My notation is slightly different from the original paper. In the paper, vectors are written as row vectors. So all formulas in these notes are rewritten accordingly. There may be mistakes.

## 3. Background

### 3.1 Self-Attention

$$\begin{aligned}
\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t &= W_Q \boldsymbol{x}_t , W_K \boldsymbol{x}_t , W_V \boldsymbol{x}_t  
\\
\\
\boldsymbol{o}_t = \frac{\sum_{i=1}^t \boldsymbol{v}_i \exp(\boldsymbol{k}_i^\top \boldsymbol{q}_t )}{\sum_{i=1}^t \exp(\boldsymbol{k}_i^\top \boldsymbol{q}_t )}
&\Leftrightarrow
\mathbf{O} = \text{softmax}\left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}\right) \mathbf{V}
\end{aligned}$$

### 3.2 Linear Attention

$$\begin{aligned}
\mathbf{o}_t &= \frac{\sum_{i=1}^t  \mathbf{v}_i \phi(\mathbf{k}_i)^\top \phi(\mathbf{q}_t)}
{\sum_{i=1}^t \phi(\mathbf{k}_i)^\top \phi(\mathbf{q}_t)} 
\\
\\
\mathbf{S}_t &= \sum_{i=1}^t \boldsymbol{v}_i \phi(\boldsymbol{k}_i)^\top  \in \mathbb{R}^{d \times d}
, \quad 
\boldsymbol{z}_t = \sum_{i=1}^t \phi(\boldsymbol{k}_i) \in \mathbb{R}^{d \times 1}
\\
\\
\mathbf{S}_t &= \mathbf{S}_{t-1} + \boldsymbol{v}_t \phi(\boldsymbol{k}_t)^\top 
, \quad 
\boldsymbol{z}_t = \boldsymbol{z}_{t-1} + \phi(\boldsymbol{k}_t)
, \quad 
\boldsymbol{o}_t = \frac{\mathbf{S}_t \phi(\boldsymbol{q}_t)}{\boldsymbol{z}_t^\top\phi(\boldsymbol{q}_t)}.
\end{aligned}$$

A useful perspective is that linear attention can be implemented recurrently by maintaining a running state. Some previous work observed that even if we remove both the kernel feature map and the normalization term, performance can still remain surprisingly strong. In that simplified case:

$$\begin{aligned}
\mathbf{S}_t = \mathbf{S}_{t-1} +  \boldsymbol{v}_t \boldsymbol{k}_t^\top
, \quad 
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
\end{aligned}$$

### 3.3 Chunkwise Linear Attention

A common trick is to split the sequence `X` into non-overlapping chunks of length `C`.

$$\begin{aligned}
\square_{[t]}^i = \square_{tC+i}
,\quad
\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d}
\quad \text{for } 
\square \in \{ \mathbf{Q, K, V, O} \}
\end{aligned}$$

Under this chunked view, the recurrent state can be updated chunk by chunk, and the output within each chunk can be decomposed into an inter-chunk part and an intra-chunk part.

$$\begin{aligned}
\mathbf{S}_{[t]}^{C} &= \mathbf{S}_{[t-1]}^{C} + \sum_{i=tC+1}^{tC+C} \boldsymbol{v}_i \boldsymbol{k}_i^\top  
\quad \in \mathbb{R}^{d \times d}
\\
\\
\mathbf{O}_{[t]} &= \mathbf{Q}_{[t]} \mathbf{S}_{[t]}  + \left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}\right) \mathbf{V}_{[t]}  
\end{aligned}$$

## 4. Flash Linear Attention

### 4.1 Design Principles

The design of Flash Linear Attention (FLA) follows a few practical principles:

1. Make full use of GPU SMs.
2. Support the batch-size = 1 regime, which requires parallelism along the time dimension.
3. Use Tensor Cores whenever possible.
4. Carefully optimize across the memory hierarchy, especially SRAM and HBM.
5. Parallelize within chunks while keeping chunk-to-chunk recurrence serial.

### 4.2 Algorithm

FLA implements two chunkwise algorithms:

![](assets/Pasted%20image%2020260324141311.png)

![](assets/Pasted%20image%2020260324141321.png)

> **Comment**: A fully serial scheme—parallel only inside each chunk, and serial across chunks—also seems fairly reasonable at first glance.

## 5. Gated Linear Attention

### 5.1 Recurrent Form

The general recurrent form can be written as:

$$\begin{aligned}
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
,\quad
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
\end{aligned}$$

For GLA, the gate is parameterized in a low-rank way:

$$\begin{aligned}
\mathbf{S}_t = ( \mathbf{1} \boldsymbol{\alpha}_t^\top) \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
=
\mathbf{S}_{t-1}\text{Diag}(\boldsymbol{\alpha}_t)  + \boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

![](<assets/Pasted%20image%2020260324141634.png>)

A couple of key points from the paper:

1. The core design challenge in GLA is how to balance **parameter efficiency**, **state size**, and **training efficiency** when parameterizing the gate.
2. In Mamba, the gate is formed from a learned matrix `A` together with data-dependent `alpha_t`, which results in a full-rank gate. The downside is that this form cannot be cleanly expressed as matrix multiplication, so it cannot directly exploit Tensor Cores. Mamba addresses this with a prefix-sum-style algorithm that makes good use of SRAM, but SRAM capacity becomes a bottleneck when scaling to larger hidden states. This is one reason why it can struggle on recall-intensive tasks.

### 5.2 Chunkwise Recurrent Form

To derive a chunkwise implementation for GLA, the paper introduces several auxiliary variables:

$$\begin{aligned}
\boldsymbol{\gamma}_{[t]}^r &= \prod_{i=tC+1}^{tC+r} \boldsymbol{\alpha}_i \in \mathbb{R}^{d \times 1}
, \quad
\\
\\
\mathbf{H}_{[t]}^{r} 
&= \sum_{i=1}^{r} 
(\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top})  
\text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) 
\in \mathbb{R}^{d \times d}
\\
\\
\mathbf{\Gamma}_{[t]} &= [ \boldsymbol{\gamma}_{[t]}^{1}, \boldsymbol{\gamma}_{[t]}^{2}, \dots, \boldsymbol{\gamma}_{[t]}^{C} ]^\top \in \mathbb{R}^{C \times d}
\\
\\
\overleftarrow{\boldsymbol{q}_{[t]}^{i}} &= \boldsymbol{q}_{[t]}^{i} \odot \boldsymbol{\gamma}_{[t]}^{i}
, \quad
\overrightarrow{\boldsymbol{k}_{[t]}^{i}} = \frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}}
\\
\\
\overleftarrow{\mathbf{Q}_{[t]}} &= \mathbf{Q}_{[t]} \odot \mathbf{\Gamma}_{[t]}
, \quad
\overrightarrow{\mathbf{K}_{[t]}} = \mathbf{Q}_{[t]} \oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$


Using these definitions, the chunkwise recurrent state can be expressed as:

$$\begin{aligned}
\mathbf{H}_{[0]}^{r} &= \mathbf{S}_{r}
\\
\\
\mathbf{S}_{[t]}^{r} &= \mathbf{S}_{[t-1]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) 
+ \mathbf{H}_{[t]}^{r}
\\
\\
\mathbf{H}_{[t]}^{r} 
&= \sum_{i=1}^{r} (\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}) 
\text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) 
=  \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i}\left(\frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}}\right)^{\top} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r})
\end{aligned}$$


From there, the output can be rewritten into a form that separates the contribution from previous chunks and the contribution from the current chunk.


$$\begin{aligned}
\boldsymbol{o}_{[t]}^{r} &= \mathbf{S}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r} =  \mathbf{S}_{[t-1]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) \boldsymbol{q}_{[t]}^{r} + \mathbf{H}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r}
\\ 
\\
\Rightarrow \boldsymbol{o}_{[t]}^{r} &= \mathbf{S}_{[t-1]}^{C} \overleftarrow{\boldsymbol{q}_{[t]}^{r}} + \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i} \left(\overrightarrow{\boldsymbol{k}_{[t]}^{i}}\right)^{\top} \overleftarrow{\boldsymbol{q}_{[t]}^{r}}
\end{aligned}$$


In matrix form, this becomes:

$$\begin{aligned}
\mathbf{O}_{[t]} &= \overleftarrow{\boldsymbol{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C \top} 
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]}
\end{aligned}$$


The chunk-level terminal state `S_[t]^C` can also be precomputed recursively:


$$\begin{aligned}
\mathbf{S}_{[t]}^{C} = \mathbf{S}_{[t-1]}^{C} 
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
+ \mathbf{H}_{[t]}^{C}
=
\left(\mathbf{S}_{[t-1]}^{C} 
+ \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
\end{aligned}$$

This reformulation is important because it turns the computation into forms that are much more hardware-friendly.

### 5.3 Secondary-Level Chunking

When the chunk size becomes large, the decay term inside `Q_left K_right^T` can become extremely small, which may introduce numerical precision issues.

To address this, GLA further divides each chunk into **sub-chunks** and computes long-range decay in the **log domain**.

![500](<assets/Pasted%20image%2020260324161324.png>)

The paper introduces `P_[t][tau]` for sub-chunk interactions, assuming each sub-chunk has length `T`:

$$\begin{aligned}
\mathbf{P}_{[t][\tau]} = \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top \odot \mathbf{M}_{[t][\tau]}
\end{aligned}$$

This leads to three cases:

- **Case 1 (pink)**: diagonal sub-chunks. These require higher numerical precision, so they are computed element-wise in full precision.
- **Case 2 (orange)**: off-diagonal sub-chunks. These are computed using half-precision matrix operations, independently for each sub-chunk pair.
- **Case 3 (gray)**: needed only in the fully parallel mode, not in the chunkwise mode.

> **Comment**: If Case 3 does need to be computed, its implementation should be the same as the orange case.

For the diagonal case:

$$\begin{aligned}
(\mathbf{P}_{[t][\tau]})_{i, j} 
=
\sum_{d} (\boldsymbol{q}_{[t]}^{i})_{d} ~(\boldsymbol{k}_{[\tau]}^{j})_{d} 
~ \exp(\log \boldsymbol{\gamma}_{[t]}^{i}  - \log \boldsymbol{\gamma}_{[\tau]}^{j}  ) 
，\quad t=\tau, i>j
\end{aligned}$$


For the off-diagonal case:

$$\begin{aligned}
\mathbf{P}_{[t][\tau]} &= \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top
,\quad t \ne \tau
\\ 
\Rightarrow
\mathbf{P}_{[t][\tau]} &= \left(\boldsymbol{Q}_{[t]} \odot \exp(\log \boldsymbol{\gamma}_{[t]}^{1:T}) \right) \left(\boldsymbol{K}_{[\tau]} \odot \exp(-\log \boldsymbol{\gamma}_{[\tau]}^{1:T}) \right)^\top
,\quad t > \tau
\end{aligned}$$


## 6. Backpropagation for GLA

### 6.1 Recurrent Form

Forward pass:

$$\begin{aligned}
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
,\quad
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
\end{aligned}$$

Backward pass:


$$\begin{aligned}
\delta \boldsymbol{o}_t &= \frac{\partial L}{\partial \boldsymbol{o}_t}
\\
\\
\delta \mathbf{G}_t &= \frac{\partial L}{\partial \mathbf{G}_t} = \delta \mathbf{S}_t \odot \mathbf{S}_{t-1}
\\
\\
\delta \mathbf{S}_t &= \mathbf{G}_{t+1} \odot \delta \mathbf{S}_{t+1} + \delta \boldsymbol{o}_t \boldsymbol{q}_t^\top
\\
\\
\delta \boldsymbol{q}_t &= \frac{\partial L}{\partial \boldsymbol{q}_t} = \mathbf{S}_t^\top \delta \boldsymbol{o}_t
\\
\\
\delta \boldsymbol{v}_t &= \frac{\partial L}{\partial \boldsymbol{v}_t} = \delta  \mathbf{S}_t \boldsymbol{k}_t
\\
\\
\delta \boldsymbol{k}_t &= \frac{\partial L}{\partial \boldsymbol{k}_t} = \delta  \mathbf{S}_t^\top \boldsymbol{v}_t
\end{aligned}$$

For GLA specifically, the gradient with respect to the gate parameter has an additional simplified form:

$$\begin{aligned}
\delta \boldsymbol{\alpha}_t = \delta \mathbf{G}_t^\top \mathbf{1}
\end{aligned}$$

### 6.2 Mathematical Preliminaries

Before deriving the chunkwise backward pass, it helps to review a few standard matrix calculus facts.

**Definition**

$$\begin{aligned}
\mathbf{A} \in \mathbb{R}^{m \times n}
, \quad 
\mathbf{B} \in \mathbb{R}^{n \times k}
, \quad 
\mathbf{C} = \mathbf{A} \mathbf{B}
, \quad 
y = f(\mathbf{C})
, \quad 
\delta \mathbf{C} := \frac{\partial y}{\partial \mathbf{C}}
\end{aligned}$$

**Trace identities**

$$\begin{aligned}
\text{Tr}(ABC) &= \text{Tr}(BCA) = \text{Tr}(CAB)
\\
\\
\text{Tr}(A^\top (B \odot C)) &= Tr((A \odot B)^\top C)=Tr((A \odot C)^\top B)
\end{aligned}$$

**Differential rules**

$$\begin{aligned}
d(\mathbf{A}\mathbf{B}) &= (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B})
\\
\\
d(\mathbf{A} \odot \mathbf{B}) &= (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B})
\end{aligned}$$

**Gradient of matrix multiplication**

$$\begin{aligned}
dy 
= \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{C}})^\top  d \mathbf{C}\right) 
&= \text{Tr}\left( (\delta \mathbf{C})^\top  (d \mathbf{C})\right) 
= \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B}) \right) 
\\
\\
\text{while}\quad dy 
= \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{B}})^\top  (d \mathbf{B})\right)
&\quad \text{so we have}\quad
\delta \mathbf{B} =  \mathbf{A}^\top \delta \mathbf{C}
, \quad 
\delta \mathbf{A} = \delta \mathbf{C} \mathbf{B}^\top
\end{aligned}$$

**Gradient of Hadamard product**

$$\begin{aligned}
dy 
= \text{Tr}\left( (\delta \mathbf{D})^\top  (d \mathbf{D})\right) 
&= \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B}) \right) 
\\
\\
&= \text{Tr}\left( (\delta \mathbf{C} \odot \mathbf{A})^\top (d \mathbf{B}) + (\delta \mathbf{C} \odot \mathbf{B})^\top (d \mathbf{A})  \right) 
\\
\\
\text{while}\quad dy 
= \text{Tr}\left( (\delta \mathbf{B})^\top  (d \mathbf{B})\right)
&\quad \text{so we have}\quad
\delta \mathbf{B} =  \delta \mathbf{C} \odot \mathbf{A} 
, \quad 
\delta \mathbf{A} = \delta \mathbf{C} \odot \mathbf{B}
\end{aligned}$$

### 6.3 Backward Pass for the Chunkwise Recurrent Form

> **Comment**: Personally, I think it is actually easier to derive the backward pass directly from the chunkwise forward equations, rather than starting from the recurrent backward form and then transforming it.

Recall the key forward equations:

$$\begin{aligned}
\mathbf{O}_{[t]} &= \overleftarrow{\boldsymbol{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C \top} 
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]}
\\
\\
\mathbf{S}_{[t]}^{C}
&=
\left(\mathbf{S}_{[t-1]}^{C} 
+ \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
\end{aligned}$$

Now we can derive the gradients term by term.

**Gradient with respect to the chunk state `S_[t]`**

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{O}_{[t]}}
&=
\delta \mathbf{O}_{[t]}^{\top} \overleftarrow{\mathbf{Q}}_{[t]}
,\quad
\left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{S}_{[t]}^C}
=
\delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})  
\\
\\
\Rightarrow
\delta \mathbf{S}_{[t]}^{C} &= \delta \mathbf{S}_{[t+1]}^C \text{Diag}(\boldsymbol{\gamma}_{[t+1]}^{C})   + \delta \mathbf{O}_{[t+1]}^{\top} \overleftarrow{\mathbf{Q}}_{[t+1]}
\end{aligned}$$

**Gradient with respect to `V_[t]`**


$$\begin{aligned}
\left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{O}_{[t]}}
&=
 \left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\\
\\
\left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
&=
\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top}
\\
\\
\Rightarrow
\delta \mathbf{V}_{[t]} 
&=  
\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top}
+ 
\left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\end{aligned}$$

**Gradient with respect to transformed `K_[t]`**

$$\begin{aligned}
\left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}}
&=
\left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}}
\\
\\
\left.\delta\overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{S}_{[t]}^C}
&=
\mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})
\\
\\
\Rightarrow
\delta\overrightarrow{\mathbf{K}_{[t]}} 
&= 
\mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})
+ 
\left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}}
\end{aligned}$$

**Gradient with respect to transformed `Q_[t]`**

$$\begin{aligned}
\delta \overleftarrow{\mathbf{Q}_{[t]}}
=
\left.\delta \overleftarrow{\mathbf{Q}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}}
=
\delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C 
+
\left( \delta\mathbf{O}_{[t]}\mathbf{V}_{[t]}^\top \odot  \mathbf{M} \right) \overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

**Gradient with respect to original `K_[t]`**

$$\begin{aligned}
\delta \mathbf{K}_{[t]}
=
\delta \overrightarrow{\mathbf{K}_{[t]}} \oslash  \mathbf{\Gamma}_{[t]}
\end{aligned}$$

**Gradient with respect to original `Q_[t]`**

$$\begin{aligned}
\delta \mathbf{Q}_{[t]}
=
\delta \overleftarrow{\mathbf{Q}_{[t]}} \odot  \mathbf{\Gamma}_{[t]}
\end{aligned}$$

**Gradient with respect to `Gamma_[t]`**

$$\begin{aligned}
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
&=
\left[0,0,..., \text{diag}\left(\left(\mathbf{S}_{[t-1]}^{C \top} 
+ \overrightarrow{\boldsymbol{K}_{[t]}}^\top \mathbf{V}_{[t]}  \right)  \delta \mathbf{S}_{[t]}^{C}\right) \right]^\top
\\
\\
\delta \mathbf{\Gamma}_{[t]}
&=
\delta \overleftarrow{\mathbf{Q}_{[t]}} \odot \mathbf{Q}_{[t]}
-
\delta \overrightarrow{\mathbf{K}_{[t]}} \odot  \mathbf{K}_{[t]} \oslash (\mathbf{\Gamma}_{[t]} \odot \mathbf{\Gamma}_{[t]})   
+ 
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
\end{aligned}$$

An equivalent form is:


$$\begin{aligned}
\delta \mathbf{\Gamma}_{[t]} \odot \mathbf{\Gamma}_{[t]}
&=
\delta \mathbf{Q}_{[t]} \odot \mathbf{Q}_{[t]}
-
\delta \mathbf{K}_{[t]} \odot  \mathbf{K}_{[t]} 
+ 
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C} \odot \mathbf{\Gamma}_{[t]} 
\\
\\
\left.\delta \mathbf{\Gamma}_{[t]}^C\right|_{\text {from } \mathbf{S}_{[t]}^C} \odot \boldsymbol{\gamma}_{[t]}^C 
&=
\text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C 
+\text{diag}\left(\overrightarrow{\boldsymbol{K}_{[t]}}^\top \mathbf{V}_{[t]}   \delta \mathbf{S}_{[t]}^{C}  \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \right)
\\ &=
\text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C 
+\text{diag}\left(\overrightarrow{\boldsymbol{K}_{[t]}}^\top \left.\delta\overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{S}_{[t]}^C}\right)
\\ &=
\text{diag}\left(\mathbf{S}_{[t-1]}^{C \top} \delta \mathbf{S}_{[t]}^{C}\right) \odot \mathbf{\gamma}_{[t]}^C 
+\text{diag}\left(\boldsymbol{K}_{[t]}^\top \left.\delta \mathbf{K}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}\right)
\end{aligned}$$


**Gradient with respect to `alpha_[t]`**

$$\begin{aligned}
\mathbf{\Gamma}_{[t]} &= [ \prod_{i=tC+1}^{tC+1} \boldsymbol{\alpha}_i, \prod_{i=tC+1}^{tC+2} \boldsymbol{\alpha}_i,...\prod_{i=tC+1}^{tC+C} \boldsymbol{\alpha}_i]^\top \in \mathbb{R}^{C \times d}
\\
\\
\delta \boldsymbol{\alpha}_r 
&= 
\sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot (\prod_{i=tC+1}^{tC+j} \boldsymbol{\alpha}_i \oslash \boldsymbol{\alpha}_r)
=
\left(\sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot \mathbf{\Gamma}_{j,:} \right) \oslash \boldsymbol{\alpha}_r
\\
\\
\delta \mathbf{A}_{[t]} 
&= [\delta \boldsymbol{\alpha}_{[t]}^1, \delta \boldsymbol{\alpha}_{[t]}^2,...\delta \boldsymbol{\alpha}_{[t]}^C]^\top 
=
\text{suffix\_sum}_{row}(\delta \mathbf{\Gamma} \odot \mathbf{\Gamma}) \oslash \mathbf{A}_{[t]} 
\end{aligned}$$

And in the log domain:

$$\begin{aligned}
\log \mathbf{\Gamma}_{[t]} &= [ \sum_{i=tC+1}^{tC+1} \log \boldsymbol{\alpha}_i, \sum_{i=tC+1}^{tC+2} \log \boldsymbol{\alpha}_i,...\sum_{i=tC+1}^{tC+C} \log \boldsymbol{\alpha}_i]^\top \in \mathbb{R}^{C \times d}
\\
\\
\Rightarrow
\delta \log \boldsymbol{\alpha}_r 
&= 
\sum_{j \ge r} \delta \log \mathbf{\Gamma}_{j,:} 
\end{aligned}$$

## 7. Network Architecture

### 7.1 Token Mixing

The token-mixing part of GLA can be written as:


$$\begin{aligned} 
\boldsymbol{\alpha}_{t} = \sigma\left(\left(\mathbf{W}_{\alpha}^{1} \mathbf{W}_{\alpha}^{2} \boldsymbol{x}_{t} + \boldsymbol{b}_{\alpha}\right)\right)^{\frac{1}{\tau}} &\in \mathbb{R}^{d_{k} \times 1}
\\
\\
\mathbf{S}_{t}^{h} = \mathbf{S}_{t-1}^{h} 
\text{Diag}(\boldsymbol{\alpha}_{t}^{h})
+ \boldsymbol{v}_{t}^{h}\boldsymbol{k}_{t}^{h \top}  &\in \mathbb{R}^{d_{v}^{\prime} \times d_{k}^{\prime}}
\\
\\
\boldsymbol{o}_{t}^{h} = \mathbf{S}_{t}^{h} \boldsymbol{q}_{t}^{h}  &\in \mathbb{R}^{d_{v}^{\prime} \times 1} 
\\
\\
\boldsymbol{o}_{t}^{\prime} = \operatorname{concat}\left(\operatorname{LN}\left(\boldsymbol{o}_{t}^{1}\right), \dots, \operatorname{LN}\left(\boldsymbol{o}_{t}^{H}\right)\right) &\in \mathbb{R}^{d_{v} \times 1}
\\
\\
\boldsymbol{r}_{t} = \operatorname{Swish}\left(\mathbf{W}_{r} \boldsymbol{x}_{t} + \boldsymbol{b}_{r}\right) &\in \mathbb{R}^{d_{v} \times 1}
\\
\\
\boldsymbol{y}_{t} = \mathbf{W}_{O} \left(\boldsymbol{r}_{t} \odot \boldsymbol{o}_{t}^{\prime}\right)  &\in \mathbb{R}^{d \times 1}
\end{aligned}$$


The parameter settings are:


$$\begin{aligned} 
\mathbf{W}_{\alpha}^{1} \in \mathbb{R}^{d \times 16}
,\quad 
\mathbf{W}_{\alpha}^{2} \in \mathbb{R}^{16 \times d_{k}}
, \quad
\tau = 16
, \quad
d_{k} = \frac{d}{2}
, \quad
d_{v} = d
\\
\\
(\mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{W}_{V}, \mathbf{W}_{O}, \mathbf{W}_{r})
\in \text{Full Rank}
\end{aligned}$$


### 7.2 Channel Mixing

The channel-mixing block uses SwiGLU:


$$\begin{aligned} 
\operatorname{SwiGLU}(\mathbf{Z}) = \left(\operatorname{Swish}(\mathbf{Z} \mathbf{W}_1) \odot \mathbf{Z} \mathbf{W}_2\right) \mathbf{W}_3
\end{aligned}$$

### 7.3 A Single Layer

A single layer is structured as follows:

$$\begin{aligned} 
\mathbf{Y}^{(l)} &= \operatorname{GLA}\left(\operatorname{LN}\left(\mathbf{X}^{(l)}\right)\right) + \mathbf{X}^{(l)} 
\\
\\
\mathbf{X}^{(l+1)} &= \operatorname{SwiGLU}\left(\operatorname{LN}\left(\mathbf{Y}^{(l)}\right)\right) + \mathbf{X}^{(l)}
\end{aligned}$$

Overall, one GLA layer uses roughly `4d^2` parameters, which is on par with a standard attention layer.

## 8. Experiments

### 8.1 Data and Tokenization

- **Dataset**: SlimPajama
- **Tokenizer**: Mistral tokenizer
- A **100B-token subset** is used

### 8.2 Baselines

The main baselines are:

- **Transformer++**: essentially a LLaMA-style variant with RoPE, SwiGLU, and RMSNorm
- **RetNet**: with its FFN replaced by SwiGLU

### 8.3 Training Setup

- Model sizes: **340M** and **1.3B**
- Optimizer: **AdamW**

For the **340M** model:

- trained on **15B tokens**
- batch size = **0.5M tokens**
- warmup = **0.5B tokens**

For the **1.3B** model:

- trained on **100B tokens**
- batch size = **2M tokens**
- warmup = **1B tokens**

Other settings:

- learning rate = **3e-5**
- weight decay = **0.01**
- gradient clipping = **1.0**

### 8.4 Evaluation

The paper uses `lm-eval` for evaluation.

![](assets/Pasted%20image%2020260324184450.png)

### 8.5 Recall Tasks

Recall-heavy tasks are usually considered one of the harder settings for linear attention, since the model needs to retrieve exact information seen much earlier in the sequence.

![320](assets/Pasted%20image%2020260324184502.png) 

![320](assets/Pasted%20image%2020260324184508.png)

### 8.6 Long-Context Training

The paper studies two training strategies:

- **Mode A**: train directly with sequence length **8K**
- **Mode B**: split the sequence into **12 segments**, each of length **2K**, for an effective training length of **24K**. Gradients are not propagated across segment boundaries.

At evaluation time, perplexity is computed segment by segment:

![](assets/Pasted%20image%2020260324184652.png)

### 8.7 Ablation Study

![320](assets/Pasted%20image%2020260324184702.png) 

![320](assets/Pasted%20image%2020260324184707.png)

