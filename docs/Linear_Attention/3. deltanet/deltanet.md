---
date: 2026-04-07
categories:
  - Linear Attention
---

# Reading Notes: DeltaNet

> **Paper**: [https://arxiv.org/abs/2406.06484](https://arxiv.org/abs/2406.06484)   
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect, so they should be cross-checked against the code later.   

## Motivation

1. Most linear attention models still underperform Transformers, especially on tasks that require in-context retrieval.
2. Training with the Delta Rule is not efficient enough.

## Notations

1. Bold uppercase letters such as `S` and `Q` denote matrices.
2. Symbols like `q_t` and `k_t` denote column vectors with shape `[d, 1]`, while matrices are written in shape `[L, d]`. Because of this convention, some transpose operations will appear.
3. Symbols like `W_t` denote learnable parameters.
4. `q_t` refers to the `t`-th row of `Q`.
5. $\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad\text{for}\quad \square \in \{ \mathbf{Q, K, V,...} \}$

## Background

### 1. GLA

$$\begin{aligned}
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}) \mathbf{V} 
\Leftrightarrow 
\boldsymbol{o}_r = \sum_{i=1}^r \boldsymbol{v}_i \boldsymbol{k}_i^\top \boldsymbol{q}_r 
\end{aligned}$$

### 2. DeltaNet

DeltaNet can be viewed as an SGD optimizer.

## Parallelizing DeltaNet

### Forward

> **Comment**: At inference time, the goal is to minimize memory usage as much as possible while relying on matrix operations whenever possible.

> **Comment**: Assuming this has not already been worked out before, the main technical difficulty in the derivation seems to come from the `Householder transform`.

First, let us recall the DeltaNet update:


$$\begin{aligned}
\mathbf{S}_t 
&= 
\mathbf{S}_{t-1} - \boldsymbol{v}_t^{\text{old}} \boldsymbol{k}_t^\top + \boldsymbol{v}_t^{\text{new}} \boldsymbol{k}_t^\top 
\\&= 
\mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \boldsymbol{k}_t) \boldsymbol{k}_t^\top + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top 
\\&= 
\mathbf{S}_{t-1}(\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

Next, following the same idea as in GLA, define the chunkwise notation and the following auxiliary quantities:

$$\begin{aligned}
\square_{[t]} &= \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad\text{for}\quad \square \in \{ \mathbf{Q, K, V,...} \}
\\
\\
\mathbf{P}_{[t]}^{r} &= \prod_{i=t C + 1}^{t C + r}(\mathbf{I} - \beta_{i} \boldsymbol{k}_{i} \boldsymbol{k}_{i}^{\top}) 
\in \mathbb{R}^{d \times d}
,\quad 
\mathbf{H}_{[t]}^{r} = \sum_{i=tC + 1}^{tC + r} \beta_{i} (\boldsymbol{v}_{i} \boldsymbol{k}_{i}^{\top}) 
\left(
\prod_{j=i + 1}^{t C + r}(\mathbf{I} - \beta_{j} \boldsymbol{k}_{j} \boldsymbol{k}_{j}^{\top}) 
\right)
\in \mathbb{R}^{d \times d}
\end{aligned}$$

Then the chunkwise state can be written as:

$$\begin{aligned}
\mathbf{S}_{[t]}^{r} 
= 
\mathbf{S}_{[t-1]}^{C} \mathbf{P}_{[t]}^{r}  + \mathbf{H}_{[t]}^{r}
, \quad
\text{where }
\mathbf{S}_{[-1]}^{C} = \mathbf{0}
\end{aligned}$$

On the other hand, suppose that:

$$\begin{aligned}
\boldsymbol{u}_1 = \beta_1 \boldsymbol{v}_1 , \quad \mathbf{S}_1 = \beta_1 \boldsymbol{v}_1 \boldsymbol{k}_1^\top
\end{aligned}$$

From this, we can show by induction that:

$$\begin{aligned}
\mathbf{S}_t = \mathbf{S}_{t-1}  + \beta_t (\boldsymbol{v}_t - \mathbf{S}_{t-1} \boldsymbol{k}_t )\boldsymbol{k}_t^\top 
= 
\sum_{i=1}^{t-1} \boldsymbol{u}_i \boldsymbol{k}_i^\top 
+ 
\underbrace{\beta_t \left( \boldsymbol{v}_t - \sum_{i=1}^{t-1} \boldsymbol{u}_i (\boldsymbol{k}_i^\top \boldsymbol{k}_t) \right)}_{\text{defined as } \boldsymbol{u}_t} \boldsymbol{k}_t^\top 
= 
\sum_{i=1}^{t} \boldsymbol{u}_i \boldsymbol{k}_i^\top
\end{aligned}$$

Therefore, we obtain:

$$\begin{aligned}
\mathbf{H}_{[t]}^{r} = 
\sum_{i=1}^{r} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}
, \quad 
\boldsymbol{u}_{[t]}^{r} = \beta_{[t]}^{r} \left( \boldsymbol{v}_{[t]}^{r} - \sum_{i=1}^{r-1} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} \boldsymbol{k}_{[t]}^{r} \right)
\end{aligned}$$

Meanwhile, a product of Householder transforms of the form `(I - beta_t k_t k_t^T)` can always be written in a low-rank form using the `WY representation`. So let us assume:

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} = \mathbf{I} - \sum_{i=1}^{r} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}
\end{aligned}$$


Then, again by induction, we get:


$$\begin{aligned}
\mathbf{P}_{[t]}^{r} 
= 
(\mathbf{I} - \sum_{i=1}^{r - 1} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}) (\mathbf{I} - \beta_{[t]}^{r} \boldsymbol{k}_{[t]}^{r} \boldsymbol{k}_{[t]}^{r \top}) 
=
\mathbf{I} - \sum_{i=1}^{r - 1} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} 
- 
\underbrace{\beta_{[t]}^{r} \left(\boldsymbol{k}_{[t]}^{r}
-
\sum_{i=1}^{r - 1} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}  \boldsymbol{k}_{[t]}^{r}  \right) }_{\text{defined as } \boldsymbol{w}_{[t]}^{r}}
\boldsymbol{k}_{[t]}^{r \top}
\end{aligned}$$ 

Thus, we arrive at:

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} = \mathbf{I} - \sum_{i=1}^{r} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}
， \quad
\boldsymbol{w}_{[t]}^{r} = \beta_{[t]}^{r} \left( \boldsymbol{k}_{[t]}^{r} - \sum_{i=1}^{r-1} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} \boldsymbol{k}_{[t]}^{r} \right)
\end{aligned}$$

Substituting this back into the chunkwise recurrence gives:


$$\begin{aligned}
\mathbf{S}_{[t]}^{r} 
&= 
\mathbf{S}_{[t-1]}^{C} \left(\mathbf{I} - \sum_{i=1}^{r} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}\right)  
+ 
\sum_{i=1}^{r} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}
=
\mathbf{S}_{[t-1]}^{C}
+ 
\sum_{i=1}^{r} \left(\boldsymbol{u}_{[t]}^{i} - \mathbf{S}_{[t-1]}^{C} \boldsymbol{w}_{[t]}^{i}  \right)\boldsymbol{k}_{[t]}^{i\top}
\\
\\
\boldsymbol{o}_{[t]}^{r}
&=
\mathbf{S}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r}
=
\mathbf{S}_{[t-1]}^{C}  \boldsymbol{q}_{[t]}^{r}
+ 
\sum_{i=1}^{r} \left(\boldsymbol{u}_{[t]}^{i} - \mathbf{S}_{[t-1]}^{C} \boldsymbol{w}_{[t]}^{i}  \right)\boldsymbol{k}_{[t]}^{i\top}  \boldsymbol{q}_{[t]}^{r}
\end{aligned}$$

As a result, we can rewrite the whole computation in matrix form as:

$$\begin{aligned}
\mathbf{S}_{[t]}^{C} 
&=
\mathbf{S}_{[t-1]}^{C}
+
\left(\mathbf{U}_{[t]}^\top - \mathbf{S}_{[t-1]}^{C} \mathbf{W}_{[t]}^\top  \right)\mathbf{K}_{[t]}
\\
\\
\mathbf{O}_{[t]}
&=
\mathbf{Q}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  
+
\left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M} \right)  \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top} \right)
\end{aligned}$$

Next, we turn to `u_[t]` and `w_[t]`. Here, `M_{-1} = M - I` denotes the strictly lower-triangular mask whose diagonal entries are zero.

Starting from the recurrence for `u_[t]`, we obtain:


$$\begin{aligned}
& \boldsymbol{u}_{[t]}^{r} = \beta_{[t]}^{r} \left( \boldsymbol{v}_{[t]}^{r} - \sum_{i=1}^{r-1} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} \boldsymbol{k}_{[t]}^{r} \right)
\\
\\
\Rightarrow &
\mathbf{U}_{[t]} 
= 
\text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} 
-
\text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \mathbf{U}_{[t]} 
\\
\\
\Rightarrow &
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right) \mathbf{U}_{[t]} 
= \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} 
\\
\\
\Rightarrow &
\mathbf{U}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} 
\end{aligned}$$


By the same argument, we also have:

$$\begin{aligned}
\mathbf{W}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{K}_{[t]} 
\end{aligned}$$

Therefore, in order to compute both `U_[t]` and `W_[t]`, the key object is:

$$\begin{aligned}
\mathbf{T}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
\end{aligned}$$

## Inversion of Unit Lower Triangular Matrices

In this section, we consider how to compute `T_[t]`. More generally, this reduces to inverting a unit lower-triangular matrix of the form `A = I + L` in `R^{N x N}`.

The overall strategy is as follows: first split the matrix into blocks, and then invert each block using the **Neumann series** together with **recursive doubling**.

### Neumann Series

We begin with the Neumann series:

$$\begin{aligned}
(\mathbf{I} + \mathbf{L})^{-1}
= 
\sum_{n=0}^{\infty} (- \mathbf{L})^n
\end{aligned}$$

Since a strictly lower-triangular matrix `L` is nilpotent, that is,

$$\begin{aligned}
\mathbf{L}^{C} = \mathbf{0} \quad \forall~ C \gt N  
\end{aligned}$$

the infinite series actually truncates after finitely many terms. Therefore:

$$\begin{aligned}
(\mathbf{I} + \mathbf{L})^{-1}
= 
\sum_{n=0}^{N - 1} (- \mathbf{L})^n
\end{aligned}$$

### Recursive Doubling

Next, define:

$$\begin{aligned}
\mathbf{S}_{k} = \sum_{n=0}^{2^k-1} (- \mathbf{L})^n
, \quad
\mathbf{G}_{k} = (- \mathbf{L})^{2^k}
\end{aligned}$$


Then the recursion becomes:

$$\begin{aligned}
\mathbf{S}_{k+1} = \mathbf{S}_{k+1}(\mathbf{I} + \mathbf{G}_{k})
, \quad
\mathbf{G}_{k+1} = \mathbf{G}_{k} \mathbf{G}_{k}
\end{aligned}$$

### Block Matrix Inversion 1

For a block lower-triangular matrix, we have the following inversion formula:

$$\begin{aligned}
\begin{pmatrix}
\mathbf{A}_{11} & \mathbf{0} \\
\mathbf{A}_{21} & \mathbf{A}_{22}
\end{pmatrix}
\begin{pmatrix}
\mathbf{A}_{11}^{-1} & \mathbf{0} \\
-\mathbf{A}_{22}^{-1} \mathbf{A}_{21} \mathbf{A}_{11}^{-1} & \mathbf{A}_{22}^{-1}
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{I}
\end{pmatrix}
\end{aligned}$$

### Block Matrix Inversion 2

More generally, suppose `AB = I`, where `A` is block lower-triangular. Then `B` is also block lower-triangular, and moreover:

$$\begin{aligned}
\mathbf{B}_{ij} = -\mathbf{A}_{ii}^{-1} \sum_{k=j}^{i-1} \mathbf{A}_{ik} \mathbf{B}_{kj}
, \quad i \gt j
\end{aligned}$$

## Network Architecture

The architecture follows a few practical design choices:

1. RMSNorm is used for more stable training, similar to what is mentioned in VMamba and Mamba-2.
2. `q` and `k` are computed as  
   $\frac{\text{SiLU}(\mathbf{W}\boldsymbol{x}_t)}{|\text{SiLU}(\mathbf{W}\boldsymbol{x}_t)|_2}$,  
   where SiLU replaces the original `ELU + 1`, and the L2 normalization ensures the eigenvalues stay below 1.  
   > **Comment**: `q` is mostly included along the way here, but `k` needs to stay below 1 because of the `k^T k` term.
3. Short convolution (shift-SSM) is still used, with motivation borrowed from H3 (Hungry Hungry Hippos).  
   > **Comment**: Personally, I find Su Jianlin’s explanation more convincing: https://spaces.ac.cn/archives/11320
4. A hybrid network design is adopted: GDN + SWA + full attention.

![](assets/Pasted%20image%2020260330162334.png)

## Experiments

### Acceleration

![350](assets/Pasted%20image%2020260330161808.png)

### MQAR

This benchmark comes from *Measuring and Improving Recall in Efficient Language Models*. MQAR is used to evaluate recall ability. In this experiment, DeltaNet does not use convolution, while the other training settings remain the same as in the paper.

### MAD

MAD is a collection of synthetic token manipulation tasks used to evaluate linear attention.

![](assets/Pasted%20image%2020260330162412.png)

### Language Modeling

The experimental setup is the same as in GLA.

![](assets/Pasted%20image%2020260330162443.png)
![](assets/Pasted%20image%2020260330162458.png)

### RegBench

![](assets/Pasted%20image%2020260330162527.png)

> **Comment**: The gap between Mamba with and without convolution looks surprisingly large. Could this be because the hidden state is smaller?

## Related Works

![](assets/Pasted%20image%2020260330162603.png)

