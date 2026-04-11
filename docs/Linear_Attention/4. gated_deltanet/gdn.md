# Reading Notes: Gated Delta Networks

> **Paper**: [https://arxiv.org/pdf/2412.06464](https://arxiv.org/pdf/2412.06464)    
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect.  

## Motivation

> **Comment**: I think the introduction of this paper is written very well.

The motivation can be summarized as follows:

1. Quadratic computational complexity motivates the move toward linear attention.
2. Although recent linear attention models have improved over earlier work, they still struggle with information management, especially in-context retrieval.
3. This is not surprising, since the capacity of state-space style models is inherently limited.
4. Mamba2 addresses this issue using decay, but it does not forget each key-value pair individually.
5. The delta rule, in contrast, can handle forgetting at the level of each individual key-value pair, but it cannot quickly erase previously stored memory.
6. Since the two approaches are complementary, the authors propose the **gated delta rule**.
7. The remaining challenge is how to implement the gated delta rule efficiently, and the paper puts substantial effort into this part.
8. The final architecture, **Gated DeltaNet**, works well.

![](assets/Pasted%20image%2020260403200954.png)

## Notations

Throughout these notes:

1. Bold uppercase letters such as `S` and `Q` denote matrices.
2. Symbols like `q_t` and `k_t` denote column vectors with shape `[d, 1]`, while matrices are written in shape `[L, d]`. Because of this convention, some transpose operations will appear.
3. Symbols like `W_t` denote learnable parameters.
4. `q_t` refers to the `t`-th row of `Q`.
5. $\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad\text{for}\quad \square \in \{ \mathbf{Q, K, V,...} \}$

## Online Learning Perspective

![](assets/Pasted%20image%2020260403144404.png)

### Mathematical Preliminaries

We first recall a few basic identities.

$$\begin{aligned}
\|\mathbf{A}\|_F^2 &= \sum_{i,j} a_{ij}^2 = \text{Tr}(\mathbf{A}^\top \mathbf{A})
,\quad
\langle\boldsymbol{k}_t, \boldsymbol{v}_t\rangle = \text{Tr}(\boldsymbol{v}_t^\top \boldsymbol{k}_t)
,\quad
\|\boldsymbol{x}\|^2 = \sum_i x_i^2 = \boldsymbol{x}^\top \boldsymbol{x}
\\
\\
d \left(\text{Tr}(\mathbf{A}^\top \mathbf{A})\right) &= \text{Tr}(d(\mathbf{A}^\top \mathbf{A})) = \text{Tr}((d\mathbf{A})^\top \mathbf{A} + \mathbf{A}^\top (d\mathbf{A}))
=
\text{Tr}(2\mathbf{A}^\top (d\mathbf{A}))
\\
\\
d(\boldsymbol{x}^\top \boldsymbol{y}) &= (d\boldsymbol{x})^\top \boldsymbol{y} + \boldsymbol{x}^\top (d\boldsymbol{y})
\Rightarrow
d(\boldsymbol{x}^\top \boldsymbol{x}) = 2 \boldsymbol{x}^\top (d\boldsymbol{x})
\end{aligned}$$

From these, we obtain:

$$\begin{aligned}
d \|\mathbf{A}\|_F^2 &= \text{Tr}( (\frac{\partial \|\mathbf{A}\|_F^2}{\partial \mathbf{A}})^\top d\mathbf{A})
\Rightarrow
\frac{\partial \|\mathbf{A}\|_F^2}{\partial \mathbf{A}}
=
2 \mathbf{A}
\\
\\
d \|\boldsymbol{x}\|^2 &= \text{Tr}( (\frac{\partial \|\boldsymbol{x}\|^2}{\partial \boldsymbol{x}})^\top d\boldsymbol{x})
\Rightarrow
\frac{\partial \|\boldsymbol{x}\|^2}{\partial \boldsymbol{x}}
=
2 \boldsymbol{x}
\\
\\
d \langle\boldsymbol{k}_t, \boldsymbol{v}_t\rangle
&=
\text{Tr}((\frac{\partial \langle\boldsymbol{k}_t, \boldsymbol{v}_t\rangle}{\partial \boldsymbol{k}_t})^\top d \boldsymbol{k}_t)
\Rightarrow
\frac{\partial \langle\boldsymbol{k}_t, \boldsymbol{v}_t\rangle}{\partial \boldsymbol{k}_t} = \boldsymbol{v}_t
\end{aligned}$$

We will also use the **Sherman–Morrison formula**:

$$\begin{aligned}
(I + \boldsymbol{u}\boldsymbol{v}^\top)^{-1} = I - \frac{\boldsymbol{u}\boldsymbol{v}^\top}{1 + \boldsymbol{v}^\top\boldsymbol{u}}
\end{aligned}$$

### Derivations

#### Longhorn

We begin with the Longhorn objective:


$$\begin{aligned}
L &= \|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2 + \beta_t \|\mathbf{S}_t \boldsymbol{k}_t - \boldsymbol{v}_t\|^2
\\
\\
\Rightarrow 
\frac{\partial}{\partial \mathbf{S}_t} L 
&=
2 (\mathbf{S}_t - \mathbf{S}_{t-1}) 
+ 2\beta_t (\mathbf{S}_t \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^\top
= 0
\\
\\
\Rightarrow
\mathbf{S}_t (\mathbf{I} + \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top)  &= \mathbf{S}_{t-1} + \beta_t\boldsymbol{v}_t \boldsymbol{k}_t^\top
\\
\\
\Rightarrow
\mathbf{S}_t   
&= \mathbf{S}_{t-1} (\mathbf{I} - \frac{ \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top}{1 + \beta_t \boldsymbol{k}_t^\top \boldsymbol{k}_t} ) + \beta_t\boldsymbol{v}_t \boldsymbol{k}_t^\top (\mathbf{I} - \frac{ \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top}{1 + \beta_t \boldsymbol{k}_t^\top \boldsymbol{k}_t} )
\\&= 
\mathbf{S}_{t-1} (\mathbf{I} - \epsilon_t  \boldsymbol{k}_t \boldsymbol{k}_t^\top) 
+ \epsilon_t  \boldsymbol{v}_t \boldsymbol{k}_t^\top
+ \epsilon_t \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top (\boldsymbol{k}_t^\top \boldsymbol{k}_t - \boldsymbol{k}_t \boldsymbol{k}_t^\top)
\\&= 
\mathbf{S}_{t-1} (\mathbf{I} - \epsilon_t  \boldsymbol{k}_t \boldsymbol{k}_t^\top) 
+ \epsilon_t  \boldsymbol{v}_t \boldsymbol{k}_t^\top
\\
\epsilon_t &= \frac{\beta_t}{1 + \beta_t \boldsymbol{k}_t^\top \boldsymbol{k}_t}
\end{aligned}$$

> **Comment**: Here `beta_t` does not seem to be the same as in the original paper.

#### Mamba2

Next, for Mamba2, we have:


$$\begin{aligned}
L &= \|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2 \langle \mathbf{S}_t \boldsymbol{k}_t, \boldsymbol{v}_t\rangle
\\
\\
\Rightarrow 
\frac{\partial}{\partial \mathbf{S}_t} L 
&=
2 (\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1})
-
2 \boldsymbol{v}_t \boldsymbol{k}_t^\top 
\Rightarrow 
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top 
\end{aligned}$$

According to the table in the paper, `LA`, `DeltaNet`, and `Gated DeltaNet` can all be derived in a way similar to Mamba2.

## Gated Delta Rule

### Forward

First, let us recall the Gated DeltaNet update:

$$\begin{aligned}
\mathbf{S}_t 
= 
\mathbf{S}_{t-1}\alpha_t (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

Next, following the derivations for DeltaNet and GLA, define:


$$\begin{aligned}
\mathbf{P}_{[t]}^{r} &= \prod_{i=t C + 1}^{t C + r}(\mathbf{I} - \beta_{i} \boldsymbol{k}_{i} \boldsymbol{k}_{i}^{\top}) 
\in \mathbb{R}^{d \times d}
,\quad 
\gamma_{[t]}^{r} = \prod_{i=t C + 1}^{t C + r} \alpha_i
\\
\\
\mathbf{H}_{[t]}^{r} &= \sum_{i=tC + 1}^{tC + r} \beta_{i} (\boldsymbol{v}_{i} \boldsymbol{k}_{i}^{\top}) 
\frac{\gamma_{[t]}^{r}}{\gamma_{[t]}^{i}}
\left(
\prod_{j=i + 1}^{t C + r}(\mathbf{I} - \beta_{j} \boldsymbol{k}_{j} \boldsymbol{k}_{j}^{\top}) 
\right)
\in \mathbb{R}^{d \times d}
\end{aligned}$$

Then the chunkwise state can be written as:

$$\begin{aligned}
\mathbf{S}_{[t]}^{r} 
= 
\mathbf{S}_{[t-1]}^{C} 
\gamma_{[t]}^{r}
\mathbf{P}_{[t]}^{r}  + \mathbf{H}_{[t]}^{r}
, \quad
\text{where }
\mathbf{S}_{[-1]}^{C} = \mathbf{0}
\end{aligned}$$

On the other hand, suppose that:

$$\begin{aligned}
\mathbf{S}_t = \eta_t \sum_{i=1}^{t} \boldsymbol{u^0}_i \boldsymbol{k}_i^\top
\end{aligned}$$

Then, by induction, we obtain:

$$\begin{aligned}
\mathbf{S}_t &=  \alpha_t \mathbf{S}_{t-1}  + \beta_t (\boldsymbol{v}_t - \alpha_t  \mathbf{S}_{t-1} \boldsymbol{k}_t )\boldsymbol{k}_t^\top 
\\&= 
\alpha_t \eta_{t-1} \sum_{i=1}^{t-1} \boldsymbol{u^0}_i \boldsymbol{k}_i^\top 
+ 
\beta_t \left( \boldsymbol{v}_t - \alpha_t \eta_{t-1}  \sum_{i=1}^{t-1} \boldsymbol{u^0}_i \boldsymbol{k}_i^\top \boldsymbol{k}_t \right) \boldsymbol{k}_t^\top 
= 
\eta_t \sum_{i=1}^{t} \boldsymbol{u^0}_i \boldsymbol{k}_i^\top
\end{aligned}$$

where

$$\begin{aligned}
\eta_t = \alpha_t \eta_{t-1} = \gamma_t
,\quad
\boldsymbol{u^0}_t = \frac{\beta_t}{\eta_t} \boldsymbol{v}_t - \beta_t  \sum_{i=1}^{t-1} \boldsymbol{u^0}_i \boldsymbol{k}_i^\top \boldsymbol{k}_t
\end{aligned}$$

Equivalently, this can be rewritten as:

$$\begin{aligned}
\mathbf{S}_t =  \sum_{i=1}^{t} \frac{\gamma_t}{\gamma_i} \boldsymbol{u}_i \boldsymbol{k}_i^\top
,\quad
\boldsymbol{u}_t = \beta_t \left(\boldsymbol{v}_t - \sum_{i=1}^{t-1} \frac{\gamma_t}{\gamma_i} \boldsymbol{u}_i \boldsymbol{k}_i^\top \boldsymbol{k}_t \right)
\end{aligned}$$

Meanwhile, the product of `Householder transforms` of the form `(I - beta_t k_t k_t^T)` can always be written in a low-rank form using the `WY representation`. From the DeltaNet notes, we have:

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} = \mathbf{I} - \sum_{i=1}^{r} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}
,\quad
\boldsymbol{w}_{[t]}^{r} = \beta_{[t]}^{r} \boldsymbol{k}_{[t]}^{r} - \beta_{[t]}^{r}  \sum_{i=1}^{r-1} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} \boldsymbol{k}_{[t]}^{r}
\end{aligned}$$

Substituting this into the chunkwise recurrence gives:

$$\begin{aligned}
\mathbf{S}_{[t]}^{r} 
&= 
\mathbf{S}_{[t-1]}^{C} \gamma_{[t]}^{r} \left(\mathbf{I} - \sum_{i=1}^{r} \boldsymbol{w}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top}\right)  
+ 
 \sum_{i=1}^{r} \frac{\gamma_{[t]}^{r}}{\gamma_{[t]}^{i}} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}
\\&=
\gamma_{[t]}^{r} \mathbf{S}_{[t-1]}^{C}
+ 
\gamma_{[t]}^{r} \sum_{i=1}^{r} \left(\boldsymbol{u}_{[t]}^{i} - \mathbf{S}_{[t-1]}^{C} \gamma_{[t]}^{i} \boldsymbol{w}_{[t]}^{i}  \right)\frac{\boldsymbol{k}_{[t]}^{i\top}}{\gamma_{[t]}^{i}} 
\\
\\
\boldsymbol{o}_{[t]}^{r}
&=
\mathbf{S}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r}
=
\mathbf{S}_{[t-1]}^{C}  \boldsymbol{q}_{[t]}^{r} \gamma_{[t]}^{r} 
+ 
\sum_{i=1}^{r} \left(\boldsymbol{u}_{[t]}^{i} - \mathbf{S}_{[t-1]}^{C} \gamma_{[t]}^{i} \boldsymbol{w}_{[t]}^{i}  \right)\frac{\boldsymbol{k}_{[t]}^{i\top}}{\gamma_{[t]}^{i}}  \boldsymbol{q}_{[t]}^{r} \gamma_{[t]}^{r} 
\end{aligned}$$

Now define:

$$\begin{aligned}
\overleftarrow{\square_{[t]}} 
&= 
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\square_{[t]}
,\quad
\overrightarrow{\square_{[t]}} 
= 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1} 
\square_{[t]}
\quad\text{for}\quad \square \in \{ \mathbf{Q}, \mathbf{K}, \mathbf{W}\}
\end{aligned}$$

Then we can rewrite the computation in matrix form as:

$$\begin{aligned}
\mathbf{S}_{[t]}^{C} 
&=
\gamma_{[t]}^{C} \mathbf{S}_{[t-1]}^{C}
+
\gamma_{[t]}^{C} \left(\mathbf{U}_{[t]}^\top - \mathbf{S}_{[t-1]}^{C} \overleftarrow{\mathbf{W}_{[t]}}^\top  \right)\overrightarrow{\mathbf{K}_{[t]}}
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t-1]}^{C \top}  
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M} \right)  \left(\mathbf{U}_{[t]} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t-1]}^{C \top} \right)
\end{aligned}$$


> **Comment**: Here `O` seems inconsistent with the expression in the original paper.

Next, we turn to `u_[t]` and `w_[t]`. Here, `M_{-1} = M - I` denotes the strictly lower-triangular mask with zeros on the diagonal.

Starting from the recurrence for `u_[t]`, we get:


$$\begin{aligned}
\boldsymbol{u}_{[t]}^{r} &= \beta_{[t]}^{r} \left( \boldsymbol{v}_{[t]}^{r} - \sum_{i=1}^{r-1} \frac{\gamma_{[t]}^{r}}{\gamma_{[t]}^{i}} \boldsymbol{u}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i \top} \boldsymbol{k}_{[t]}^{r} \right)
\\
\\
\Rightarrow
\mathbf{U}_{[t]} 
&= 
\text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} 
-
\text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) \mathbf{U}_{[t]} 
\\
\\
\Rightarrow
\mathbf{U}_{[t]} 
&= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} 
\end{aligned}$$

By the same argument, we also have:

$$\begin{aligned}
\mathbf{W}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{K}_{[t]} 
\end{aligned}$$

Therefore, in order to compute `U_[t]` and `W_[t]`, the key quantities are:

$$\begin{aligned}
\mathbf{T}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
\end{aligned}$$

and

$$\begin{aligned}
\mathbf{\widetilde{T}}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
\end{aligned}$$

Note that this expression is equivalent to the one in the original paper:

$$\begin{aligned}
\mathbf{\widetilde{T}}_{[t]} 
= 
\left(\mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top  \odot \mathbf{\Gamma}_{[t]}  \right)  \odot \mathbf{M}_{-1}  \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
,\quad
(\mathbf{\Gamma}_{[t]})_{ij} = \frac{\gamma_i}{\gamma_j}
\end{aligned}$$

The matrix inversion step can be handled in the same way as in the DeltaNet notes.

## Network Architecture

![](assets/Pasted%20image%2020260405125026.png)
![](assets/Pasted%20image%2020260405125347.png)
![](assets/Pasted%20image%2020260405131545.png)

## Experiments

### Language Modeling

![](assets/Pasted%20image%2020260405125057.png)
![](assets/Pasted%20image%2020260405131751.png)

### In-Context Retrieval

![](assets/Pasted%20image%2020260405130802.png)

### Length Extrapolation on Long Sequences

![](assets/Pasted%20image%2020260405130852.png)

### Long-Context Understanding

![](assets/Pasted%20image%2020260405131518.png)

