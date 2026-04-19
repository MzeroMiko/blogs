
# Reading Notes: Diagonal Plus Low Rank

> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect, so they should be cross-checked against the code later.   

## Motivation

1. DPLR with Gated Parameterization

## Notations

1. Bold uppercase letters such as `S` and `Q` denote matrices.
2. Symbols like `q_t` and `k_t` denote column vectors with shape `[d, 1]`, while matrices are written in shape `[L, d]`. Because of this convention, some transpose operations will appear.
3. Symbols like `W_t` denote learnable parameters.
4. `q_t` refers to the `t`-th row of `Q`.
5. $\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d} \quad\text{for}\quad \square \in \{ \mathbf{Q, K, V,...} \}$


## Gated DPLR Forward

The original formulation is simple:

$$\begin{aligned}
\mathbf{S}_t 
&= \mathbf{S}_{t-1}
\text{Diag}(\boldsymbol{\alpha}_t)
(\mathbf{I} - \boldsymbol{b}_t \boldsymbol{a}_t^\top) 
+
\boldsymbol{v}_t \boldsymbol{k}_t^\top
\in \mathbb{R}^{d_v \times d_k}
\\
\\
\boldsymbol{o}_t 
&= \mathbf{S}_{t}\boldsymbol{q}_t
\end{aligned}$$


Next, following the derivation for DeltaNet, define the chunkwise notation and the following auxiliary quantities:


$$\begin{aligned}
\mathbf{P}_{[t]}^{r} &= \prod_{i=t C + 1}^{t C + r}
\text{Diag}(\boldsymbol{\alpha}_i)(\mathbf{I} - \boldsymbol{b}_{i} \boldsymbol{a}_{i}^{\top}) 
\in \mathbb{R}^{d_k \times d_k}
\\
\\
\mathbf{H}_{[t]}^{r} &= \sum_{i=tC + 1}^{tC + r} 
(\boldsymbol{v}_{i} \boldsymbol{k}_{i}^{\top}) 
\prod_{j=i + 1}^{t C + r}
\text{Diag}(\boldsymbol{\alpha}_j)(\mathbf{I} - \boldsymbol{b}_{j} \boldsymbol{a}_{j}^{\top}) 
\in \mathbb{R}^{d_v \times d_k}
\end{aligned}$$


Then the chunkwise state can be written as:

$$\begin{aligned}
\mathbf{S}_{[t]}^{r} 
= 
\mathbf{S}_{[t-1]}^{C} 
\mathbf{P}_{[t]}^{r}  + \mathbf{H}_{[t]}^{r}
, \quad
\text{where }
\mathbf{S}_{[-1]}^{C} = \mathbf{0}
\end{aligned}$$


The product of `Householder transforms` of the form `(I - beta_t k_t k_t^T)` can always be written in a low-rank form using the `WY representation`.  So we further derive this, again with the almost the same induction.

When `k = 0`, we have 

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} 
= \prod_{i=t C + 1}^{t C + r} \text{Diag}(\boldsymbol{\alpha}_{[t]}^i)
= \text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\end{aligned}$$

So we assuming that 

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} = \text{Diag}(\boldsymbol{\gamma}_{[t]}^r) - \sum_{i=1}^{r}
\text{Diag}(\boldsymbol{\eta}_{[t]}^r)
\text{Diag}(\boldsymbol{\xi}_{[t]}^i)
\boldsymbol{w}_{[t]}^{i} \boldsymbol{a}_{[t]}^{i \top}
\text{Diag}(\boldsymbol{\epsilon}_{[t]}^i)
\text{Diag}(\boldsymbol{\lambda}_{[t]}^r)
\end{aligned}$$

then we have

$$\begin{aligned}
\mathbf{P}_{[t]}^{r} 
&=
\mathbf{P}_{[t]}^{r-1} 
\text{Diag}(\boldsymbol{\alpha}_{[t]}^r)
(\mathbf{I} - \boldsymbol{b}_{[t]}^r \boldsymbol{a}_{[t]}^{r\top}) 
\\ \\&=
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r) 
-
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r) 
\boldsymbol{b}_{[t]}^r \boldsymbol{a}_{[t]}^{r\top}
\\ \\&-
\sum_{i=1}^{r-1}
\text{Diag}(\boldsymbol{\eta}_{[t]}^{r-1})
\text{Diag}(\boldsymbol{\xi}_{[t]}^i)
\boldsymbol{w}_{[t]}^{i} \boldsymbol{a}_{[t]}^{i \top}
\text{Diag}(\boldsymbol{\epsilon}_{[t]}^i)
\text{Diag}(\boldsymbol{\lambda}_{[t]}^{r-1})
\\ \\&+
\sum_{i=1}^{r-1}
\text{Diag}(\boldsymbol{\eta}_{[t]}^{r-1})
\text{Diag}(\boldsymbol{\xi}_{[t]}^i)
\boldsymbol{w}_{[t]}^{i} \boldsymbol{a}_{[t]}^{i \top}
\text{Diag}(\boldsymbol{\epsilon}_{[t]}^i)
\text{Diag}(\boldsymbol{\lambda}_{[t]}^{r-1})
\text{Diag}(\boldsymbol{\alpha}_{[t]}^r) 
\boldsymbol{b}_{[t]}^r \boldsymbol{a}_{[t]}^{r\top}
\end{aligned}$$

by cancelling the same items, setting parameters as following and absorbing `\xi` to `w`, we have

$$\begin{aligned}
\text{Diag}(\boldsymbol{\lambda}_{[t]}^i)
&=
\text{Diag}(\boldsymbol{\gamma}_{[t]}^i) 
= 
\prod_{j=1}^i 
\text{Diag}(\boldsymbol{\alpha}_{[t]}^j)
,\quad
\text{Diag}(\boldsymbol{\eta}_{[t]}^i) = \mathbf{I}
,\quad
\text{Diag}(\boldsymbol{\epsilon}_{[t]}^i) 
= \text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1}
\\
\\ 
\boldsymbol{w}_{[t]}^{r}
&=
\left(\mathbf{I} -
\sum_{i=1}^{r-1}
\boldsymbol{w}_{[t]}^{i} 
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1}
\boldsymbol{a}_{[t]}^{i}
\right)^\top
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\boldsymbol{b}_{[t]}^r 
\\
\\
\mathbf{P}_{[t]}^{r} &=
\left(\mathbf{I}- \sum_{i=1}^{r}
\boldsymbol{w}_{[t]}^{i} 
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1}
\boldsymbol{a}_{[t]}^{i}
\right)^\top
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\end{aligned}$$


**The `H` part is the key difference, we now assuming that `S` has two accumulation items.**

$$\begin{aligned}
\mathbf{S}_t 
=  
\sum_{i=1}^{t} 
\text{Diag}(\boldsymbol{\eta}_t)
\boldsymbol{u}_i 
\boldsymbol{k}_i^\top
\text{Diag}(\boldsymbol{\epsilon}_i)
\text{Diag}(\boldsymbol{\gamma}_t)
+
\sum_{i=1}^{t} 
\text{Diag}(\boldsymbol{\eta^1}_t)
\boldsymbol{c}_i 
\boldsymbol{a}_i^\top
\text{Diag}(\boldsymbol{\epsilon^1}_i)
\text{Diag}(\boldsymbol{\gamma^1}_t)
\end{aligned}$$


Then, by induction, we obtain:

$$\begin{aligned}
\mathbf{S}_t 
&=  
\mathbf{S}_{t-1} 
\text{Diag}(\boldsymbol{\alpha}_t)
- \mathbf{S}_{t-1}
\text{Diag}(\boldsymbol{\alpha}_t)
\boldsymbol{b}_t 
\boldsymbol{a}_t^\top 
+ \boldsymbol{v}_t \boldsymbol{k}_t^\top
\\&= 
\sum_{i=1}^{t-1} 
\text{Diag}(\boldsymbol{\eta}_{t-1})
\boldsymbol{u}_i 
\boldsymbol{k}_i^\top
\text{Diag}(\boldsymbol{\epsilon}_i)
\text{Diag}(\boldsymbol{\gamma}_{t-1})
\text{Diag}(\boldsymbol{\alpha}_{t})
\\&+
\sum_{i=1}^{t-1} 
\text{Diag}(\boldsymbol{\eta^1}_{t-1})
\boldsymbol{c}_i 
\boldsymbol{a}_i^\top
\text{Diag}(\boldsymbol{\epsilon^1}_i)
\text{Diag}(\boldsymbol{\gamma^1}_{t-1})
\text{Diag}(\boldsymbol{\alpha}_{t})
\\&- 
\sum_{i=1}^{t-1} 
\text{Diag}(\boldsymbol{\eta}_{t-1})
\boldsymbol{u}_i 
\boldsymbol{k}_i^\top
\text{Diag}(\boldsymbol{\epsilon}_i)
\text{Diag}(\boldsymbol{\gamma}_{t-1})
\text{Diag}(\boldsymbol{\alpha}_{t})
\boldsymbol{b}_t
\boldsymbol{a}_t^\top 
\\&- 
\sum_{i=1}^{t-1} 
\text{Diag}(\boldsymbol{\eta^1}_{t-1})
\boldsymbol{c}_i 
\boldsymbol{a}_i^\top
\text{Diag}(\boldsymbol{\epsilon^1}_i)
\text{Diag}(\boldsymbol{\gamma^1}_{t-1})
\text{Diag}(\boldsymbol{\alpha}_{t})
\boldsymbol{b}_t
\boldsymbol{a}_t^\top 
\\&+ \boldsymbol{v}_t \boldsymbol{k}_t^\top 
\\&= 
\sum_{i=1}^{t} 
\text{Diag}(\boldsymbol{\eta}_{t})
\boldsymbol{u}_i 
\boldsymbol{k}_i^\top
\text{Diag}(\boldsymbol{\epsilon}_i)
\text{Diag}(\boldsymbol{\gamma}_{t})
\\&+
\sum_{i=1}^{t} 
\text{Diag}(\boldsymbol{\eta^1}_{t})
\boldsymbol{c}_i 
\boldsymbol{a}_i^\top
\text{Diag}(\boldsymbol{\epsilon^1}_i)
\text{Diag}(\boldsymbol{\gamma^1}_{t})
\end{aligned}$$

after we set

$$\begin{aligned}
\text{Diag}(\boldsymbol{\gamma}_t) 
&= \text{Diag}(\boldsymbol{\gamma^1}_t) 
= \prod_{i=1}^t 
\text{Diag}(\boldsymbol{\alpha}_i)
,\quad
\text{Diag}(\boldsymbol{\epsilon}_t) 
= \text{Diag}(\boldsymbol{\epsilon^1}_t) 
= \text{Diag}(\boldsymbol{\gamma}_t)^{-1}
\\
\\
\text{Diag}(\boldsymbol{\eta}_t) 
&= \text{Diag}(\boldsymbol{\eta^1}_t) 
= \mathbf{I}
\end{aligned}$$

then  we can easily get

$$\begin{aligned}
\boldsymbol{u}_{[t]}^r &= \boldsymbol{v}_{[t]}^r 
\\
\\
\boldsymbol{c}_{[t]}^r
&= 
- \sum_{i=1}^{r-1} \left(
\boldsymbol{v}_i 
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{k}_{[t]}^i\right)^\top
+ \boldsymbol{c}_{[t]}^i
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{a}_{[t]}^i\right)^\top
\right)
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^r) \boldsymbol{b}_{[t]}^r\right)
\end{aligned}$$

so we have `S` and `o`:

$$\begin{aligned}
\mathbf{S}_{[t]}^r 
&=  
\mathbf{S}_{[t-1]}^C
\left(\mathbf{I}- \sum_{i=1}^{r}
\boldsymbol{w}_{[t]}^{i} 
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1}
\boldsymbol{a}_{[t]}^{i}
\right)^\top
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\\&+
\sum_{i=1}^{r} 
\boldsymbol{v}_i 
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{k}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
+
\sum_{i=1}^{r} 
\boldsymbol{c}_i 
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{a}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\\ &=  
\mathbf{S}_{[t-1]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
+
\sum_{i=1}^{r} 
\boldsymbol{v}_i 
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{k}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\\&+
\sum_{i=1}^{r} 
\left( 
\boldsymbol{c}_i 
- \mathbf{S}_{[t-1]}^C \boldsymbol{w}_{[t]}^{i}
\right)
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{a}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\\
\\
\boldsymbol{o}_{[t]}^r
&=
\mathbf{S}_{[t-1]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\boldsymbol{q}_{[t]}^r
+
\sum_{i=1}^{r} 
\boldsymbol{v}_i 
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{k}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\boldsymbol{q}_{[t]}^r
\\&+
\sum_{i=1}^{r} 
\left( 
\boldsymbol{c}_{[t]}^i
- \mathbf{S}_{[t-1]}^C \boldsymbol{w}_{[t]}^{i}
\right)
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{a}_{[t]}^i\right)^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]}^r)
\boldsymbol{q}_{[t]}^r
\end{aligned}$$

define

$$\begin{aligned}
\mathbf{\Gamma}_{[t]} &= [
\boldsymbol{\gamma}_{[t]}^1, 
\boldsymbol{\gamma}_{[t]}^2, 
...,
\boldsymbol{\gamma}_{[t]}^C 
]^\top
\\
\\
\overleftarrow{\square_{[t]}} 
&= 
\square_{[t]}
\odot 
\mathbf{\Gamma}_{[t]}
,\quad
\overrightarrow{\square_{[t]}} 
= 
\square_{[t]}
\oslash
\mathbf{\Gamma}_{[t]}
\quad\text{for}\quad \square \in \{ \mathbf{Q}, \mathbf{K}, \mathbf{A}, \mathbf{B}\}
\end{aligned}$$


then we can derive its co-responding matrix form:

$$\begin{aligned}
\mathbf{S}_{[t]}^C
&=  
\mathbf{S}_{[t-1]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\mathbf{V}_{[t]}^\top
\overrightarrow{\mathbf{K}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\left( 
\mathbf{C}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\right)^\top
\overrightarrow{\mathbf{A}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C\top}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot \mathbf{M} \right) \mathbf{V}_{[t]}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot \mathbf{M} \right) 
\left(
\mathbf{C}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\right) 
\end{aligned}$$

Next, we turn to `c_[t]` and `w_[t]`. Here, `M_{-1} = M - I` denotes the strictly lower-triangular mask with zeros on the diagonal.

Starting from the recurrence for `w_[t]`, we get:

$$\begin{aligned}
\boldsymbol{w}_{[t]}^r 
&= 
\left(
\mathbf{I}  - 
\sum_{i=1}^{r-1} 
\boldsymbol{w}_{[t]}^i  
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^i)^{-1} \boldsymbol{a}_{[t]}^i\right)^\top
\right)
\left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^r) \boldsymbol{b}_{[t]}^r\right)
\\
\\
\Rightarrow
\mathbf{W}_{[t]} 
&= 
\overleftarrow{\mathbf{B}_{[t]}}  - 
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{W}_{[t]} 
\\
\\
\Rightarrow
\mathbf{W}_{[t]} 
&=
\left(
\mathbf{I} +  
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\right)^{-1}
\overleftarrow{\mathbf{B}_{[t]}}
\end{aligned}$$

By the same argument, we also have:

$$\begin{aligned}
\mathbf{C}_{[t]} 
&= 
- \left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
- \left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{C}_{[t]} 
\\
\\
\Rightarrow
\mathbf{C}_{[t]} 
&=
- \left(
\mathbf{I} +  
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\right)^{-1}
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
\end{aligned}$$

Therefore, in order to compute `U_[t]` and `W_[t]`, the key quantities are:

$$\begin{aligned}
\mathbf{E}_{[t]} 
= 
\left(\mathbf{I} +  \left( \overleftarrow{\mathbf{B}_{[t]}} \overrightarrow{\mathbf{A}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1}
\end{aligned}$$

The matrix inversion step can be handled in the same way as in the DeltaNet notes.

## Gated DPLR Backward
### Key results for forward

$$\begin{aligned}
\mathbf{F}_{[t]} 
&= 
\mathbf{I} +  \left( \overleftarrow{\mathbf{B}_{[t]}} \overrightarrow{\mathbf{A}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
,\quad
\mathbf{E}_{[t]} = \mathbf{F}_{[t]}^{-1}
\\
\\
\mathbf{W}_{[t]} &= \mathbf{E}_{[t]} \overleftarrow{\mathbf{B}_{[t]}}  
,\quad
\mathbf{C}_{[t]} = 
- \mathbf{E}_{[t]} 
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
\\
\\
\mathbf{V}_{[t],1} &:= \mathbf{V}_{[t]}
,\quad
\mathbf{V}_{[t],2} := \left(\mathbf{C}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  \right) 
\\
\\
\mathbf{S}_{[t]}^C
&=  
\mathbf{S}_{[t-1]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\mathbf{V}_{[t],1}^\top
\overrightarrow{\mathbf{K}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\mathbf{V}_{[t],2}^\top
\overrightarrow{\mathbf{A}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C\top}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot \mathbf{M} \right) \mathbf{V}_{[t],1}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot \mathbf{M} \right) 
\mathbf{V}_{[t],2}
\\
\\
\mathbf{\Gamma}_{[t]} &= [
\boldsymbol{\gamma}_{[t]}^1, 
\boldsymbol{\gamma}_{[t]}^2, 
...,
\boldsymbol{\gamma}_{[t]}^C 
]^\top
\\
\\
\overleftarrow{\square_{[t]}} 
&= 
\square_{[t]}
\odot 
\mathbf{\Gamma}_{[t]}
,\quad
\overrightarrow{\square_{[t]}} 
= 
\square_{[t]}
\oslash
\mathbf{\Gamma}_{[t]}
\quad\text{for}\quad \square \in \{ \mathbf{Q}, \mathbf{K}\}
\end{aligned}$$

### For $\delta \mathbf{C}_{[t]}$, $\delta \mathbf{V}_{[t],1}$, $\delta \mathbf{W}_{[t]}$

The gradient with respect to $\mathbf{C}_{[t]}$ is:

$$\begin{aligned}
\delta \mathbf{C}_{[t]}
&= \delta \mathbf{V}_{[t],2}
=
\overrightarrow{\mathbf{A}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\delta \mathbf{S}_{[t]}^{C \top}
+
\left( 
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top 
\odot \mathbf{M} 
\right)^\top 
\delta \mathbf{O}_{[t]}
\\
\\
\delta \mathbf{V}_{[t],1}
& =
\overrightarrow{\mathbf{K}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\delta \mathbf{S}_{[t]}^{C \top}
+
\left( 
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top 
\odot \mathbf{M} 
\right)^\top 
\delta \mathbf{O}_{[t]}
\\
\\
\delta \mathbf{W}_{[t]}
&=
- \delta \mathbf{C}_{[t]} \mathbf{S}_{[t]}^{C}
\end{aligned}$$



### For $\delta \mathbf{E}_{[t]}$, $\delta \mathbf{F}_{[t]}$


$$\begin{aligned}
\delta \mathbf{E}_{[t]}
&= 
- \delta \mathbf{C}_{[t]}
\mathbf{V}_{[t]}^\top
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)^\top
+ \delta \mathbf{W}_{[t]} \overleftarrow{\mathbf{B}_{[t]}}^\top
\\
\\
\delta \mathbf{F}_{[t]}
&=
- \mathbf{E}_{[t]}^\top \delta \mathbf{E}_{[t]} \mathbf{E}_{[t]}^\top
\end{aligned}$$




### For $\delta \mathbf{S}_{[t]}^C$

$$\begin{aligned}
\delta \mathbf{S}_{[t-1]}^{C}
&=
-\delta \mathbf{C}_{[t]}^\top
\mathbf{W}_{[t]}
+
\delta \mathbf{S}_{[t]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+ \delta \mathbf{O}_{[t]}^\top  
\overleftarrow{\mathbf{Q}_{[t]}}
\\
\\
\Rightarrow
\delta \mathbf{S}_{[t]}^{C}
&=
\delta \mathbf{S}_{[t+1]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t+1]}^C)
+ \delta \mathbf{O}_{[t+1]}^\top  
\overleftarrow{\mathbf{Q}_{[t+1]}}
-\delta \mathbf{C}_{[t+1]}^\top
\mathbf{W}_{[t+1]}
\end{aligned}$$


### For $\delta \mathbf{Q}_{[t]}$

$$\begin{aligned}
\delta \overleftarrow{\mathbf{Q}_{[t]}}
&=
\delta \mathbf{O}_{[t]} 
\mathbf{S}_{[t-1]}^C
+ 
\left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],1}^\top \odot \mathbf{M}\right) \overrightarrow{\mathbf{K}_{[t]}}
+ 
\left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],2}^\top \odot \mathbf{M}\right) \overrightarrow{\mathbf{A}_{[t]}}
\\
\\
\delta \mathbf{Q}_{[t]}
&=
\delta \overleftarrow{\mathbf{Q}_{[t]}}
\odot \mathbf{\Gamma}_{[t]}
\end{aligned}$$

### For $\delta \mathbf{B}_{[t]}$

$$\begin{aligned}
\delta \overleftarrow{\mathbf{B}_{[t]}}
&=
\mathbf{E}_{[t]}^\top
\delta \mathbf{W}_{[t]} 
-
\left(
\mathbf{E}_{[t]}^\top
\delta \mathbf{C}_{[t]}
\mathbf{V}_{[t]}^\top
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{K}_{[t]}}
+
\left(
\delta \mathbf{F}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{A}_{[t]}}
\\
\\
\delta \mathbf{B}_{[t]}
&=
\delta \overleftarrow{\mathbf{Q}_{[t]}}
\odot \mathbf{\Gamma}_{[t]}
\end{aligned}$$


### For $\delta \mathbf{K}_{[t]}$

$$\begin{aligned}
\delta \overrightarrow{\mathbf{K}_{[t]}}
&=
\left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],1}^\top \odot \mathbf{M}\right)^\top
\overleftarrow{\mathbf{Q}_{[t]}}
-
\left(
\mathbf{E}_{[t]}^\top
\delta \mathbf{C}_{[t]}
\mathbf{V}_{[t]}^\top
\odot \mathbf{M}_{-1}
\right)^\top
\overleftarrow{\mathbf{B}_{[t]}}
\\
\\
\delta \mathbf{K}_{[t]}
&=
\delta \overrightarrow{\mathbf{K}_{[t]}}
\oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$

### For $\delta \mathbf{A}_{[t]}$


$$\begin{aligned}
\delta \overrightarrow{\mathbf{A}_{[t]}}
&=
\left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],2}^\top \odot \mathbf{M}\right)^\top
\overleftarrow{\mathbf{Q}_{[t]}}
+
\left(
\delta \mathbf{F}_{[t]}
\odot \mathbf{M}_{-1}
\right)^\top
\overleftarrow{\mathbf{B}_{[t]}}
\\
\\
\delta \mathbf{A}_{[t]}
&=
\delta \overrightarrow{\mathbf{A}_{[t]}}
\oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$

### For $\delta \mathbf{V}_{[t]}$

$$\begin{aligned}
\delta \mathbf{V}_{[t]}
=
\delta \mathbf{V}_{[t],1}
-
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)^\top
\mathbf{E}_{[t]} ^\top
\delta \mathbf{C}_{[t]}
\end{aligned}$$

### For $\delta \mathbf{\Gamma}_{[t]}$


$$\begin{aligned}
\delta \boldsymbol{\gamma}_{[t]}^C
&=
\text{diag}\left(
\left(
\mathbf{S}_{[t-1]}^C
+
\mathbf{V}_{[t],1}^\top
\overrightarrow{\mathbf{K}_{[t]}}
+
\mathbf{V}_{[t],2}^\top
\overrightarrow{\mathbf{A}_{[t]}}
\right)^\top
\delta \mathbf{S}_{[t]}^C 
\right)
=
\left(\boldsymbol{\gamma}_{[t]}^{C}\right)^{-1}
\odot
\text{diag}\left(
\mathbf{S}_{[t]}^{C\top}
\delta \mathbf{S}_{[t]}^C 
\right)
\end{aligned}$$

$$\begin{aligned}
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text{w/o extra} \boldsymbol{\gamma}_{[t]}^C}
&=
\delta \overleftarrow{\mathbf{Q}_{[t]}} \odot \mathbf{Q}_{[t]}
+
\delta \overleftarrow{\mathbf{B}_{[t]}} \odot \mathbf{B}_{[t]}
\\&-
\delta \overrightarrow{\mathbf{K}_{[t]}} \odot \mathbf{K}_{[t]} \oslash \mathbf{\Gamma}_{[t]} \oslash \mathbf{\Gamma}_{[t]}
-
\delta \overrightarrow{\mathbf{A}_{[t]}} \odot \mathbf{A}_{[t]} \oslash \mathbf{\Gamma}_{[t]} \oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$

specially:


$$\begin{aligned}
\delta \log \mathbf{\Gamma}_{[t]}
&=
\delta \mathbf{\Gamma}_{[t]}
\odot
\mathbf{\Gamma}_{[t]}
\\&=
\delta \mathbf{Q}_{[t]} \odot \mathbf{Q}_{[t]}
+
\delta \mathbf{B}_{[t]} \odot \mathbf{B}_{[t]}
-
\delta \mathbf{K}_{[t]} \odot \mathbf{K}_{[t]}
-
\delta \mathbf{A}_{[t]} \odot \mathbf{A}_{[t]}
\\&+
[0,0,...,\text{diag}\left(
\mathbf{S}_{[t]}^{C\top}
\delta \mathbf{S}_{[t]}^C 
\right)]
\end{aligned}$$




## Discussions
### Reducing to KDA

from the perspective of the `recurrent mode`, its is easy to reduce to `KDA` with

$$\begin{aligned}
\boldsymbol{a}_t &= \boldsymbol{k}_t
,\quad
\boldsymbol{b}_t = \beta_t \boldsymbol{k}_t
,\quad
\boldsymbol{v}_t = \beta_t \boldsymbol{v'}_t
\\
\\
\mathbf{S}_t 
&= \mathbf{S}_{t-1}
\text{Diag}(\boldsymbol{\alpha}_t)
(\mathbf{I} - \boldsymbol{b}_t \boldsymbol{a}_t^\top) 
+
\boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

we choose to decouple `b` rather than `a` , which is equivalent in `recurrent mode`, as it is easier to transfer in `chunk-wise` form. The core adapters are

$$\begin{aligned}
\mathbf{E}_{[t]} 
&= 
\left(\mathbf{I} +  \left( \overleftarrow{\mathbf{B}_{[t]}} \overrightarrow{\mathbf{A}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) \right)^{-1}
=
\left( \mathbf{I} +  
\text{Diag}(\boldsymbol{\beta}_{[t]})
\left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right) 
\right)^{-1}
\\
\\
\mathbf{C}_{[t]} &= 
- \mathbf{E}_{[t]} 
\left(
\overleftarrow{\mathbf{B}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
=
- \mathbf{E}_{[t]} 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\left(
\overleftarrow{\mathbf{K}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
\\
\\
\mathbf{U}_{[t]} &:= 
\mathbf{V}_{[t]} + \mathbf{C}_{[t]} 
=
\left(
\mathbf{I}
- \mathbf{E}_{[t]} 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\left(
\overleftarrow{\mathbf{K}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\right)
\mathbf{V}_{[t]} 
\\&=
\mathbf{E}_{[t]} 
\left(
\mathbf{E}_{[t]}^{-1}
- 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\left(
\overleftarrow{\mathbf{K}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\right)
\mathbf{V}_{[t]} 
=
\mathbf{E}_{[t]} \mathbf{V}_{[t]} 
\\
\\
\mathbf{V}_{[t],new}
&:= \mathbf{V}_{[t],1} + \mathbf{V}_{[t],2}
= \mathbf{V}_{[t]} + \mathbf{C}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}
= \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}
\end{aligned}$$

### DPLR Without Explicit Gating

we can absorb the diagnoal into `b`.

$$\begin{aligned}
\boldsymbol{b}_t &= \text{Diag}(\boldsymbol{\alpha_t})^{-1} \boldsymbol{d}_t
\\
\\
\mathbf{S}_t 
&= \mathbf{S}_{t-1}
\text{Diag}(\boldsymbol{\alpha}_t)
(\mathbf{I} - \boldsymbol{b}_t \boldsymbol{a}_t^\top) 
+
\boldsymbol{v}_t \boldsymbol{k}_t^\top
= \mathbf{S}_{t-1}
(\text{Diag}(\boldsymbol{\alpha}_t) - \boldsymbol{d}_t \boldsymbol{a}_t^\top) 
+
\boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

then we have

$$\begin{aligned}
\mathbf{F}_{[t]} 
&= 
\mathbf{I} +  \left(\mathbf{D}_{[t]} \overrightarrow{\mathbf{A}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
,\quad
\mathbf{E}_{[t]} = \mathbf{F}_{[t]}^{-1}
\\
\\
\mathbf{W}_{[t]} &= \mathbf{E}_{[t]} \overleftarrow{\mathbf{B}_{[t]}}  
,\quad
\mathbf{C}_{[t]} = 
- \mathbf{E}_{[t]} 
\left(
\mathbf{D}_{[t]}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot 
\mathbf{M}_{-1}
\right)
\mathbf{V}_{[t]} 
\\
\\
\mathbf{V}_{[t],1} &:= \mathbf{V}_{[t]}
,\quad
\mathbf{V}_{[t],2} := \left(\mathbf{C}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  \right) 
\\
\\
\mathbf{S}_{[t]}^C
&=  
\mathbf{S}_{[t-1]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\mathbf{V}_{[t],1}^\top
\overrightarrow{\mathbf{K}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\mathbf{V}_{[t],2}^\top
\overrightarrow{\mathbf{A}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C\top}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot \mathbf{M} \right) \mathbf{V}_{[t],1}
+
\left(
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{A}_{[t]}}^\top
\odot \mathbf{M} \right) 
\mathbf{V}_{[t],2}
\\
\\
\mathbf{\Gamma}_{[t]} &= [
\boldsymbol{\gamma}_{[t]}^1, 
\boldsymbol{\gamma}_{[t]}^2, 
...,
\boldsymbol{\gamma}_{[t]}^C 
]^\top
\\
\\
\overleftarrow{\square_{[t]}} 
&= 
\square_{[t]}
\odot 
\mathbf{\Gamma}_{[t]}
,\quad
\overrightarrow{\square_{[t]}} 
= 
\square_{[t]}
\oslash
\mathbf{\Gamma}_{[t]}
\quad\text{for}\quad \square \in \{ \mathbf{Q}, \mathbf{K}\}
\end{aligned}$$










