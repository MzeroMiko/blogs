---
date: 2026-04-07
categories:
    - Linear Attention
---

# Reading Notes: Implementation Notes for DeltaNet in FLA

> **Paper**: [https://arxiv.org/abs/2406.06484](https://arxiv.org/abs/2406.06484)   
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These notes provide a step-by-step mathematical reading of the Triton implementation of DeltaNet in the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/) repository, and are intended to be read together with the previous note on the DeltaNet paper.   

## 1. Forward

Entry function: 

```python
def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    return o, A, final_state
```

### Step 1: Compute the $KK^\top$ matrix

$$\begin{aligned}
\mathbf{A} = (\boldsymbol{\beta}_{[t]}^\top \boldsymbol{1} ) \odot \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right)
\end{aligned}$$

**Corresponding code**: `fla.ops.delta_rule.wy_fast -> prepare_wy_repr_fwd -> chunk_scaled_dot_kkt_fwd`

### Step 2: Invert the unit lower-triangular matrix

In the code, this step is implemented using row-wise forward substitution together with block matrix inversion. The `16 x 16` submatrices are handled with forward substitution. Here, `A_i` corresponds to `X^{-1}` in the previous note.

$$\begin{aligned}
\mathbf{A_i} = (\mathbf{I} + \mathbf{A})^{-1}
\end{aligned}$$


**Corresponding code**: `fla.ops.delta_rule.wy_fast -> prepare_wy_repr_fwd -> solve_tril`

> **Comment**: Neumann series plus recursive doubling can be faster, but the numerical precision is worse.

### Step 3: Compute U and W


$$\begin{aligned}
\mathbf{U}_{[t]} 
&= 
\mathbf{A_i} \left((\boldsymbol{\beta}_{[t]}\top \boldsymbol{1}) \odot \mathbf{V}_{[t]} \right)
\\
\\
\mathbf{W}_{[t]} 
&= 
\mathbf{A_i}  \left((\boldsymbol{\beta}_{[t]}\top \boldsymbol{1}) \odot \mathbf{K}_{[t]} \right)
\end{aligned}$$


**Corresponding code**: `fla.ops.delta_rule.wy_fast -> prepare_wy_repr_fwd -> recompute_w_u_fwd`

### Step 4: Recursively compute the hidden state

> **Notation note**: Here, `S_[t]^C` uses the `[d_k, d_v]` layout, so it is transposed relative to the derivation in the previous paper note.


$$\begin{aligned}
\mathbf{V}_{[t],new} &= \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\mathbf{S}_{[t]}^{C} 
&=
\mathbf{S}_{[t-1]}^{C}
+ \mathbf{K}_{[t]}^\top \mathbf{V}_{[t],new}
\end{aligned}$$


**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`

### Step 5: Compute the final output


$$\begin{aligned}
\mathbf{O}_{[t]}
=
\mathbf{Q}_{[t]} \mathbf{S}_{[t-1]}^{C}  
+
\left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M} \right)  \mathbf{V}_{[t],new}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_o -> chunk_fwd_o`

## 2. Backward

Entry function: 


```python
def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    return dq, dk, dv, db, dh0
```


### Step 1: Recompute U and W

`A_i` has already been stored in the forward pass, while `U` and `W` need to be recomputed.


**Corresponding code**: `fla.ops.delta_rule.wy_fast -> prepare_wy_repr_fwd -> recompute_w_u_fwd`

### Step 2: Recompute the hidden state


**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`

### Step 3: Compute the intra-chunk part of $\delta U$


$$\begin{aligned}
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
=
\left( \mathbf{K}_{[t]} \mathbf{Q}_{[t]}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_o -> chunk_bwd_dv_local`

### Step 4: Recursively compute $\delta U$ and $\delta S$


$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\mathbf{K}_{[t]} \delta \mathbf{S}_{[t]}^{C} 
+
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
\end{aligned}$$

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C}
=
\delta \mathbf{S}_{[t+1]}^{C}
+
\mathbf{Q}_{[t+1]}^\top   \delta \mathbf{O}_{[t+1]}
-
\mathbf{W}_{[t+1]} ^\top
\delta \mathbf{U}_{[t+1]} 
\end{aligned}$$


**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_bwd_dhu`

### Step 5: Compute $\delta Q$, $\delta K_{\text{part1}}$, and $\delta W$



$$\begin{aligned}
\delta \mathbf{B}_{[t]}
&=
\delta \mathbf{O}_{[t]} \mathbf{V}_{[t],new}^\top \odot \mathbf{M}
\\
\\
\delta \mathbf{Q}_{[t]}
&=
\delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
+
\delta \mathbf{B}_{[t]} \mathbf{K}_{[t]}
\\
\\
\delta \mathbf{K}_{[t], \text{part1}}
&=
\mathbf{V}_{[t],new}  \delta \mathbf{S}_{[t]}^{C\top}
+
\delta \mathbf{B}_{[t]}^\top\mathbf{Q}_{[t]}
\\
\\
\delta \mathbf{W}_{[t]}
&=
- \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\end{aligned}$$


**Corresponding code**: `fla.ops.common.chunk_o -> chunk_bwd_dqkwg`

### Step 6: Compute $\delta V$, $\delta K_{\text{part2}}$, and $\delta \beta$

In this step, the backward pass through the WY representation is used to compute `dV`, the remaining part of `dK`, and `d beta`.


$$\begin{aligned}
\delta \mathbf{V}_{[t]}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})  \mathbf{A_i}_{[t]} \delta \mathbf{U}_{[t]}
\\
\\
\delta \mathbf{A_i}_{[t]}
&=
\delta \mathbf{U}_{[t]} (\mathbf{V}_{[t]} \text{Diag}(\boldsymbol{\beta}_{[t]}))^\top
+
\delta \mathbf{W}_{[t]} (\mathbf{K}_{[t]}\text{Diag}(\boldsymbol{\beta}_{[t]}))^\top 
\\
\\
\delta \mathbf{A_x}_{[t]}
&= \delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1}
=
-\mathbf{A_i}_{[t]}^\top \delta \mathbf{A_i}_{[t]} \mathbf{A_i}_{[t]}^\top \odot \mathbf{M}_{-1}
\\
\\
\delta \mathbf{K}_{[t], \text{part2}}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{A_i}_{[t]}^\top \delta \mathbf{W}_{[t]}
+
\left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right)^\top  (\text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{K}_{[t]})
+
\text{Diag}(\boldsymbol{\beta}_{[t]}) \left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right) \mathbf{K}_{[t]}
\\
\\
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}\left(\mathbf{A_i}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top\right)
+
\text{diag}\left(\mathbf{A_i}_{[t]}^\top \delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top\right)
+
\text{diag}\left((\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1})
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top\right)
\end{aligned}$$



**Corresponding code**: `fla.ops.delta_rule.wy_fast -> prepare_wy_repr_bwd`