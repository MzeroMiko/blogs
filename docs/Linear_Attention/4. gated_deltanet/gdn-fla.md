# Reading Notes: Implementation Notes for GDN in FLA

> **Paper**: [https://arxiv.org/pdf/2412.06464](https://arxiv.org/pdf/2412.06464)    
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect.  

## Forward 

Entry function signature:  

```python

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
):
    return g, o, A, final_state, initial_state
```

**Notation**


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


### Step 1: Compute the cumulative gate in the log domain

Here, `gamma` corresponds to what was previously defined as `log gamma`, so this should be kept in mind in the formulas that follow.

$$\begin{aligned}
\boldsymbol{\gamma}_{[t]}^r &= \sum_{i=tC+1}^{tC+r} \log \alpha_i \in \mathbb{R}
\end{aligned}$$

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`

### Step 2: Solve the lower-triangular system

$$\begin{aligned}
\widetilde{\mathbf{A_0}}_{[t]} = \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
\\
\\
\widetilde{\mathbf{A}}_{[t]} = (\mathbf{I} + \widetilde{\mathbf{A_0}}_{[t]})^{-1}
\end{aligned}$$


**Corresponding code**: `fla.ops.gated_delta_rule.chunk_fwd -> chunk_gated_delta_rule_fwd_intra -> chunk_gated_delta_rule_fwd_kkt_solve_kernel`

### Step 3: Compute U and the left-scaled W

$$\begin{aligned}
\mathbf{U}_{[t]} 
&= 
\widetilde{\mathbf{A}}_{[t]} 
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{V}_{[t]}
\\
\\
\overleftarrow{\mathbf{W}_{[t]}} 
&= 
\widetilde{\mathbf{A}}_{[t]} 
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]}) 
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{K}_{[t]}
\end{aligned}$$

**Corresponding code**: `fla.ops.gated_delta_rule.wy_fast -> recompute_w_u_fwd`

### Step 4: Recursively compute the hidden state

Here, `S_[t]^C` uses the `[d_k, d_v]` layout, so it is `transposed relative` to the previous derivation.

$$\begin{aligned}
\mathbf{V}_{[t],new} 
&= 
\mathbf{U}_{[t]} 
- \overleftarrow{\mathbf{W}_{[t]}} 
\mathbf{S}_{[t-1]}^{C}
\\
\\
\mathbf{S}_{[t]}^{C} 
&=
\boldsymbol{\gamma}_{[t]}^C
\mathbf{S}_{[t-1]}^{C}
+ \mathbf{K}_{[t]}^\top 
\left(
\text{Diag}(\exp(\boldsymbol{\gamma}_{[t]}^C -  \boldsymbol{\gamma}_{[t]})
\mathbf{V}_{[t],new}
\right)
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`

### Step 5: Compute the final output

$$\begin{aligned}
\mathbf{O}_{[t]}
=
\text{Diag}\left(\exp \boldsymbol{\gamma}_{[t]} \right)
\mathbf{Q}_{[t]} \mathbf{S}_{[t-1]}^{C}  
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M} \right)  \mathbf{V}_{[t],new}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_o -> chunk_fwd_o`

> **Comment**: From the code, it seems clear that Equation 9 in [2412.06464v3](https://arxiv.org/pdf/2412.06464v3) is indeed a typo.

## Backward

Entry function signature:

```python
def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
):
    return dq, dk, dv, db, dg, dh0
```

### Step 1: Reuse the stored inverse

`A_[t] = (I + A0_[t])^{-1}` has already been stored. Everything else is the same as in the forward pass.

**Corresponding code**: `fla.ops.gated_delta_rule.wy_fast -> recompute_w_u_fwd`

### Step 2: Recompute the hidden state

This is the same as in the forward pass.

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`

### Step 3: Compute the contribution to δU from O

$$\begin{aligned}
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
=
\left(
\overrightarrow{\mathbf{K}_{[t]}} \overleftarrow{\mathbf{Q}_{[t]}}^\top 
\odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_o -> chunk_bwd_dv_local`

### Step 4: Backward recursion for U and the state

$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\text{Diag}\left(\exp(\boldsymbol{\gamma}_{[t]}^C -  \boldsymbol{\gamma}_{[t]})\right)
\mathbf{K}_{[t]} \delta \mathbf{S}_{[t]}^{C} 
+
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
\end{aligned}$$

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C}
=
\exp(\boldsymbol{\gamma}_{[t]}^C)
\delta \mathbf{S}_{[t+1]}^{C}
+
\overleftarrow{\mathbf{Q}_{[t+1]}}^\top   \delta \mathbf{O}_{[t+1]}
-
\overleftarrow{\mathbf{W}_{[t+1]}} ^\top
\delta \mathbf{U}_{[t+1]} 
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_bwd_dhu`

### Step 5: Compute dB, dQ, part of dK, dW, and part of dgamma


$$\begin{aligned}
\delta \mathbf{B}_{[t]}
&=
\delta \mathbf{O}_{[t]} \mathbf{V}_{[t],new}^\top \odot \mathbf{M}
\\
\\
\delta \mathbf{Q}_{[t]}
&=
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
+
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \delta \mathbf{B}_{[t]} 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\right)
\mathbf{K}_{[t]}
\\
\\
\delta \mathbf{K}_{[t], \text{part1}}
&=
\text{Diag}(\exp(\gamma_{[t]}^C  - \boldsymbol{\gamma}_{[t]})) \mathbf{V}_{[t],new} \delta \mathbf{S}_{[t]}^{C\top}
+
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \delta \mathbf{B}_{[t]} 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\right)^\top
\mathbf{Q}_{[t]}
\\&=
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
+
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
\\
\\
\delta \mathbf{W}_{[t]}
&=
- \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\end{aligned}$$

$$\begin{aligned}
\delta \boldsymbol{\gamma}_{[t]}^C
&=
\delta \boldsymbol{\gamma}_{[t]}^C
\exp \boldsymbol{\gamma}_{[t]}^C
=
\text{Tr}\left(
\delta \mathbf{S}_{[t]}^{C\top} \mathbf{S}_{[t-1]}^C
\right) 
\exp \boldsymbol{\gamma}_{[t]}^C
+
\text{Tr}\left(
\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
\right)
\mathbf{K}_{[t]}^\top\right)
\\ \\ &=
\text{Tr}\left(
\delta \mathbf{S}_{[t]}^{C\top} \mathbf{S}_{[t-1]}^C
+
\delta \mathbf{S}_{[t]}^{C\top} 
\left(
\mathbf{K}_{[t]}^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\mathbf{V}_{[t]}^\top
\right)
\right)
\exp \boldsymbol{\gamma}_{[t]}^C
\\
\\
\delta \boldsymbol{\gamma}_{[t],\text{part1}}
&=
\delta \boldsymbol{\gamma}_{[t],\text{part1}} \odot \exp\boldsymbol{\gamma}_{[t]}
\\&=
\text{diag}\left(
\left(
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]}) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\right)
\mathbf{Q}_{[t]}^\top
\right)
\\&-
\text{diag}\left(
\mathbf{K}_{[t]}
\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
\right)^\top
\right)
\\&+
\text{diag}\left(
\left(
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]}) \delta \mathbf{B}_{[t]} 
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]})^{-1}
\right)
\left(
\mathbf{Q}_{[t]}\mathbf{K}_{[t]}^\top
\right)^\top
\right)
\\&-
\text{diag}\left(
\left(
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]}) \delta \mathbf{B}_{[t]} 
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]})^{-1}
\right)^\top
\left(
\mathbf{Q}_{[t]}\mathbf{K}_{[t]}^\top
\right)
\right)
\\&+
[0,0,...,\delta \boldsymbol{\gamma}_{[t]}^C
\exp \boldsymbol{\gamma}_{[t]}^C]^\top
\\&=
\left(
\left.\delta \text{Diag}(\exp \boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{S}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
+
\left.\delta \text{Diag}(\exp \boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
+
\left.\delta \text{Diag}(\exp \boldsymbol{\gamma}_{[t]})\right|_{\text{from } \gamma_{[t]}^C}
\right)
\odot \exp\boldsymbol{\gamma}_{[t]}
\end{aligned}$$


**Corresponding code**: `fla.ops.common.chunk_o -> chunk_bwd_dqkwg`

### Step 6: Compute the remaining contributions to dK, dV, d beta, and d gamma


$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \overleftarrow{\mathbf{W}_{[t]}}  \text{ w/o } \mathbf{T}_{[t]} }
&=
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]}) 
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\left(
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
\right)
\\
\\
\delta \mathbf{V}_{[t]}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \mathbf{U}_{[t]}
\\
\\
\delta \widetilde{\mathbf{A}}_{[t]}
&=
\delta \overleftarrow{\mathbf{W}_{[t]}} (\text{Diag}(\boldsymbol{\beta}_{[t]} \exp \boldsymbol{\gamma}_{[t]})\mathbf{K}_{[t]})^\top 
+
\delta \mathbf{U}_{[t]} (\text{Diag}(\boldsymbol{\beta}_{[t]})\mathbf{V}_{[t]} )^\top
\\
\\
\delta \widetilde{\mathbf{A_x}}_{[t]}
&= -
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \widetilde{\mathbf{A}}_{[t]} 
\widetilde{\mathbf{A}}_{[t]}^\top
\text{Diag}(\exp \boldsymbol{\gamma}_{[t]})^{-1}
\odot 
\mathbf{M_{-1}}
\\&=
\text{Diag}(\boldsymbol{\beta}_{[t]})^{-1}
\left.\delta (\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top)\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{X}_{[t]}}
&=
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{A_x}}_{[t]}
\right) 
\mathbf{K}_{[t]}
+
\left(
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{K}_{[t]}
\right)^\top
\delta \widetilde{\mathbf{A_x}}_{[t]}
\right)^\top
\\
\\
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \overleftarrow{\mathbf{W}_{[t]}}
\mathbf{K}_{[t]}^\top
\right)
+
\text{diag}\left(
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{U}_{[t]} 
\mathbf{V}_{[t]}^\top
\right)
+
\text{diag}\left(
\delta \widetilde{\mathbf{A_x}}_{[t]}
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top\right)
\\
\\
\delta \boldsymbol{\gamma}_{[t],\text{part2}} 
&=
\text{diag}\left(
\left.\delta \text{Diag}(\exp \boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{U}_{[t]} \text{w/ } \mathbf{W}_{[t]} \text{w/o } \widetilde{\mathbf{A}}_{[t]} }
\right)
\odot \exp \boldsymbol{\gamma}_{[t]}
+
\text{diag}\left(
\left.\delta \text{Diag}(\exp \boldsymbol{\gamma}_{[t]})\right|_{\text{from } \widetilde{\mathbf{A}}_{[t]} }
\right)
\odot \exp \boldsymbol{\gamma}_{[t]}
\\&=
\text{diag}\left(
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}}_{[t]}
\left(
\mathbf{K}_{[t]}
\text{Diag}(\boldsymbol{\beta}_{[t]})
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\right)^\top 
\right)
\\&+
\text{diag}\left(
\delta \widetilde{\mathbf{A_x}}_{[t]}
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)^\top
\right)
-
\text{diag}\left(
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)^\top
\delta \widetilde{\mathbf{A_x}}_{[t]}
\right)
\end{aligned}$$

**Corresponding code**: `fla.ops.gated_delta_rule.wy_fast -> prepare_wy_repr_bwd`

### Step 7: Recover the gradient with respect to log α

$$\begin{aligned}
\delta \log \boldsymbol{\alpha}_{[t]}
&=
\text{suffix\_cumsum}(\delta\mathbf{\gamma}_{[t]})
\end{aligned}$$

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`

