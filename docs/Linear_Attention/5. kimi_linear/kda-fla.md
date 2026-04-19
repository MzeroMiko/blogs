# Reading Notes: Implementation Notes for KDA in FLA

> **Paper**: [https://arxiv.org/pdf/2510.26692](https://arxiv.org/pdf/2510.26692)   
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect.  

## Forward 

Entry function signature:  

```python

def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
):
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
```

**Notation**


$$\begin{aligned}
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


### Step 1: Compute the cumulative gate in the log domain

Here, `gamma` corresponds to what was previously defined as `log gamma`, so this should be kept in mind in the formulas that follow.

$$\begin{aligned}
\boldsymbol{\gamma}_{[t]}^r &= \sum_{i=tC+1}^{tC+r} \log \alpha_i \in \mathbb{R}
\end{aligned}$$

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`


### Step 2: Solve the lower-triangular system


$$\begin{aligned}
\mathbf{A_{qk}}_{[t]} &= \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}
,\quad
\mathbf{A_{kk0}}_{[t]} = 
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\left(
\overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}
\right)
\\
\\
\mathbf{A}_{[t]} &= (\mathbf{I} + \mathbf{A_{kk0}}_{[t]})^{-1}
\end{aligned}$$


**Corresponding code**: `fla.ops.kda.chunk_intra -> chunk_kda_fwd_intra -> chunk_kda_fwd_kernel_intra_sub_chunk / chunk_kda_fwd_kernel_inter_solve_fused`



### Step 3: Compute U and the W

$$\begin{aligned}
\mathbf{U}_{[t]} 
&= 
\mathbf{A}_{[t]} 
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{V}_{[t]}
\right)
\\
\\
\mathbf{W}_{[t]} 
&= 
\mathbf{A}_{[t]}
\left(
(\exp \boldsymbol{\Gamma}_{[t]}) \odot
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{K}_{[t]}
\right)
\\
\\
\mathbf{Q_g}_{[t]} 
&= 
(\exp \boldsymbol{\Gamma}_{[t]}) \odot \mathbf{Q}_{[t]}
\\
\\
\mathbf{K_g}_{[t]} 
&= 
(\exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]})) \odot
\mathbf{K}_{[t]}
\end{aligned}$$

To save memory bandwidth during the forward pass, $\mathbf{Q_g}$ is not materialized if activation recomputation is enabled. Instead, it is computed on-the-fly during the backward pass.

**Corresponding code**: `fla.ops.kda.wy_fast -> recompute_w_u_fwd`



### Step 4: Recursively compute the hidden state

Here, `S_[t]^C` uses the `[d_k, d_v]` layout, so it is `transposed relative` to the previous derivation.

$$\begin{aligned}
\mathbf{V}_{[t],new} 
&= \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\mathbf{S}_{[t]}^{C} 
&= \boldsymbol{\gamma}_{[t]}^C \mathbf{S}_{[t-1]}^{C}
+ \mathbf{K_g}_{[t]}^\top \mathbf{V}_{[t],new}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`



### Step 5: Compute the final output

$$\begin{aligned}
\mathbf{O}_{[t]}
= 
\left((\exp \boldsymbol{\Gamma}_{[t]}) \odot \mathbf{Q}_{[t]}\right) \mathbf{S}_{[t-1]}^{C}  
+ \mathbf{A_{qk}}_{[t]}   \mathbf{V}_{[t],new}
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_o_gk`



## Backward

Entry function signature:

```python
def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
):
	return dq, dk, dv, db, dg, dh0, dA, dbias
```

### Step 1: Recompute U and W

`A_[t] = (I + A0_[t])^{-1}` has already been stored. Everything else is the same as in the forward pass except for extra calculation for `Q_g`

**Corresponding code**: `fla.ops.kda.wy_fast -> recompute_w_u_fwd`


### Step 2: Recompute the hidden state

This is the same as in the forward pass.

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_fwd_h`


### Step 3: Derive Initial Gradients from Output ($\delta \mathbf{A_{qk}}$ and $\delta \mathbf{U}$)

$$\begin{aligned}
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
&=
\mathbf{A_{qk}}_{[t]}^\top \delta \mathbf{O}_{[t]} 
\\
\\
\delta \mathbf{A_{qk}}_{[t]}
&=
\delta \mathbf{O}_{[t]} \mathbf{V}_{[t],new}^\top \odot \mathbf{M}
\end{aligned}$$

**Corresponding code**: `fla.ops.kda.chunk_bwd -> chunk_kda_bwd_dAv -> chunk_kda_bwd_kernel_dAv`


### Step 4: Inter-Chunk Backward Recurrence for Hidden States

$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\mathbf{K_g}_{[t]} \delta \mathbf{S}_{[t]}^{C} 
+
\left.\delta \mathbf{U}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
\end{aligned}$$

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C}
=
\exp(\boldsymbol{\gamma}_{[t]}^C)
\delta \mathbf{S}_{[t+1]}^{C}
+
\mathbf{Q_g}_{[t+1]}^\top   \delta \mathbf{O}_{[t+1]}
-
\mathbf{W}_{[t+1]}^\top
\delta \mathbf{U}_{[t+1]} 
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_delta_h -> chunk_gated_delta_rule_bwd_dhu`

### Step 5:  The Fused WY-Representation Kernel

$$\begin{aligned}
\delta \mathbf{Q}_{[t],part1}
&=
(\exp \boldsymbol{\Gamma}_{[t]}) \odot
\mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C\top}
\\
\\
\delta \mathbf{W}_{[t]}
&= - \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\left.\delta \mathbf{K}_{[t], part1}\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \mathbf{A}_{[t]} }
&=
\exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]}) \odot
\mathbf{V}_{[t],new} \delta \mathbf{S}_{[t]}^{C\top}
\\
\\
\left.\delta \mathbf{K}_{[t],part1}\right|_{\text{from } \mathbf{W}_{[t]} \text{w/o} \mathbf{A}_{[t]} }
&=
\mathbf{A}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
\odot (\text{Diag}(\boldsymbol{\beta}_{[t]}) \exp \mathbf{\Gamma}_{[t]} )
\\
\\
\delta \mathbf{V}_{[t]}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{A}_{[t]}^\top 
\delta \mathbf{U}_{[t]}
\\
\\
\delta \boldsymbol{\gamma}_{[t]}^C
&=
\delta \exp \boldsymbol{\gamma}_{[t]}^C 
\odot \exp \boldsymbol{\gamma}_{[t]}^C
\\ &=
\text{diag}\left(
\mathbf{S}_{[t-1]}^C
\delta \mathbf{S}_{[t]}^C 
\right)
\odot \exp \boldsymbol{\gamma}_{[t]}^C
+
\left.\delta \mathbf{K}_{[t], part1}\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \mathbf{A}_{[t]} }
\odot \mathbf{K}_{[t]}
\\
\\
\delta \mathbf{\Gamma}_{[t],part1}
&=
\left.\delta \exp \mathbf{\Gamma}_{[t]}\right|_{\text{w/o extra} \boldsymbol{\gamma}_{[t]}^C}
\odot \exp \mathbf{\Gamma}_{[t]}
\\ &=
\delta \mathbf{Q}_{[t],part1} \odot \mathbf{Q}_{[t]}
-
\left.\delta \mathbf{K}_{[t], part1}\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
\odot \mathbf{K}_{[t]}
+
\left.\delta \mathbf{K}_{[t],part1}\right|_{\text{from } \mathbf{W}_{[t]} \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
\odot \mathbf{K}_{[t]}
\\ &+
[0,0,...,\delta \boldsymbol{\gamma}_{[t]}^C ]
\\
\\
\delta \boldsymbol{\beta}_{[t],part1} 
&=
\text{diag}(\delta \text{Diag}(\boldsymbol{\beta}_{[t]}))
=
\text{diag}(
\mathbf{A}_{[t]}^\top
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
)
+
\text{diag}(
\mathbf{A}_{[t]}^\top
\delta \mathbf{W}_{[t]}
(
\boldsymbol{\Gamma}_{[t]} \odot \mathbf{K}_{[t]}
)^\top
)
\\
\\
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}(\delta \text{Diag}(\boldsymbol{\beta}_{[t]}))
\\ &=
\text{diag}\left(
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \widetilde{\mathbf{A}}_{[t]}
\right)
\odot 
\boldsymbol{\beta}_{[t]}^{-1}
+
\text{diag}\left(
\left(
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1} 
\right)
\left( 
\overrightarrow{\mathbf{K}_{[t]}} \overleftarrow{\mathbf{K}_{[t]}}^\top  
\right)
\right)
\\
\\
\delta \mathbf{A}_{[t]} 
&=
\left(
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
+
\delta \mathbf{W}_{[t]}
(\boldsymbol{\Gamma}_{[t]} \odot \mathbf{K}_{[t]})^\top
\right)
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\\
\\
\delta \mathbf{A_X}_{[t]}
&=
- \mathbf{A}_{[t]}^\top 
\delta \mathbf{A}_{[t]}
\mathbf{A}_{[t]}^\top \odot \mathbf{M}_{-1}
\end{aligned}$$


**Corresponding code**: `fla.ops.kda.chunk_bwd -> chunk_kda_bwd_wy_dqkg_fused`


### Step 6: Intra-Chunk Gradients

$$\begin{aligned}
\delta \mathbf{Q}_{[t],part2}
&=
\delta \mathbf{A_{qk}}_{[t]}
\left(
\exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]})
\odot \mathbf{K}_{[t]})
\right)
\odot \exp (\boldsymbol{\Gamma}_{[t]} - \boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top})
\\
\\
\delta \mathbf{K}_{[t],part2}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \mathbf{A_X}_{[t]}
\left(
\mathbf{K}_{[t]} \odot \exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]})
\right)
\odot \exp (\boldsymbol{\Gamma}_{[t]} - \boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top})
\\&+
\delta \mathbf{A_{qk}}_{[t]}^\top
\left( \mathbf{Q}_{[t]}  \odot \exp (\boldsymbol{\Gamma}_{[t]} - \boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top})\right)
\odot \exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]})
\\&+
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \mathbf{A_X}_{[t]}^\top
\left( \mathbf{K}_{[t]}  \odot \exp (\boldsymbol{\Gamma}_{[t]} - \boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top})\right)
\odot \exp (\boldsymbol{1} \boldsymbol{\gamma}_{[t]}^{C\top} - \boldsymbol{\Gamma}_{[t]})
\\&=\left.\delta \mathbf{K}_{[t]}\right|_{\leftarrow \text{from } \mathbf{A}_{[t]} }
+
\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\rightarrow \text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{A}_{[t]} }
+
\left.\delta \mathbf{K}_{[t]}\right|_{\rightarrow \text{from }\mathbf{A}_{[t]} }
\right)
\\
\\
\delta \mathbf{\Gamma}_{[t],part2}
&=
\delta \exp \mathbf{\Gamma}_{[t]}
\odot \exp \mathbf{\Gamma}_{[t]}
\\&=
\delta \mathbf{Q}_{[t],part2} \odot \mathbf{Q}_{[t]}
+
\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\leftarrow \text{from } \mathbf{A}_{[t]} }
-
\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\rightarrow \text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{A}_{[t]} }
+
\left.\delta \mathbf{K}_{[t]}\right|_{\rightarrow \text{from }\mathbf{A}_{[t]} }
\right)
\right)
\odot \mathbf{K}_{[t]}
\end{aligned}$$


**Corresponding code**: `fla.ops.kda.chunk_intra -> chunk_kda_bwd_intra -> chunk_kda_bwd_kernel_intra`

> ***Comment*** In the function `chunk_kda_bwd_kernel_intra`, the diagonal computation of `secondary chunking` is carefully optimized using the `SAFE_GATE` mechanism. Naively calculating $\exp(g_i - g_j)$ for intra-block causal dependencies risks FP16 overflow, typically forcing a fallback to slow, token-by-token vector multiplications. `SAFE_GATE` elegantly resolves this by gathering a local reference point to normalize the exponents (acting as a local max-trick). This numerical stabilization safely replaces the sequential vector loops with hardware-accelerated `tl.dot`, drastically boosting computational speed while preserving mathematical precision.

### Step 7: Recover Gradients for the Log Gate

$$\begin{aligned}
\delta \log \boldsymbol{\alpha}_{[t]}
&=
\text{suffix\_cumsum}(\delta\mathbf{\gamma}_{[t]})
\end{aligned}$$

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`

