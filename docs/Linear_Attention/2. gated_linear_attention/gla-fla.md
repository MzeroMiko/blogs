# Reading Notes: Implementation Notes for GLA in FLA 

> **Paper**: [https://arxiv.org/pdf/2312.06635](https://arxiv.org/pdf/2312.06635)   
> **Code**: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/)   
> **Disclaimer**: These notes provide a step-by-step mathematical reading of the Triton implementation of GLA in the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/) repository, and are intended to be read together with the previous note on the GLA paper.   

## 1. Forward Entry function signature:  

```python
def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
return g_cumsum, A, h, ht, o
```

### Step 1: Compute the cumulative gate in the log domain

> **Notation note**: In this note, `gamma` and `Gamma` correspond to `log gamma` and `log Gamma` in the previous GLA note. Therefore, all formulas involving `gamma` and `Gamma` below are written in the log domain.


$$\begin{aligned}
\boldsymbol{\gamma}_{[t]}^r &= \sum_{i=tC+1}^{tC+r} \log \boldsymbol{\alpha}_i \in \mathbb{R}^{d \times 1}
\\
\\
\mathbf{\Gamma}_{[t]} &= [ \boldsymbol{\gamma}_{[t]}^{1}, \boldsymbol{\gamma}_{[t]}^{2}, \dots, \boldsymbol{\gamma}_{[t]}^{C} ]^\top \in \mathbb{R}^{C \times d}
\end{aligned}$$

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`

### Step 2: Recursively compute the hidden state

> **Notation note**: In this note, `S_[t]^C` corresponds to `S_[t]^{C\top}` in the previous GLA note. The same convention is used throughout.

$$\begin{aligned}
\mathbf{S}_{[t]}^{C}
&=
\mathbf{S}_{[t-1]}^{C} \odot \exp(\boldsymbol{\gamma}_{[t]}^{C} \boldsymbol{1}^\top) 
+ 
\left( 
\boldsymbol{K}_{[t]}^\top \odot
\exp\left(\boldsymbol{\gamma}_{[t]}^{C} \boldsymbol{1}^\top - \mathbf{\Gamma}_{[t]}^\top \right) 
\right)
\mathbf{V}_{[t]}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_h -> chunk_fwd_h`

### Step 3: Compute the intra-chunk attention matrix A for off-diagonal sub-blocks

In this step, we only consider the `[t]`-th chunk. Let `A_[i,j]` denote the `(i,j)`-th sub-block, let `Gamma_[i]` denote the cumsum values of the `i`-th sub-chunk, and let `gamma_[i]^k` denote the `k`-th element in the `i`-th sub-chunk.

$$\begin{aligned}
\mathbf{A}_{[i,j]} 
= 
\left(\mathbf{Q}_{[i]} \odot 
\exp\left(\mathbf{\Gamma}_{[i]} -  \boldsymbol{\gamma}_{[i]}^1 \boldsymbol{1}^\top  \right)  
\right)
\left(\mathbf{K}_{[j]}^\top \odot 
\exp\left(\boldsymbol{\gamma}_{[i]}^{1\top} \boldsymbol{1} - \mathbf{\Gamma}_{[j]}^{\top} \right)  
\right)
,\quad i \gt j
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_intra_gk -> chunk_gla_fwd_A_kernel_intra_sub_inter`

### Step 4: Compute the intra-chunk attention matrix A for diagonal sub-blocks

Again, we only consider the `[t]`-th chunk, and the notation is the same as in Step 3.

On diagonal sub-blocks, `A` is computed column by column in the implementation. For simplicity, however, we write it here in elementwise form.

$$\begin{aligned}
\mathbf{A}_{[i,j]}^{k_1, k_2} 
= 
\sum_{d}
\left(\mathbf{Q}_{[i]}^{k_1} \odot \mathbf{K}_{[j]}^{k_2} \odot
\exp\left(\mathbf{\Gamma}_{[i]}^{k_1} -   \mathbf{\Gamma}_{[j]}^{k_2}  \right)  
\right)
, \quad i = j, k_1 \geq k_2
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_intra_gk -> chunk_gla_fwd_A_kernel_intra_sub_intra`

If the computation is split along the `d` dimension and then merged, the following two functions are used. Their mathematical content is essentially the same as above:

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_intra_gk -> chunk_gla_fwd_A_kernel_intra_sub_intra_split`  
**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_intra_gk -> chunk_gla_fwd_A_kernel_intra_sub_intra_merge`

### Step 5: Compute the final output

Since `A` is initialized with `torch.empty` in `chunk_gla_fwd_intra_gk`, the upper-triangular part may contain garbage values. Therefore, a causal mask needs to be applied once again when computing the final output.

$$\begin{aligned}
\mathbf{O}_{[t]} 
= 
(\mathbf{Q}_{[t]} \odot \exp\mathbf{\Gamma}_{[t]}) \mathbf{S}_{[t-1]}^{C} 
+
(\mathbf{A}_{[t]} \odot \mathbf{M}) \mathbf{V}_{[t]}
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_fwd_o_gk`

## 2. Backward

Entry function signature:

```python
def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
	return dq, dk, dv, dg, dh0
```

### Step 1: Compute the cumulative gate in the log domain

This is the same as Forward Step 1.

**Corresponding code**: `fla.ops.utils.cumsum -> chunk_local_cumsum`

### Step 2: Recompute the hidden state recursively

This is the same as Forward Step 2.

**Corresponding code**: `fla.ops.common.chunk_h -> chunk_fwd_h`

### Step 3: Recursively compute the gradient of the hidden state

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C} = \delta \mathbf{S}_{[t+1]}^{C} \odot \exp (\boldsymbol{\gamma}_{[t+1]}^{C} \boldsymbol{1}^\top)   + \left(\mathbf{Q}_{[t+1]}^\top \odot \exp(\mathbf{\Gamma}_{[t+1]}^\top)\right) \delta \mathbf{O}_{[t+1]}
\end{aligned}$$

**Corresponding code**: `fla.ops.common.chunk_h -> chunk_bwd_dh`

### Step 4: Compute the gradient with respect to V

$$\begin{aligned}
\delta \mathbf{V}_{[t]} = \left(\mathbf{A}_{[t]}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]} +  \left(\mathbf{K}_{[t]} \odot \exp(\boldsymbol{\gamma}_{[t]}^C - \mathbf{\Gamma}_{[t]}) \right) \delta \mathbf{S}_{[t]}^{C}
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_bwd_dv`

### Step 5: Compute the gradient with respect to A

$$\begin{aligned}
\delta \mathbf{A}_{[t]} = \delta \mathbf{O}_{[t]} \mathbf{V}_{[t]}^{\top} \odot \mathbf{M}
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_bwd_dA`

### Step 6: Compute the intra-chunk gradients of Q and K

In this step, we only consider the `[t]`-th chunk. Let `A_[i,j]` denote the `(i,j)`-th sub-block, let `gamma_[i]^k` denote the cumsum value of the `k`-th element in the `i`-th sub-chunk, and let `gamma_[i]^{C_i}` denote the cumsum value of the last element in the `i`-th sub-chunk.

**Intra-chunk gradient of `delta Q`**:

$$\begin{aligned}
\left.\delta \mathbf{Q}_{[i]}\right|_{\text{from } \mathbf{A}_{[i,j]}} 
&= \delta \mathbf{A}_{[i,j]} \left(\mathbf{K}_{[j]} \odot \exp\left( \boldsymbol{\gamma}_{[i]}^1 \boldsymbol{1}^\top  - \mathbf{\Gamma}_{[j]} 
\right)  \right) \odot \exp\left( 
\mathbf{\Gamma}_{[i]}  - \boldsymbol{\gamma}_{[i]}^1 \boldsymbol{1}^\top
\right)
,\quad i \gt j
\\
\\
\left.\delta \mathbf{Q}_{[i]}^{k_1}\right|_{\text{from } \mathbf{A}_{[i,j]}} 
&= \sum_{k_2 \leq k_1}
\delta \mathbf{A}_{[i,j]}^{k_1, k_2} \mathbf{K}_{[j]}^{k_2} \odot \exp\left( \boldsymbol{\gamma}_{[i]}^{k_1}  - \boldsymbol{\gamma}_{[j]}^{k_2} 
\right)
,\quad i = j
\end{aligned}$$

**Intra-chunk gradient of `delta K`**:


$$\begin{aligned}
\left.\delta \mathbf{K}_{[j]}\right|_{\text{from } \mathbf{A}_{[i,j]}} 
&= \delta \mathbf{A}_{[i,j]}^\top \left(\mathbf{Q}_{[i]} \odot \exp\left( 
\mathbf{\Gamma}_{[i]}  - \boldsymbol{\gamma}_{[i]}^{C_i} \boldsymbol{1}^\top
\right)  \right) \odot \exp\left( \boldsymbol{\gamma}_{[i]}^{C_i} \boldsymbol{1}^\top  - \mathbf{\Gamma}_{[j]} 
\right) 
,\quad i \gt j
\\
\\
\left.\delta \mathbf{K}_{[i]}^{k_2}\right|_{\text{from } \mathbf{A}_{[i,j]}} 
&= \sum_{k_1 \geq k_2}
\delta \mathbf{A}_{[i,j]}^{k_1, k_2 \top} \mathbf{Q}_{[i]}^{k_1} \odot \exp\left( 
\boldsymbol{\gamma}_{[i]}^{k_1}  - \boldsymbol{\gamma}_{[i]}^{k_2}  
\right) 
,\quad i = j
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_bwd_dqk_intra`

### Step 7: Compute the inter-chunk gradients and merge them to obtain the final dQ, dK, and dg

In this step, the inter-chunk contributions are combined with the intra-chunk contributions from Step 6, and the gradient of the gate is also computed.


$$\begin{aligned}
\delta \mathbf{\gamma}_{[t]}^C
&=
(\delta \exp\mathbf{\gamma}_{[t]}) \odot \exp  \boldsymbol{\gamma}_{[t]}^C
\\&=
\left(\sum_{d_v}
\mathbf{S}_{[t-1]}^{C\top} \odot \delta \mathbf{S}_{[t]}^{C\top} \right)
\odot \exp  \boldsymbol{\gamma}_{[t]}^C
+
\sum_{r}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]}} \odot \mathbf{K}_{[t]}
 \quad \text{in float32}
\\
\\
\left.\delta \mathbf{Q}_{[t]}\right|_{\text{from } \mathbf{O}_{[t], \text{part 1}}}
&=
\delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C\top} \odot \exp\mathbf{\Gamma}_{[t]}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]}}
&=
\mathbf{V}_{[t]} \delta  \mathbf{S}_{[t-1]}^{C\top}
\odot \exp\left(\boldsymbol{\gamma}_{[t]}^C - \mathbf{\Gamma}_{[t]}\right)
\\
\\
\delta \mathbf{Q}_{[t]} 
&= 
\left.\delta \mathbf{Q}_{[t]}\right|_{\text{from } \mathbf{O}_{[t], \text{part 1}}} + \left.\delta \mathbf{Q}_{[t]}\right|_{\text{from } \mathbf{A}_{[t]}}
\\
\\
\delta \mathbf{K}_{[t]} 
&= 
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]}} + \left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{A}_{[t]}}
\\
\\
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}
&=
\delta \mathbf{Q}_{[t]} \odot \mathbf{Q}_{[t]}
-
\delta \mathbf{K}_{[t]} \odot \mathbf{K}_{[t]}
\\
\\
\delta \log \boldsymbol{\alpha}_{[t]}
&=
\text{suffix\_cumsum}(\left.\delta\mathbf{\Gamma}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]}}) + \delta \mathbf{\gamma}_{[t]}^C
\end{aligned}$$

**Corresponding code**: `fla.ops.gla.chunk -> chunk_gla_bwd_dqkg`

