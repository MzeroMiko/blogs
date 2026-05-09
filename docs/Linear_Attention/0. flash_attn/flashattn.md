
# The Hidden Recurrence in Softmax Attention: A Path to Flash Attention

> **Paper**: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)   
> **Paper**: [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)   
> **Code**: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/)  
> **Disclaimer**: These are personal reading notes. Some derivations are my own and may be incorrect, so they should be cross-checked against the code later.   

## Standard Attention

### Forward Pass
$$\begin{aligned}
\mathbf{O} &= 
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1}\right)^{-1}
\exp(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}) \mathbf{V}
\end{aligned}$$

![](assets/Pasted%20image%2020260506141503.png)

### Backward Pass

Before deriving the standard attention backward, considering a simple case:

$$\begin{aligned}
\mathbf{Y} &= \text{Diag}\left( \mathbf{X}\boldsymbol{1}\right)^{-1}
\\ \\ \Rightarrow
\delta \mathbf{X} &= 
\text{diag}\left(
- \mathbf{Y}^\top \delta \mathbf{Y} \mathbf{Y}^\top
\right) \boldsymbol{1}^\top
= 
- \text{diag}\left(
\mathbf{Y} \delta \mathbf{Y} \mathbf{Y}
\right) \boldsymbol{1}^\top
= 
- \mathbf{Y}^2  \text{diag}\left(
\delta \mathbf{Y}
\right) \boldsymbol{1}^\top
\end{aligned}$$

Then the standard attention backward can be derived as follow:

$$\begin{aligned}
\mathbf{O} &= 
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1}\right)^{-1}
\exp(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}) 
\mathbf{V}
\\ 
\\ 
\Rightarrow
\delta \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M})
&=
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1}\right)^{-1} 
\delta \mathbf{O}
\mathbf{V}^\top
\\ &-
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1}\right)^{-2} 
\text{diag}\left(
\delta \mathbf{O}  \mathbf{V}^\top
\exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M})^\top\right) 
\boldsymbol{1}^\top
\\ &=
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1}\right)^{-1} 
\left(
\delta \mathbf{O} \mathbf{V}^\top
-
\text{diag}\left(\delta \mathbf{O}  \mathbf{O}^\top
\right) 
\boldsymbol{1}^\top
\right)
\\ \\ \Rightarrow
\delta \mathbf{V} &= 
\exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M})^\top
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \boldsymbol{1} \right)^{-1}
\delta \mathbf{O}
\\ \\
\delta \mathbf{Q} &= \left(
\left( \delta \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \right) 
\odot \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M})
\odot \mathbf{M}
\right) \mathbf{K}
\\ \\
\delta \mathbf{K} &=  \left(
\left( \delta \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M}) \right) 
\odot \exp(\mathbf{Q}\mathbf{K}^\top  \odot \mathbf{M})
\odot \mathbf{M}
\right) ^\top \mathbf{Q}
\end{aligned}$$

![](assets/Pasted%20image%2020260507175224.png)


## The Hidden Recurrence behind Softmax Attention

### Recurrent mode
> **comment**: The key to designing a tile-based algorithm where every local part relates to the global information is to find the recurrence. So here we find the hidden recurrent mode behind softmax dot-product attention.

Considering 
$$\begin{aligned}
\mathbf{O} 
&=  \frac{\exp(\mathbf{Q}\mathbf{K}^\top)}{\sum_k \exp(\mathbf{Q}\mathbf{K}^\top)} \mathbf{V}
\quad \Leftrightarrow \quad
\mathbf{O} = 
\text{Diag}\left( \exp(\mathbf{Q}\mathbf{K}^\top) \boldsymbol{1}\right)^{-1}
\exp(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}
\\ \\ \Leftrightarrow 
\boldsymbol{o}_t 
&=  \frac{ \sum_{i=1}^L  \boldsymbol{v}_i \exp(\boldsymbol{k}_i^\top\boldsymbol{q}_t) }
{\sum_{i=1}^L \exp(\boldsymbol{k}_i^\top\boldsymbol{q}_t)} 
\end{aligned}$$

We set

$$\begin{aligned}
\boldsymbol{a}^j_t 
=  
\sum_{i=1}^t  \boldsymbol{v}_i \exp(\boldsymbol{k}_i^\top\boldsymbol{q}_j)
,\quad
l^j_t =  \sum_{i=1}^t \exp(\boldsymbol{k}_i^\top\boldsymbol{q}_j)
,\quad
\boldsymbol{a}^t_L = l^j_t \boldsymbol{o}_t
\end{aligned}$$

Then we have

$$\begin{aligned}
l^j_t &= l^j_{t-1} + \exp(\boldsymbol{k}_t^\top\boldsymbol{q}_j)
\\
\\
\boldsymbol{a}^j_{t} &=  \sum_{i=1}^t  \boldsymbol{v}_i \exp(\boldsymbol{k}_i^\top\boldsymbol{q}_j) 
= 
\boldsymbol{a}^j_{t-1} 
+ \boldsymbol{v}_t \exp(\boldsymbol{k}_t^\top\boldsymbol{q}_j)
\end{aligned}$$

### Chunk-wise Mode

We set $\mathbf{M}$ as the causal mask or no mask below.

$$\begin{aligned}
\mathbf{O}_{[t]} &= \text{Diag}\left(\boldsymbol{l}^j_{t}\right)^{-1} \mathbf{A}^{[t]}_{[L]} 
\\ \\
\boldsymbol{ll}^j_{[t]} &= \boldsymbol{ll}^j_{[t-1]} 
+ 
\exp \left(\mathbf{Q}_{[j]}\mathbf{K}_{[t]}^\top  \odot \mathbf{M}\right)  \boldsymbol{1}
\\ \\
\mathbf{A}^{[j]}_{[t]} &= 
\mathbf{A}^{[j]}_{[t-1]}
+
\exp \left(\mathbf{Q}_{[j]}\mathbf{K}_{[t]}^\top  \odot \mathbf{M}\right) \mathbf{V}_{[t]}
\end{aligned}$$

After we set

$$\begin{aligned}
\mathbf{S}_{[t],[\tau]} &= \mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top  \odot \mathbf{M}
\in \mathbb{R}^{N \times M}
,\quad
\boldsymbol{m}_{[t],\tau} 
\in \mathbb{R}^{N}
\\ \\
\mathbf{P}_{[t],[\tau]} &= \exp(\mathbf{S}_{[t],[\tau]} - \log\boldsymbol{m}_{[t],\tau} \boldsymbol{1}^\top)
=
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)^{-1}
\exp \left(\mathbf{S}_{[t],[\tau]}\right)
\in \mathbb{R}^{N \times M}
\\ \\
\boldsymbol{l}_{[t],[\tau]} &=  \mathbf{P}_{[t],[\tau]} \boldsymbol{1}
\in \mathbb{R}^{N}
\end{aligned}$$

we have

$$\begin{aligned}
\boldsymbol{l}^t_{[\tau]} &= 
\boldsymbol{ll}^t_{[\tau]} \oslash \boldsymbol{m}^{\tau}_{[t]}
=
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\boldsymbol{l}^t_{[\tau-1]}
+ \text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\boldsymbol{l}_{[t],[\tau]}
\\ \\
\mathbf{A}^{[t]}_{[\tau]} &= \mathbf{A}^{[t]}_{[\tau]} 
+ \text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\mathbf{P}_{[t],[\tau]}\mathbf{V}_{[\tau]}
\end{aligned}$$


## Chunk-wise Mode with Online Softmax Re-scale

For numerical stability, we often consider the rescaling trick for softmax. We set the scale as the row-max of past $\mathbf{QK}^\top$. So we define:

$$\begin{aligned}
\boldsymbol{m}^{\tau}_{[t]} &= 
\max(\boldsymbol{m}^{\tau-1}_{[t]}, \boldsymbol{m}_{[t],\tau})
\\
\\
\boldsymbol{l}^t_{[\tau]} &=
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\boldsymbol{l}^t_{[\tau-1]}
+ \text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\boldsymbol{l}_{[t],[\tau]}
\\
\\
\mathbf{C}^{[t]}_{[\tau]} &=
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{A}^{[t]}_{[\tau]} 
\\&=
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{C}^{[t]}_{[\tau-1]} 
+
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{P}_{[t],[\tau]}\mathbf{V}_{[\tau]}
\\
\\
\mathbf{O}^{[t]} &= 
\text{Diag}\left(\boldsymbol{l}^t_{L}\right)^{-1}
\mathbf{C}^{[t]}_{[L]} 
\end{aligned}$$

Now, with the addition of the inner-outer loop exchange and the position where the logsumexp scale is introduced, we've essentially derived the core logic of FlashAttention v1/v2 from scratch.


![](assets/Pasted%20image%2020260506142138.png)

![](assets/Pasted%20image%2020260509154608.png)

## Flash Attention Backward Pass

> **comment**: The row-max $\boldsymbol{m}$ is computed from the input, but can be treated as a constant for gradient purposes. However, $\boldsymbol{l}$ is a smooth function of the input, so gradient does flow through it.

### Revisiting the Forward Pass

$$\begin{aligned}
\mathbf{P}_{[t],[\tau]} &= 
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)^{-1}
\exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top \odot \mathbf{M} \right)
\in \mathbb{R}^{N \times M}
\\
\\
\boldsymbol{l}^t_{[\tau]} &= 
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\boldsymbol{l}^t_{[\tau-1]}
+ \text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{P}_{[t],[\tau]} \boldsymbol{1}
\\
\\
\mathbf{C}^{[t]}_{[\tau]} &=
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{C}^{[t]}_{[\tau-1]} 
+
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\mathbf{P}_{[t],[\tau]}\mathbf{V}_{[\tau]}
\\
\\
\mathbf{O}^{[t]} &= 
\text{Diag}\left(\boldsymbol{l}^t_{L}\right)^{-1}
\mathbf{C}^{[t]}_{[L]} 
\end{aligned}$$

### Gradient at the Output

Assuming we have stored $\boldsymbol{l}$ and $\boldsymbol{m}$ from the forward pass, we begin back propagation from $\mathbf{O}^{[t]}$:

$$\begin{aligned}
\delta \mathbf{C}^{[t]}_{[L]} &=  
\text{Diag}\left(\boldsymbol{l}^t_{L}\right)^{-1}
\delta \mathbf{O}^{[t]}
\\
\\
\delta \boldsymbol{l}^{[t]}_{[L]} &= -
\text{diag}\left(
\text{Diag}\left(\boldsymbol{l}^t_{L}\right)^{-\top}
\left(\delta \mathbf{O}^{[t]} (\mathbf{C}^{[t]}_{[L]})^\top \right) 
\text{Diag}\left(\boldsymbol{l}^t_{L}\right)^{-\top}
\right) 
= -
\text{diag}\left(\delta \mathbf{O}^{[t]} (\mathbf{O}^{[t]})^\top \right)
\oslash \boldsymbol{l}^t_{L}
\\
\\
\delta \boldsymbol{l}^t_{[\tau-1]}
&=
\left(
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\right)^\top
\delta \boldsymbol{l}^t_{[\tau]} 
= 
\left(
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^L_{[t]}\right)^{-1}
\right)^\top
\delta \boldsymbol{l}^{[t]}_{[L]}
\\
\\
\delta \mathbf{C}^{[t]}_{[\tau-1]}
&=
\left(
\text{Diag}\left(\boldsymbol{m}^{\tau-1}_{[t]}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\right)^\top
\delta \mathbf{C}^{[t]}_{[\tau]}
\\
\\
\left.\delta \mathbf{V}_{[\tau]}\right|_{\text{from [t]}}
&= \mathbf{P}_{[t],[\tau]}^\top
\left(
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\right)^\top
\delta \mathbf{C}^{[t]}_{[\tau]}
= 
\exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top\right)^\top
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\delta \mathbf{C}^{[t]}_{[\tau]} 
\\
\\
\delta \mathbf{P}_{[t],[\tau]}
&=
\left(
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\right)^\top
\left(
\delta \boldsymbol{l}^t_{[\tau]} 
\boldsymbol{1}^\top
+
\delta \mathbf{C}^{[t]}_{[\tau]} \mathbf{V}_{[\tau]}^\top
\right)
\\ &=
\left(
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^L_{[t]}\right)^{-1}
\right)^\top
\delta \mathbf{l}^{[t]}_{[L]}
\boldsymbol{1}^\top
+
\left(
\text{Diag}\left(\boldsymbol{m}_{[t],\tau}\right)
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\right)^\top
\delta \mathbf{C}^{[t]}_{[\tau]} \mathbf{V}_{[\tau]}^\top
\\
\\
\delta \exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top \odot \mathbf{M}\right)
&=
\text{Diag}\left(\boldsymbol{m}^L_{[t]}\right)^{-1}
\delta \mathbf{l}^{[t]}_{[L]}
\boldsymbol{1}^\top
+
\text{Diag}\left(\boldsymbol{m}^\tau_{[t]}\right)^{-1}
\delta \mathbf{C}^{[t]}_{[\tau]} \mathbf{V}_{[\tau]}^\top
\\
\\
\left.\delta \mathbf{Q}_{[t]}\right|_{\text{from } \mathbf{P}_{[t],[\tau]}}
&=
\left(
\delta \exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top  \odot \mathbf{M}\right) 
\odot \exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top  \odot \mathbf{M}\right)
\odot \mathbf{M}
\right) \mathbf{K}_{[\tau]}
\\ &=
\left(
\delta \mathbf{P}_{[t],[\tau]} \odot \mathbf{P}_{[t],[\tau]}
\right) \mathbf{K}_{[\tau]}
\\ 
\\
\left.\delta \mathbf{K}_{[\tau]}\right|_{\text{from } \mathbf{P}_{[t],[\tau]}}
&=
\left( 
\delta \exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top  \odot \mathbf{M}\right) 
\odot \exp \left(\mathbf{Q}_{[t]}\mathbf{K}_{[\tau]}^\top  \odot \mathbf{M}\right)
\odot \mathbf{M}
\right) ^\top \mathbf{Q}_{[t]}
\\ &=
\left(
\delta \mathbf{P}_{[t],[\tau]} \odot \mathbf{P}_{[t],[\tau]}
\right) ^\top \mathbf{Q}_{[t]}
\end{aligned}$$

> **comment**: The key insight of the backward pass is that re-computation replaces storage. Rather than saving the full attention matrix $\mathbf{P}$ from the forward pass, we only store $\boldsymbol{l}$ and $\boldsymbol{m}$, then recompute the attention matrix tile-by-tile during the backward pass. 

![](assets/Pasted%20image%2020260507175302.png)

![](assets/Pasted%20image%2020260509154912.png)

