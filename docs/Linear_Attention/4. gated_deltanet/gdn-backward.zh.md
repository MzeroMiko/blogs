# 阅读笔记：Gated DeltaNet 的反向传播

### 回顾前向传播

我们先整理反传推导中会用到的前传公式：

$$\begin{aligned}
\mathbf{V}_{[t], new} &= \left(\mathbf{U}_{[t]} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t-1]}^{C \top} \right)
\\
\\
\mathbf{\widetilde{X}}_{[t]} &= \mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
=
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\mathbf{X}_{[t]}
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\
\\
\widetilde{\mathbf{A}}_{[t]} &= \mathbf{\widetilde{X}}_{[t]}^{-1}
,\quad
\widetilde{\mathbf{T}}_{[t]} 
= 
\widetilde{\mathbf{A}}_{[t]} \text{Diag}(\boldsymbol{\beta}_{[t]})
,\quad
\mathbf{T}_{[t]} 
= 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\widetilde{\mathbf{A}}_{[t]}
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\text{Diag}(\boldsymbol{\beta}_{[t]})
\\
\\
\mathbf{S}_{[t]}^{C\top} 
&=
\gamma_{[t]}^{C} \mathbf{S}_{[t-1]}^{C\top}
+
\gamma_{[t]}^{C} 
\overrightarrow{\mathbf{K}_{[t]}}^\top
\mathbf{V}_{[t], new}
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t-1]}^{C \top}  
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M} \right)  \mathbf{V}_{[t], new}
\\
\\
\overleftarrow{\mathbf{W}_{[t]}} 
&= 
\widetilde{\mathbf{A}}_{[t]}
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{K}_{[t]}  
\\
\\
\mathbf{U}_{[t]} 
&= 
\widetilde{\mathbf{A}}_{[t]}
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{V}_{[t]}
\\
\\
\overleftarrow{\mathbf{Q}_{[t]}} &= \text{Diag}(\boldsymbol{\gamma}_{[t]}) \mathbf{Q}_{[t]}
,\quad
\overleftarrow{\mathbf{K}_{[t]}} = \text{Diag}(\boldsymbol{\gamma}_{[t]}) \mathbf{K}_{[t]}
,\quad
\overrightarrow{\mathbf{K}_{[t]}} = \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1} \mathbf{K}_{[t]}
\end{aligned}$$

### 对于 `U_[t]` 的梯度


$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\gamma_{[t]}^C \overrightarrow{\mathbf{K}_{[t]}} \delta \mathbf{S}_{[t]}^{C \top}
+
\left( \overleftarrow{\mathbf{Q}_{[t]}}\overrightarrow{ \mathbf{K}_{[t]}}^\top \odot \mathbf{M}\right)^\top  \delta \mathbf{O}_{[t]}
\end{aligned}$$

### 对于 `W_left_[t]` 的梯度

接下来，对于 `W_left_[t]`，有：

$$\begin{aligned}
\delta \overleftarrow{\mathbf{W}_{[t]}}
=
- \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\end{aligned}$$

### 对于 `S_[t]^C` 的梯度

 
首先，来自 `S_[t]^C` 本身的梯度贡献为：

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{S}_{[t]}^{C}}
&=
\gamma_{[t]}^C \delta \mathbf{S}_{[t]}^{C}
-
\gamma_{[t]}^C  \delta \mathbf{S}_{[t]}^{C} \overrightarrow{\mathbf{K}_{[t]}}^\top  \overleftarrow{\mathbf{W}_{[t]}} 
\end{aligned}$$

同时，来自 `O_[t]` 的梯度贡献为：

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{O}_{[t]}}
&=
\delta \mathbf{O}_{[t]}^\top  \overleftarrow{\mathbf{Q}_{[t]}}
-
\delta \mathbf{O}_{[t]}^\top
\left( \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M} \right)  
\overleftarrow{\mathbf{W}_{[t]}}
\end{aligned}$$

将这两部分合并，就得到：

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C}
&=
\gamma_{[t]}^C \delta \mathbf{S}_{[t+1]}^{C}
+
\delta \mathbf{O}_{[t+1]}^\top  \overleftarrow{\mathbf{Q}_{[t+1]}}
-
\delta \mathbf{U}_{[t+1]}^\top
\overleftarrow{\mathbf{W}_{[t+1]}}
\end{aligned}$$

### 对于 `Q_[t]` 的梯度

接着，`Q_[t]` 的梯度可以写成：

$$\begin{aligned}
\delta \mathbf{Q}_{[t]}
=
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C
+
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],new}^\top \odot \mathbf{M}\right) \overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

### 对于 `V_[t]` 的梯度

由于 `U_[t]` 是 `V_[t]` 的线性变换，因此有：

$$\begin{aligned}
\delta \mathbf{V}_{[t]}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{U}_{[t]}
\end{aligned}$$

### 对于 `A_[t]` 和 `X_[t]` 的梯度


首先，对于 `A_[t]`，有：

$$\begin{aligned}
\delta \widetilde{\mathbf{A}}_{[t]} 
&=
\delta \overleftarrow{\mathbf{W}_{[t]}}
\mathbf{K}_{[t]}^\top
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\text{Diag}(\boldsymbol{\gamma}_{[t]})
+
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\end{aligned}$$

然后，利用逆矩阵的微分公式，可以得到：

$$\begin{aligned}
\delta \widetilde{\mathbf{X}}_{[t]}
&=
- \widetilde{\mathbf{X}}_{[t]}^{-\top} 
\delta (\widetilde{\mathbf{X}}_{[t]}^{-1}) 
\widetilde{\mathbf{X}}_{[t]}^{-\top}
=
- \widetilde{\mathbf{A}}_{[t]}^\top 
\delta \widetilde{\mathbf{A}}_{[t]}
\widetilde{\mathbf{A}}_{[t]}^\top
\end{aligned}$$

### 对于 `K_[t]` 的梯度

`K_[t]` 的梯度来自多条路径。

首先，来自 `X_[t]` 的那部分为：

$$\begin{aligned}
\left.\delta (\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top)\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}
&=
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}
&=
\left(\left.\delta (\mathbf{K}_{[t]}\mathbf{K}_{[t]}^\top)\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}\right) \mathbf{K}_{[t]}
+
\left(\left.\delta (\mathbf{K}_{[t]}\mathbf{K}_{[t]}^\top)\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}\right) ^\top
\mathbf{K}_{[t]}
\end{aligned}$$

接下来，来自 `S_[t]` 且不经过 `V_[t],new` 的那部分为：

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
&=
\gamma_{[t]}^C 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1} 
\mathbf{V}_{[t],new} 
\delta \mathbf{S}_{[t]}^{C}
\end{aligned}$$

类似地，来自 `O_[t]` 且不经过 `V_[t],new` 的那部分为：

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{ w/o } \mathbf{V}_{[t],new} }
&=
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\mathbf{V}_{[t],new} 
\delta \mathbf{O}_{[t]}^\top 
\odot \mathbf{M}^\top 
\right)
\overleftarrow{\mathbf{Q}_{[t]}}
\end{aligned}$$

另外，来自 `W_left_[t]` 且不经过 `T_[t]` 的那部分为：

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \overleftarrow{\mathbf{W}_{[t]}}  \text{ w/o } \mathbf{T}_{[t]} }
&=
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
\end{aligned}$$

因此，把上述所有贡献合并之后，可得：

$$\begin{aligned}
\delta \mathbf{K}_{[t]}
&=
\gamma_{[t]}^C 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1} 
\mathbf{V}_{[t],new} 
\delta \mathbf{S}_{[t]}^{C}
\\&+
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\mathbf{V}_{[t],new} 
\delta \mathbf{O}_{[t]}^\top 
\odot \mathbf{M}^\top 
\right)
\overleftarrow{\mathbf{Q}_{[t]}}
\\&+
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
+ 
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}
\end{aligned}$$

### 对于 `beta_[t]` 的梯度


$$\begin{aligned}
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}\left(\delta \text{Diag}(\boldsymbol{\beta}_{[t]})\right)
\\ &=
\text{diag}\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
\mathbf{K}_{[t]}^\top
+
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
+
\delta \widetilde{\mathbf{X}}_{[t]}
\left(
\overleftarrow{\mathbf{K}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot \mathbf{M}_{-1}
\right)^\top
\right)
\\ &=
\text{diag}\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
\mathbf{K}_{[t]}^\top
+
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
\right)
\\& +
\text{diag}\left(
\left(
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{K}_{[t]}}
\overleftarrow{\mathbf{K}_{[t]}}^\top
\right)
\end{aligned}$$

### 对于 `gamma_[t]` 的梯度


首先有：

$$\begin{aligned}
\delta \boldsymbol{\gamma}_{[t]}^C 
&=
\text{Tr}(\delta \boldsymbol{\gamma}_{[t]}^C \mathbf{I})
=
\text{Tr}\left(
\delta \mathbf{S}_{[t]}^C 
\left(
\mathbf{S}_{[t-1]}^{C}
+ \mathbf{V}_{[t],new}^\top
\overrightarrow{\mathbf{K}_{[t]}}
\right)^\top
\right)
=
\frac{1}{\boldsymbol{\gamma}_{[t]}^C} 
\text{Tr}\left(
\delta \mathbf{S}_{[t]}^C \mathbf{S}_{[t]}^{C \top}
\right)
\end{aligned}$$

接下来，来自 `S_[t]^C` 且不经过 `V_[t],new` 的贡献为：

$$\begin{aligned}
\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \mathbf{V}_{[t],new} }
&=
- \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \mathbf{V}_{[t],new} }\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\ \\ &=
- \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\boldsymbol{\gamma}_{[t]}^C 
\mathbf{V}_{[t],new}
\delta \mathbf{S}_{[t]}^C
\mathbf{K}_{[t]}^\top 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\ \\ &= - \left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
\mathbf{K}_{[t]}^\top 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\end{aligned}$$

同时，来自 `O_[t]` 且不经过 `V_[t],new` 的贡献为：

$$\begin{aligned}
\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
&=
\delta \mathbf{O}_{[t]} 
\mathbf{S}_{[t-1]}^C 
\mathbf{Q}_{[t]}^\top
+
\left( 
\delta \mathbf{O}_{[t]} 
\mathbf{V}_{[t],new}^\top 
\odot \mathbf{M}  
\right) 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\mathbf{K}_{[t]}
\mathbf{Q}_{[t]}^\top
\\ &-
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left( 
\delta \mathbf{O}_{[t]} 
\mathbf{V}_{[t],new}^\top 
\odot \mathbf{M}  
\right)^\top 
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\mathbf{Q}_{[t]}
\mathbf{K}_{[t]}^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\&=
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\delta \mathbf{Q}_{[t]}
\mathbf{Q}_{[t]}^\top
-
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{V}_{[t],new} }
\mathbf{K}_{[t]}^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\end{aligned}$$

同样地，来自 `U_[t]` 与 `W_[t]`，但不经过 `A_[t]` 的贡献为：

$$\begin{aligned}
\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{U}_{[t]} \text{w/ } \mathbf{W}_{[t]} \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
&=
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}} 
\mathbf{K}_{[t]}^\top 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\end{aligned}$$

而来自 `A_[t]` 的贡献为：

$$\begin{aligned}
\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \widetilde{\mathbf{A}}_{[t]} }
&=
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\left(
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\right)^\top
\\&
-\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\left(
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)^\top
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\ &=
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)
\\&
-\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\end{aligned}$$

把所有这些部分合并起来，就得到：

$$\begin{aligned}
\delta \boldsymbol{\gamma}_{[t]} 
&=
\text{diag}\left(\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right)
\\
&= - 
\text{diag}\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
\mathbf{K}_{[t]}^\top 
\right)
\odot \boldsymbol{\gamma}_{[t]}^{-1}
\\ &+
\boldsymbol{\gamma}_{[t]}^{-1} \odot
\text{diag}\left(
\delta \mathbf{Q}_{[t]}
\mathbf{Q}_{[t]}^\top
\right)
-
\text{diag}\left(
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{V}_{[t],new} }
\mathbf{K}_{[t]}^\top
\right)
\odot \boldsymbol{\gamma}_{[t]}^{-1}
\\&+ 
\text{diag}\left(
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}} 
\mathbf{K}_{[t]}^\top 
\right)
\odot
\boldsymbol{\beta}_{[t]}
\\&+
\text{diag}\left(
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\left(
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)
\right)
\\&
-\boldsymbol{\gamma}_{[t]}^{-1} \odot
\text{diag}\left(
\left(
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\right)
\text{Diag}(\boldsymbol{\gamma}_{[t]})
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\right)
\odot \boldsymbol{\gamma}_{[t]}^{-1}
\\ &+
[0, 0, ..., \delta \boldsymbol{\gamma}_{[t]}^C]^\top
\end{aligned}$$

### 对于 `alpha_[t]` 的梯度

最后，有：

$$\begin{aligned}
\delta \log \boldsymbol{\alpha}_{[t]}
&=
\text{suffix\_cumsum}(\delta \log \mathbf{\gamma}_{[t]})
\end{aligned}$$

