# 阅读笔记：DeltaNet 的反向传播

### 数学预备知识

除 GLA 笔记中已补充的内容外，这里还涉及**逆矩阵的微分**：

$$\begin{aligned}
& \mathbf{A} = \mathbf{B}^{-1}
\Rightarrow 
d \mathbf{A} = - \mathbf{B}^{-1} (d \mathbf{B}) \mathbf{B}^{-1}
\\
\\
\Rightarrow &
dy = \text{Tr}\left((\delta \mathbf{A})^\top (d \mathbf{A})\right) 
= \text{Tr}\left(-(\delta \mathbf{A})^\top \mathbf{B}^{-1} (d \mathbf{B}) \mathbf{B}^{-1}\right) 
= \text{Tr}\left(- \mathbf{B}^{-1} (\delta \mathbf{A})^\top \mathbf{B}^{-1} (d \mathbf{B}) \right) 
\\
\\
\Rightarrow &
\delta \mathbf{B} = - \mathbf{B}^{-1 \top} (\delta \mathbf{A}) \mathbf{B}^{-1 \top} 
= - \mathbf{A}^\top (\delta \mathbf{A}) \mathbf{A}^\top 
\end{aligned}$$


以及一个有用的**转置公式**：

$$\begin{aligned}
\text{diag}\left(\mathbf{A} \left(\mathbf{B} \odot \mathbf{C}\right)^\top \right)
=
\text{diag}\left(\left(\mathbf{A} \odot \mathbf{C}\right) \mathbf{B}^\top \right)
\end{aligned}$$

### 回顾前向传播公式

$$\begin{aligned}
\mathbf{V}_{[t],new} &:= \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  \right) 
\\
\\
\mathbf{X}_{[t]} &= \mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right)
\\
\\
\mathbf{A}_{[t]} &= \mathbf{X}_{[t]}^{-1}
,\quad
\mathbf{T}_{[t]} 
= 
\mathbf{A}_{[t]} \text{Diag}(\boldsymbol{\beta}_{[t]})
\\
\\
\mathbf{W}_{[t]} &= \mathbf{T}_{[t]} \mathbf{K}_{[t]} 
,\quad
\mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]} 
\\
\\
\mathbf{S}_{[t]}^{C} 
&=
\mathbf{S}_{[t-1]}^{C}
+
\mathbf{V}_{[t],new}^\top\mathbf{K}_{[t]}
\\
\\
\mathbf{O}_{[t]}
&=
\mathbf{Q}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  
+
\left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M} \right)  \mathbf{V}_{[t],new}
\end{aligned}$$

### 对于 $\delta \mathbf{S}_{[t]}^C$


$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{S}_{[t]}^{C}}
&=
\delta \mathbf{S}_{[t]}^{C}
-
\delta \mathbf{S}_{[t]}^{C} \mathbf{K}_{[t]}^\top  \mathbf{W}_{[t]} 
\\
\\
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{O}_{[t]}}
&=
\delta \mathbf{O}_{[t]}^\top  \mathbf{Q}_{[t]}
-
\delta \mathbf{O}_{[t]}^\top
\left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M} \right)  
\mathbf{W}_{[t]}
\\
\\
\Rightarrow
\delta \mathbf{S}_{[t]}^{C}
&=
\delta \mathbf{S}_{[t+1]}^{C}
-
\delta \mathbf{S}_{[t+1]}^{C} \mathbf{K}_{[t+1]}^\top  \mathbf{W}_{[t+1]} 
+
\delta \mathbf{O}_{[t+1]}^\top  \mathbf{Q}_{[t+1]}
-
\delta \mathbf{O}_{[t+1]}^\top
\left( \mathbf{Q}_{[t+1]} \mathbf{K}_{[t+1]}^\top \odot \mathbf{M} \right)  
\mathbf{W}_{[t+1]}
\\
\\
&=
\delta \mathbf{S}_{[t+1]}^{C}
+
\delta \mathbf{O}_{[t+1]}^\top  \mathbf{Q}_{[t+1]}
-
\delta \mathbf{U}_{[t+1]}^\top  \mathbf{W}_{[t+1]} 
\end{aligned}$$

### 对于 $\delta \mathbf{Q}_{[t]}$

$$\begin{aligned}
\delta \mathbf{Q}_{[t]}
&=
\delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C
+
\left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],new}^\top \odot \mathbf{M}\right) \mathbf{K}_{[t]}
\end{aligned}$$

### 对于 $\delta \mathbf{U}_{[t]}$

$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\mathbf{K}_{[t]} \delta \mathbf{S}_{[t]}^{C \top}
+
\left( \mathbf{K}_{[t]} \mathbf{Q}_{[t]}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\end{aligned}$$

### 对于 $\delta \mathbf{W}_{[t]}$

$$\begin{aligned}
\delta \mathbf{W}_{[t]}
=
- 
\mathbf{K}_{[t]} \delta \mathbf{S}_{[t]}^{C \top} \mathbf{S}_{[t-1]}^{C}
-
\left( \mathbf{K}_{[t]} \mathbf{Q}_{[t]}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C}
=
- 
\delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\end{aligned}$$

### 对于 $\delta \mathbf{V}_{[t]}$

$$\begin{aligned}
\delta \mathbf{V}_{[t]}
=
\mathbf{T}_{[t]}^\top \delta \mathbf{U}_{[t]}
\end{aligned}$$

### 对于 $\delta \mathbf{K}_{[t]}$

这是最复杂的部分，因为 $\mathbf{K}_{[t]}$ 的梯度会从多条路径传递过来。

我们先整理几个相关的中间结果：

$$\begin{aligned}
\delta \mathbf{W}_{[t]}
&=
- 
\mathbf{K}_{[t]} \delta \mathbf{S}_{[t]}^{C \top} \mathbf{S}_{[t-1]}^{C}
-
\left( \mathbf{K}_{[t]} \mathbf{Q}_{[t]}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^{C}
= - \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\left.\delta \mathbf{T}_{[t]}\right|_{\text{from } \mathbf{U}_{[t]} \text{ and } \mathbf{W}_{[t]}}
&=
\delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top
+
\delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top
\\
\\
\left.\delta \mathbf{X}_{[t]}\right|_{\text{from } \mathbf{T}_{[t]}}
&=
- \mathbf{X}_{[t]}^{-\top} \delta \mathbf{T}_{[t]}  \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{X}_{[t]}^{-\top}
=
- \mathbf{X}_{[t]}^{-\top} \delta \mathbf{T}_{[t]}  \mathbf{T}_{[t]}^{\top}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{U}_{[t]} \text{ or } \mathbf{W}_{[t]} }
&=
\mathbf{V}_{[t],new} \delta \mathbf{S}_{[t]}^{C}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{ w/o } \mathbf{U}_{[t]} \text{ or } \mathbf{W}_{[t]} }
&=
\left(\mathbf{V}_{[t],new} \delta \mathbf{O}_{[t]}^\top \odot \mathbf{M}^\top \right)\mathbf{Q}_{[t]}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{W}_{[t]}  \text{ w/o } \mathbf{T}_{[t]} }
&=
\mathbf{T}_{[t]}^\top \delta \mathbf{W}_{[t]}
=
- \mathbf{T}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\left.\delta (\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top)\right|_{\text{from } \mathbf{T}_{[t]}}
&=
- \text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{X}_{[t]}^{-\top} 
\delta \mathbf{T}_{[t]}
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{X}_{[t]}^{-\top} \odot \mathbf{M}_{-1}
= - \mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\mathbf{T}_{[t]}^{\top} \odot \mathbf{M}_{-1}
\end{aligned}$$

由此，来自 $\mathbf{T}_{[t]}$ 的那部分梯度可以写成：

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{T}_{[t]}}
&=
- \left(\mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\mathbf{T}_{[t]}^{\top} \odot \mathbf{M}_{-1} \right) \mathbf{K}_{[t]}
-
\left(\mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\mathbf{T}_{[t]}^{\top} \odot \mathbf{M}_{-1} \right)^\top \mathbf{K}_{[t]}
\\&=
- \left(\mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\mathbf{T}_{[t]}^{\top} \odot \mathbf{M}_{-1} 
+
\mathbf{T}_{[t]} \delta \mathbf{T}_{[t]}^\top
\mathbf{T}_{[t]} \odot \mathbf{M}_{-1}^\top \right) \mathbf{K}_{[t]}
\\
\\
\left.\delta \mathbf{T}_{[t]}\right|_{\text{from } \mathbf{U}_{[t]} \text{ and } \mathbf{W}_{[t]}}
&=
\delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top
+
\delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top
=
\delta \mathbf{U}_{[t]} 
\left(\mathbf{V}_{[t]}^\top - \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}\mathbf{K}_{[t]}^\top \right)
\end{aligned}$$

把所有项汇总起来，最终得到：

$$\begin{aligned}
\delta \mathbf{K}_{[t]}
&=
\mathbf{V}_{[t],new}  \delta \mathbf{S}_{[t]}^{C}
+
\left(\mathbf{V}_{[t],new}  \delta \mathbf{O}_{[t]}^\top \odot \mathbf{M}^\top \right)\mathbf{Q}_{[t]}
-
\mathbf{T}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\ &- 
\left(\mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\mathbf{T}_{[t]}^{\top} \odot \mathbf{M}_{-1} 
+
\mathbf{T}_{[t]} \delta \mathbf{T}_{[t]} ^\top
\mathbf{T}_{[t]} \odot \mathbf{M}_{-1}^\top \right) \mathbf{K}_{[t]}
\end{aligned}$$
### 对于 $\delta \boldsymbol{\beta}_{[t]}$

$$\begin{aligned}
\delta \text{Diag}(\boldsymbol{\beta}_{[t]})
&=
\mathbf{X}_{[t]}^{-\top} \delta \mathbf{T}_{[t]}
-
\mathbf{X}_{[t]}^{-\top} \delta \mathbf{T}_{[t]} \mathbf{T}_{[t]}^\top \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}^\top  \right)
\\ &=
\text{Diag}(\boldsymbol{\beta}_{[t]})^{-1}\mathbf{T}_{[t]}^{\top} \delta \mathbf{T}_{[t]}
\left(\mathbf{I} - \mathbf{T}_{[t]}^\top \left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}^\top  \right) \right)
\\
\\
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}(\delta \text{Diag}(\boldsymbol{\beta}_{[t]}))
\end{aligned}$$

### $\delta \mathbf{K}_{[t]}, \delta \boldsymbol{\beta}_{[t]}$ 的另一种等价表达

令 $\mathbf{A} = \mathbf{X}^{-1}$，可以得到一种更适合代码实现的等价形式：


$$\begin{aligned}
\delta \mathbf{W}_{[t]}
&= - \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\\
\\
\left.\delta \mathbf{T}_{[t]}\right|_{\text{from } \mathbf{U}_{[t]} \text{ and } \mathbf{W}_{[t]}}
&=
\delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top
+
\delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top
\\
\\
\delta \mathbf{A}_{[t]}
&= \delta \mathbf{T}_{[t]}  \text{Diag}(\boldsymbol{\beta}_{[t]})^\top 
=
\delta \mathbf{U}_{[t]} (\mathbf{V}_{[t]} \text{Diag}(\boldsymbol{\beta}_{[t]}))^\top
+
\delta \mathbf{W}_{[t]} (\mathbf{K}_{[t]}\text{Diag}(\boldsymbol{\beta}_{[t]}))^\top 
\\
\\
\delta \mathbf{X}_{[t]}
&=
- \mathbf{A}_{[t]}^\top \delta \mathbf{A}_{[t]} \mathbf{A}_{[t]}^\top
, \quad
\mathbf{T}_{[t]}^\top \delta \mathbf{T}_{[t]} \mathbf{T}_{[t]}^\top
=
\text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{A}_{[t]}^\top \delta \mathbf{A}_{[t]} \mathbf{A}_{[t]}^\top
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/ } \mathbf{O}_{[t]}  \text{ w/o } \mathbf{U}_{[t]} \text{ or } \mathbf{W}_{[t]} }
&=
\mathbf{V}_{[t],new} \delta \mathbf{S}_{[t]}^{C}
+\left(\mathbf{V}_{[t],new} \delta \mathbf{O}_{[t]}^\top \odot \mathbf{M}^\top \right)\mathbf{Q}_{[t]}
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{W}_{[t]}  \text{ w/o } \mathbf{T}_{[t]} }
&=
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{A}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
\\
\\
\left.\delta (\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top)\right|_{\text{from } \mathbf{T}_{[t]}}
&=
- \text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{X}_{[t]}^{-\top} 
\delta \mathbf{T}_{[t]}
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{X}_{[t]}^{-\top} \odot \mathbf{M}_{-1}
= \text{Diag}(\boldsymbol{\beta}_{[t]}) (\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1})
\\
\\
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{T}_{[t]}}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})  \left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right) \mathbf{K}_{[t]}
+
\left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right)^\top  \text{Diag}(\boldsymbol{\beta}_{[t]})  \mathbf{K}_{[t]}
\\
\\
\delta \mathbf{K}_{[t]}
&=
\mathbf{V}_{[t],new}  \delta \mathbf{S}_{[t]}^{C}
+
\left(\mathbf{V}_{[t],new}  \delta \mathbf{O}_{[t]}^\top \odot \mathbf{M}^\top \right)\mathbf{Q}_{[t]}
+
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\mathbf{A}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
\\ &+ 
\text{Diag}(\boldsymbol{\beta}_{[t]})  \left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right) \mathbf{K}_{[t]}
+
\left(\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1} \right)^\top  \text{Diag}(\boldsymbol{\beta}_{[t]})  \mathbf{K}_{[t]}
\end{aligned}$$

以及 $\delta \boldsymbol{\beta}$ 的等价形式：


$$\begin{aligned}
\delta \text{Diag}(\boldsymbol{\beta}_{[t]})
&=
\mathbf{X}_{[t]}^{-\top} \delta \mathbf{T}_{[t]}
+
\delta \mathbf{X}_{[t]}
\left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}^\top  \right)
\\ &=
\mathbf{A}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top
+
\mathbf{A}_{[t]}^\top \delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top 
+
\delta \mathbf{X}_{[t]} 
\left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right)^\top
\\
\\
\delta \boldsymbol{\beta}_{[t]} 
&=
\text{diag}(\delta \text{Diag}(\boldsymbol{\beta}_{[t]}))
\\ &=
\text{diag}\left(\mathbf{A}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top\right)
+
\text{diag}\left(\mathbf{A}_{[t]}^\top \delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top\right)
+
\text{diag}\left(\delta \mathbf{X}_{[t]} 
\left( \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}_{-1}  \right)^\top\right)
\\&=
\text{diag}\left(\mathbf{A}_{[t]}^\top \delta \mathbf{U}_{[t]} \mathbf{V}_{[t]}^\top\right)
+
\text{diag}\left(\mathbf{A}_{[t]}^\top \delta \mathbf{W}_{[t]} \mathbf{K}_{[t]}^\top\right)
+
\text{diag}\left((\delta \mathbf{X}_{[t]} \odot \mathbf{M}_{-1})
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top\right)
\end{aligned}$$

