# Gated Linear Attention Transformers with Hardware-Efficient Training

【***评论***：请注意，所有来自我自己的推导都可能有错，之后看代码需要交叉验证。】

## 动机
为linear attention构造硬件高效的算法


## 标记
1. 采用 $\mathbf{S, Q}$ 等表示矩阵
2. 采用$\mathbf{q}_t, \mathbf{k}_t$等表示**列向量** （[d, 1]的版本） ，矩阵则是[L, d]的版本，所以会有额外转置。【***评论***：我们这里与原文不同，原文均为行向量。因此***所有公式都是重写版本***，如有错误请指正】
3. 采用$W_t$等表示可学习参数
4. 采用$\mathbf{q}_t$表示$\mathbf{Q}$的第$t$行


## 背景

### Self-Attention

$$
\begin{aligned}
\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t = W_Q \boldsymbol{x}_t , W_K \boldsymbol{x}_t , W_V \boldsymbol{x}_t  
\\
\\
\boldsymbol{o}_t = \frac{\sum_{i=1}^t \boldsymbol{v}_i \exp(\boldsymbol{k}_i^\top \boldsymbol{q}_t )}{\sum_{i=1}^t \exp(\boldsymbol{k}_i^\top \boldsymbol{q}_t )}
\Leftrightarrow
\mathbf{O} = \text{softmax}\left(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}\right) \mathbf{V}
\\
\\
\end{aligned}
$$

### Linear Attention

$$
\begin{aligned}
\mathbf{o}_t = \frac{\sum_{i=1}^t  \mathbf{v}_i \phi(\mathbf{k}_i)^\top \phi(\mathbf{q}_t)}
{\sum_{i=1}^t \phi(\mathbf{k}_i)^\top \phi(\mathbf{q}_t)} 
\\
\\
\mathbf{S}_t = \sum_{i=1}^t \boldsymbol{v}_i \phi(\boldsymbol{k}_i)^\top  \in \mathbb{R}^{d \times d}
, \quad 
\boldsymbol{z}_t = \sum_{i=1}^t \phi(\boldsymbol{k}_i) \in \mathbb{R}^{d \times 1}
\\
\\
\mathbf{S}_t = \mathbf{S}_{t-1} + \boldsymbol{v}_t \phi(\boldsymbol{k}_t)^\top 
, \quad 
\boldsymbol{z}_t = \boldsymbol{z}_{t-1} + \phi(\boldsymbol{k}_t)
, \quad 
\boldsymbol{o}_t = \frac{\mathbf{S}_t \phi(\boldsymbol{q}_t)}{\boldsymbol{z}_t^\top\phi(\boldsymbol{q}_t)}.
\end{aligned}
$$

之前的工作发现不需要非线性和norm也很好，也就是说

$$
\mathbf{S}_t = \mathbf{S}_{t-1} +  \boldsymbol{v}_t \boldsymbol{k}_t^\top
, \quad 
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
$$

### Linear Attention + Chunkwise

将$\mathbf{X}$切成不相交的一些chunk，每个chunk的长度为$C$。定义

$$
\begin{aligned}
\square_{[t]}^i = \square_{tC+i}
,\quad
\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d}
\quad \text{for } 
\square \in \{ \mathbf{Q, K, V, O} \}
\end{aligned}
$$

则有

$$\begin{aligned}
\mathbf{S}_{[t]}^{C} = \mathbf{S}_{[t-1]}^{C} + \sum_{i=tC+1}^{tC+C} \boldsymbol{v}_i \boldsymbol{k}_i^\top  
\quad \in \mathbb{R}^{d \times d}
\\
\\
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]} \mathbf{S}_{[t]}  + \left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}\right) \mathbf{V}_{[t]}  
\end{aligned}$$


## Flash Linear Attention

### 设计方向

1. 需要使用足够多的SM
2. 考虑batch-size=1，因此需要在时间维度上并行
3. 使用tensor-core
4. 采用层级显存的设计，最佳利用SRAM和HBM
5. 采用块内并行，块间串行

### 算法

FLA做了两种chunkwise的算法

![](assets/Pasted%20image%2020260324141311.png)
![](assets/Pasted%20image%2020260324141321.png)

【***评论***：纯串行（只在每个块内并行，块间串行）看起来似乎还可以？】

## Gated Linear Attention

### 递推模式

一般形式

$$\begin{aligned}
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
,\quad
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
\end{aligned}$$

对于GLA

$$\begin{aligned}
\mathbf{S}_t = ( \mathbf{1} \boldsymbol{\alpha}_t^\top) \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
=
\mathbf{S}_{t-1}\text{Diag}(\boldsymbol{\alpha}_t)  + \boldsymbol{v}_t \boldsymbol{k}_t^\top
\end{aligned}$$

![](<assets/Pasted%20image%2020260324141634.png>)

点评（来自原文）：
1. GLA最核心的设计在于Gate的参数化需要平衡parameter-efficiency, state size, and training efficiency
2. mamba中Gate来自于可学习矩阵$\mathbf{A}$与数据相关的$\boldsymbol{\alpha}_t$的组合，也就是说Gate是一个full rank的矩阵。然而，这个设计 **prevents the use of tensor cores because it cannot be reformulated into a matrix-multiply format**。为此mamba设计了一种prefix sum的算法充分利用SRAM。然而，**Due to limited SRAM capacity, this approach cannot scale to larger hidden states, which, as we will show in our experiments, results in suboptimal performance on recall-intensive tasks.**   

### 分块递推模式

定义

$$\begin{aligned}
\boldsymbol{\gamma}_{[t]}^r = \prod_{i=tC+1}^{tC+r} \boldsymbol{\alpha}_i \in \mathbb{R}^{d \times 1}
, \quad
\\
\\
\mathbf{H}_{[t]}^{r} 
= \sum_{i=1}^{r} 
(\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top})  
\text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) 
\in \mathbb{R}^{d \times d}
\\
\\
\mathbf{\Gamma}_{[t]} = [ \boldsymbol{\gamma}_{[t]}^{1}, \boldsymbol{\gamma}_{[t]}^{2}, \dots, \boldsymbol{\gamma}_{[t]}^{C} ]^\top \in \mathbb{R}^{C \times d}
\\
\\
\overleftarrow{\boldsymbol{q}_{[t]}^{i}} = \boldsymbol{q}_{[t]}^{i} \odot \boldsymbol{\gamma}_{[t]}^{i}
, \quad
\overrightarrow{\boldsymbol{k}_{[t]}^{i}} = \frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}}
\\
\\
\overleftarrow{\mathbf{Q}_{[t]}} = \mathbf{Q}_{[t]} \odot \mathbf{\Gamma}_{[t]}
, \quad
\overrightarrow{\mathbf{K}_{[t]}} = \mathbf{Q}_{[t]} \oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$


那么有

$$\begin{aligned}
\mathbf{H}_{[0]}^{r} = \mathbf{S}_{r}
\\
\\
\mathbf{S}_{[t]}^{r} = \mathbf{S}_{[t-1]}^{C} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) 
+ \mathbf{H}_{[t]}^{r}
\\
\\
\mathbf{H}_{[t]}^{r} 
= \sum_{i=1}^{r} (\boldsymbol{v}_{[t]}^{i} \boldsymbol{k}_{[t]}^{i\top}) 
\text{Diag}(\frac{\boldsymbol{\gamma}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{i}}) 
=  \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i}\left(\frac{\boldsymbol{k}_{[t]}^{i}}{\boldsymbol{\gamma}_{[t]}^{i}}\right)^{\top} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r})
\end{aligned}$$

进一步

$$\begin{aligned}
\boldsymbol{o}_{[t]}^{r} = \mathbf{S}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r} =  \mathbf{S}_{[t-1]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) \boldsymbol{q}_{[t]}^{r} + \mathbf{H}_{[t]}^{r} \boldsymbol{q}_{[t]}^{r}
\\ 
\\
\Rightarrow \boldsymbol{o}_{[t]}^{r} = \mathbf{S}_{[t-1]}^{C} \overleftarrow{\boldsymbol{q}_{[t]}^{r}} + \sum_{i=1}^{r} \boldsymbol{v}_{[t]}^{i} \left(\overrightarrow{\boldsymbol{k}_{[t]}^{i}}\right)^{\top} \overleftarrow{\boldsymbol{q}_{[t]}^{r}}
\end{aligned}$$

从而有矩阵形式

$$\begin{aligned}
\mathbf{O}_{[t]} = \overleftarrow{\boldsymbol{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C \top} 
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]}
\end{aligned}$$

其中，$\mathbf{S}_{[t]}$可根据下式提前算出

$$\begin{aligned}
\mathbf{S}_{[t]}^{C} = \mathbf{S}_{[t-1]}^{C} 
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
+ \mathbf{H}_{[t]}^{C}
=
\left(\mathbf{S}_{[t-1]}^{C} 
+ \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
\end{aligned}$$

特别的（为了给后面的梯度推导做准备），如果

$\boldsymbol{z}_{[t]}^{r} = \mathbf{S}_{[t]}^{r \top} \boldsymbol{x}_{[t]}^{r}$，那么有
$$\begin{aligned}
\boldsymbol{z}_{[t]}^{r} = \text{Diag}(\boldsymbol{\gamma}_{[t]}^{r})  \mathbf{S}_{[t-1]}^{C \top} \boldsymbol{x}_{[t]}^{r} + 
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{r}) \sum_{i=1}^{r} \left(\overrightarrow{\boldsymbol{k}_{[t]}^{i}}\right)
\boldsymbol{v}_{[t]}^{i \top} \boldsymbol{x}_{[t]}^{r}
\\
\\
\Rightarrow
\overrightarrow{\boldsymbol{z}_{[t]}^{r}} := \frac{\boldsymbol{z}_{[t]}^{r}}{\boldsymbol{\gamma}_{[t]}^{r}} = \mathbf{S}_{[t-1]}^{C \top} \boldsymbol{x}_{[t]}^{r} + 
\sum_{i=1}^{r} \left(\overrightarrow{\boldsymbol{k}_{[t]}^{i}}\right)
\boldsymbol{v}_{[t]}^{i \top} \boldsymbol{x}_{[t]}^{r}
\\
\\
\Rightarrow
\overrightarrow{\mathbf{Z}_{[t]}} = \mathbf{X}_{[t]} \mathbf{S}_{[t-1]}^{C} 
+
\left(\mathbf{X}_{[t]} \mathbf{V}_{[t]}^\top  \odot \mathbf{M} \right) \overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$


【***评论***：上述推导只是一个chunkwise的示范，实际代码中可能需要将$\text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})$合并入$\overrightarrow{\mathbf{K}_{[t]}}$。事实上，GLA的伪代码就是合并的。】


### 分块递推模式（续，Secondary-level chunking）

还有一点需要注意的是，对于较大的chunk-size，可能会因为$\overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[t]}} \right)^\top$ 中衰减过大而丢失精度。GLA提出的一个解决方案是将chunk分成subchunk，其中跨度大的不跟在log域计算衰减。

![500](<assets/Pasted%20image%2020260324161324.png>)

我们引入变量$\mathbf{P}_{[t][\tau]}$, 并假定sub-chunk的长度为$T$:
$$
\mathbf{P}_{[t][\tau]} = \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top \odot \mathbf{M}_{[t][\tau]}
$$

- 情况1，粉色部分sub-chunk。该部分的精度需要足够高，因此采用逐元素全精度计算。
- 情况2，橙色部分sub-chunk。该部分使用半精度矩阵运算，每个sub-chunk内单独算。
- 情况3，灰色部分sub-chunk。只有在并行模式下才需要算，在chunkwise模式下不需要计算。【***评论***：如果要算，计算方式和橙色部分相同】

对于情况1，我们有

$$\begin{aligned}
(\mathbf{P}_{[t][\tau]})_{i, j} 
=
\sum_{d} (\boldsymbol{q}_{[t]}^{i})_{d} ~(\boldsymbol{k}_{[\tau]}^{j})_{d} 
~ \exp(\log \boldsymbol{\gamma}_{[t]}^{i}  - \log \boldsymbol{\gamma}_{[\tau]}^{j}  ) 
，\quad t=\tau, i>j
\end{aligned}$$

对于情况2.我们有（请注意，这不是对角块）

$$\begin{aligned}
\mathbf{P}_{[t][\tau]} = \overleftarrow{\boldsymbol{Q}_{[t]}} \left(\overrightarrow{\boldsymbol{K}_{[\tau]}} \right)^\top
,\quad t \ne \tau
\\ 
\Rightarrow
\mathbf{P}_{[t][\tau]} = \left(\boldsymbol{Q}_{[t]} \odot \exp(\log \boldsymbol{\gamma}_{[t]}^{1:T}) \right) \left(\boldsymbol{K}_{[\tau]} \odot \exp(-\log \boldsymbol{\gamma}_{[\tau]}^{1:T}) \right)^\top
,\quad t > \tau
\end{aligned}$$


## 【***评论***：Gated Linear Attention Backward】

### 递推模式

一般形式前传

$$\begin{aligned}
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top
,\quad
\boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t
\end{aligned}$$

一般形式反传

$$\begin{aligned}
\delta \boldsymbol{o}_t = \frac{\partial L}{\partial \boldsymbol{o}_t}
\\
\\
\delta \mathbf{G}_t = \frac{\partial L}{\partial \mathbf{G}_t} = \delta \mathbf{S}_t \odot \mathbf{S}_{t-1}
\\
\\
\delta \mathbf{S}_t = \mathbf{G}_{t+1} \odot \delta \mathbf{S}_{t+1} + \delta \boldsymbol{o}_t \boldsymbol{q}_t^\top
\\
\\
\delta \boldsymbol{q}_t = \frac{\partial L}{\partial \boldsymbol{q}_t} = \mathbf{S}_t^\top \delta \boldsymbol{o}_t
\\
\\
\delta \boldsymbol{v}_t = \frac{\partial L}{\partial \boldsymbol{v}_t} = \delta  \mathbf{S}_t \boldsymbol{k}_t
\\
\\
\delta \boldsymbol{k}_t = \frac{\partial L}{\partial \boldsymbol{k}_t} = \delta  \mathbf{S}_t^\top \boldsymbol{v}_t
\end{aligned}$$

GLA反传

$$\begin{aligned}
\delta \boldsymbol{\alpha}_t = \delta \mathbf{G}_t^\top \mathbf{1}
\end{aligned}$$


### 矩阵求导补充知识

1. 定义

$$\begin{aligned}
\mathbf{A} \in \mathbb{R}^{m \times n}
, \quad 
\mathbf{B} \in \mathbb{R}^{n \times k}
, \quad 
\mathbf{C} = \mathbf{A} \mathbf{B}
, \quad 
y = f(\mathbf{C})
, \quad 
\delta \mathbf{C} := \frac{\partial y}{\partial \mathbf{C}}
\end{aligned}$$

2. 矩阵迹的性质

$$\begin{aligned}
\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)
\\
\\
\text{Tr}(A^\top (B \odot C))=Tr((A \odot B)^\top C)=Tr((A \odot C)^\top B)
\end{aligned}$$

3. 常用微分公式

$$\begin{aligned}
d(\mathbf{A}\mathbf{B}) = (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B})
\\
\\
d(\mathbf{A} \odot \mathbf{B}) = (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B})
\end{aligned}$$

4. 矩阵梯度

$$\begin{aligned}
dy 
= \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{C}})^\top  d \mathbf{C}\right) 
= \text{Tr}\left( (\delta \mathbf{C})^\top  (d \mathbf{C})\right) 
= \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} (d \mathbf{B}) + (d \mathbf{A}) \mathbf{B}) \right) 
\\
\\
\text{while}\quad dy 
= \text{Tr}\left( (\frac{\partial y}{\partial \mathbf{B}})^\top  (d \mathbf{B})\right)
\quad \text{so we have}\quad
\delta \mathbf{B} =  \mathbf{A}^\top \delta \mathbf{C}
, \quad 
\delta \mathbf{A} = \delta \mathbf{C} \mathbf{B}^\top
\end{aligned}$$

5. 矩阵梯度Hadamard积 

$$\begin{aligned}
dy 
= \text{Tr}\left( (\delta \mathbf{D})^\top  (d \mathbf{D})\right) 
= \text{Tr}\left( (\delta \mathbf{C})^\top (\mathbf{A} \odot (d \mathbf{B}) + (d \mathbf{A}) \odot \mathbf{B}) \right) 
\\
\\
= \text{Tr}\left( (\delta \mathbf{C} \odot \mathbf{A})^\top (d \mathbf{B}) + (\delta \mathbf{C} \odot \mathbf{B})^\top (d \mathbf{A})  \right) 
\\
\\
\text{while}\quad dy 
= \text{Tr}\left( (\delta \mathbf{B})^\top  (d \mathbf{B})\right)
\quad \text{so we have}\quad
\delta \mathbf{B} =  \delta \mathbf{C} \odot \mathbf{A} 
, \quad 
\delta \mathbf{A} = \delta \mathbf{C} \odot \mathbf{B}
\end{aligned}$$


### 分块递推模式 
【***评论***：我感觉与其从递推模式的反传开始推导，不如从分块递推的前传开始更简单】

回顾之前的重要结论：

$$\begin{aligned}
\mathbf{O}_{[t]} = \overleftarrow{\boldsymbol{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C \top} 
+
\left( \overleftarrow{\mathbf{Q}_{[t]}} \left(\overrightarrow{\mathbf{K}_{[t]}} \right)^\top \odot \mathbf{M} \right) \mathbf{V}_{[t]}
\\
\\
\mathbf{S}_{[t]}^{C}
=
\left(\mathbf{S}_{[t-1]}^{C} 
+ \mathbf{V}_{[t]}^\top \overrightarrow{\boldsymbol{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) 
\end{aligned}$$

则有对于$\delta \mathbf{S}_{[t]}^{C}$

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{O}_{[t]}}
=
\delta \mathbf{O}_{[t]}^{\top} \overleftarrow{\mathbf{Q}}_{[t]}
,\quad
\left.\delta \mathbf{S}_{[t-1]}^{C}\right|_{\text {from } \mathbf{S}_{[t]}^C}
=
\delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})  
\\
\\
\Rightarrow
\delta \mathbf{S}_{[t]}^{C} = \delta \mathbf{S}_{[t+1]}^C \text{Diag}(\boldsymbol{\gamma}_{[t+1]}^{C})   + \delta \mathbf{O}_{[t+1]}^{\top} \overleftarrow{\mathbf{Q}}_{[t+1]}
\end{aligned}$$

对于$\delta \mathbf{V}_{[t]}$

$$\begin{aligned}
\left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{O}_{[t]}}
=
 \left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
,\quad
\left.\delta \mathbf{V}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
=
\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top}
\\
\\
\Rightarrow
\delta \mathbf{V}_{[t]} 
=  
\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C}) \delta \mathbf{S}_{[t]}^{C \top}
+ 
\left(\left(\overrightarrow{\mathbf{K}_{[t]}} \right) \overleftarrow{\mathbf{Q}_{[t]}}^\top \odot \mathbf{M}^\top \right) \delta \mathbf{O}_{[t]}
\end{aligned}$$

对于$\delta \overrightarrow{\mathbf{K}_{[t]}}$

$$\begin{aligned}
\left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}}
=
\left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}}
,\quad
\left.\delta\overrightarrow{\mathbf{K}_{[t]}}\right|_{\text {from } \mathbf{S}_{[t]}^C}
=
\mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})
\\
\\
\Rightarrow
\delta\overrightarrow{\mathbf{K}_{[t]}} 
= 
\mathbf{V}_{[t]} \delta \mathbf{S}_{[t]}^C \text{Diag}(\boldsymbol{\gamma}_{[t]}^{C})
+ 
\left( \mathbf{V}_{[t]} \left(\delta \mathbf{O}_{[t]} \right)^\top \odot \mathbf{M}^\top \right) \overleftarrow{\mathbf{Q}_{[t]}}
\end{aligned}$$

对于$\delta \overleftarrow{\mathbf{Q}_{[t]}}$

$$\begin{aligned}
\delta \overleftarrow{\mathbf{Q}_{[t]}}
=
\left.\delta \overleftarrow{\mathbf{Q}_{[t]}}\right|_{\text {from } \mathbf{O}_{[t]}}
=
\delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C 
+
\left( \delta\mathbf{O}_{[t]}\mathbf{V}_{[t]}^\top \odot  \mathbf{M} \right) \overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

对于$\delta \mathbf{K}_{[t]}$

$$\begin{aligned}
\delta \mathbf{K}_{[t]}
=
\delta \overrightarrow{\mathbf{K}_{[t]}} \oslash  \mathbf{\Gamma}_{[t]}
\end{aligned}$$

对于$\delta \mathbf{Q}_{[t]}$

$$\begin{aligned}
\delta \mathbf{Q}_{[t]}
=
\delta \overleftarrow{\mathbf{Q}_{[t]}} \odot  \mathbf{\Gamma}_{[t]}
\end{aligned}$$

对于$\delta \mathbf{\Gamma}_{[t]}$

$$\begin{aligned}
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
=
\left[0,0,..., \text{diag}\left(\left(\mathbf{S}_{[t-1]}^{C \top} 
+ \overrightarrow{\boldsymbol{K}_{[t]}}^\top \mathbf{V}_{[t]}  \right)  \delta \mathbf{S}_{[t]}^{C}\right) \right]^\top
\\
\\
\delta \mathbf{\Gamma}_{[t]}
=
\delta \overleftarrow{\mathbf{Q}_{[t]}} \odot \mathbf{Q}_{[t]}
-
\delta \overrightarrow{\mathbf{K}_{[t]}} \odot  \mathbf{K}_{[t]} \oslash (\mathbf{\Gamma}_{[t]} \odot \mathbf{\Gamma}_{[t]})   
+ 
\left.\delta \mathbf{\Gamma}_{[t]}\right|_{\text {from } \mathbf{S}_{[t]}^C}
\end{aligned}$$

对于$\delta \mathbf{\alpha}_{[t]}$

$$\begin{aligned}
\mathbf{\Gamma}_{[t]} = [ \prod_{i=tC+1}^{tC+1} \boldsymbol{\alpha}_i, \prod_{i=tC+1}^{tC+2} \boldsymbol{\alpha}_i,...\prod_{i=tC+1}^{tC+C} \boldsymbol{\alpha}_i]^\top \in \mathbb{R}^{C \times d}
\\
\\
\delta \boldsymbol{\alpha}_r 
= 
\sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot (\prod_{i=tC+1}^{tC+j} \boldsymbol{\alpha}_i \oslash \boldsymbol{\alpha}_r)
=
\left(\sum_{j \ge r} \delta \mathbf{\Gamma}_{j,:} \odot \mathbf{\Gamma}_{j,:} \right) \oslash \boldsymbol{\alpha}_r
\\
\\
\delta \mathbf{A}_{[t]} 
= [\delta \boldsymbol{\alpha}_{[t]}^1, \delta \boldsymbol{\alpha}_{[t]}^2,...\delta \boldsymbol{\alpha}_{[t]}^C]^\top 
=
\text{suffix\_sum}_{row}(\delta \mathbf{\Gamma} \odot \mathbf{\Gamma}) \oslash \mathbf{A}_{[t]} 
\end{aligned}$$


## 网络结构

### Token-Mixing 部分

$$\begin{aligned} 
\boldsymbol{\alpha}_{t} = \sigma\left(\left(\mathbf{W}_{\alpha}^{1} \mathbf{W}_{\alpha}^{2} \boldsymbol{x}_{t} + \boldsymbol{b}_{\alpha}\right)\right)^{\frac{1}{\tau}} \in \mathbb{R}^{d_{k} \times 1}
\\
\\
\mathbf{S}_{t}^{h} = \left( \boldsymbol{\alpha}_{t}^{h}\mathbf{1} \right) \odot \mathbf{S}_{t-1}^{h} + \boldsymbol{v}_{t}^{h}\boldsymbol{k}_{t}^{h \top}  \in \mathbb{R}^{d_{v}^{\prime} \times d_{k}^{\prime}}
\\
\\
\boldsymbol{o}_{t}^{h} = \mathbf{S}_{t}^{h} \boldsymbol{q}_{t}^{h}  \in \mathbb{R}^{d_{v}^{\prime} \times 1} 
\\
\\
\boldsymbol{o}_{t}^{\prime} = \operatorname{concat}\left(\operatorname{LN}\left(\boldsymbol{o}_{t}^{1}\right), \dots, \operatorname{LN}\left(\boldsymbol{o}_{t}^{H}\right)\right) \in \mathbb{R}^{d_{v} \times 1}
\\
\\
\boldsymbol{r}_{t} = \operatorname{Swish}\left(\mathbf{W}_{r} \boldsymbol{x}_{t} + \boldsymbol{b}_{r}\right) \in \mathbb{R}^{d_{v} \times 1}
\\
\\
\boldsymbol{y}_{t} = \mathbf{W}_{O} \left(\boldsymbol{r}_{t} \odot \boldsymbol{o}_{t}^{\prime}\right)  \in \mathbb{R}^{d \times 1}
\end{aligned}$$


其中 

$$\begin{aligned} 
\mathbf{W}_{\alpha}^{1} \in \mathbb{R}^{d \times 16}
,\quad 
\mathbf{W}_{\alpha}^{2} \in \mathbb{R}^{16 \times d_{k}}
, \quad
\tau = 16
, \quad
d_{k} = \frac{d}{2}
, \quad
d_{v} = d
\\
\\
(\mathbf{W}_{Q}, \mathbf{W}_{K}, \mathbf{W}_{V}, \mathbf{W}_{O}, \mathbf{W}_{r})
\in \text{Full Rank}
\end{aligned}$$


### Channel-Mixing 部分


$$\begin{aligned} 
\operatorname{SwiGLU}(\mathbf{Z}) = \left(\operatorname{Swish}(\mathbf{Z} \mathbf{W}_1) \odot \mathbf{Z} \mathbf{W}_2\right) \mathbf{W}_3
\end{aligned}$$


### 单层模式

$$\begin{aligned} 
\mathbf{Y}^{(l)} = \operatorname{GLA}\left(\operatorname{LN}\left(\mathbf{X}^{(l)}\right)\right) + \mathbf{X}^{(l)} 
\\
\\
\mathbf{X}^{(l+1)} = \operatorname{SwiGLU}\left(\operatorname{LN}\left(\mathbf{Y}^{(l)}\right)\right) + \mathbf{X}^{(l)}
\end{aligned}$$

最终GLA层大概占用$4d^{2}$参数，与普通attention对齐。


## 实验

1. 数据集：SlimPajama dataset； Tokenizer: Mistral tokenizer; 采用了100B token 子集。
2. 对比方法：Transformer++是含rope、swiglu、rmsnorm的llama；RetNet中的FFN层被替换为Swiglu。
3. 模型有340M和1.3B，采用adamw训练。340M模型训练了15BT数据，batch-size=0.5MT, warmup=0.5BT。1.3B模型训了100BT数据，batch-size=2MT, warmup=1BT。lr=3e-5, weight-decay=0.01 grad-clip=1.0。
4. 模型测试采用lm-eval。

![](assets/Pasted%20image%2020260324184450.png)

5. Recall任务：该任务通常被认为是linear attention做的比较差的任务，也就是让模型回想之前的精确信息。
![320](assets/Pasted%20image%2020260324184502.png) ![320](assets/Pasted%20image%2020260324184508.png)

6. 采用两种训练模式 a. 8K长度训练 b. 2K长度一个segment 总共12个segment 训练24k长度。但是每个segment间梯度不反传。测试分段的ppl。

![](assets/Pasted%20image%2020260324184652.png)


7. GLA的ablation
![320](assets/Pasted%20image%2020260324184702.png)![320](assets/Pasted%20image%2020260324184707.png)

