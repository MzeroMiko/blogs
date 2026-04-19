# Reading Notes: Backward Pass for Kimi Delta Attention

### Revisiting the Forward Equations

We first recall the key forward equations:

$$\begin{aligned}
\mathbf{V}_{[t],new} &:= \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t-1]}^{C \top}  \right) 
\\
\\
\widetilde{\mathbf{X}}_{[t]} &= \mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
,\quad
\widetilde{\mathbf{A}}_{[t]} 
= \widetilde{\mathbf{X}}_{[t]}^{-1}
\\
\\
\mathbf{W}_{[t]} &= 
\widetilde{\mathbf{A}}_{[t]} 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\overleftarrow{\mathbf{K}_{[t]}}  
,\quad
\mathbf{U}_{[t]} = 
\widetilde{\mathbf{A}}_{[t]} 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\mathbf{V}_{[t]} 
\\
\\
\mathbf{S}_{[t]}^{C} 
&=
\mathbf{S}_{[t-1]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+ \mathbf{V}_{[t],new}^\top
\overrightarrow{\mathbf{K}_{[t]}}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\\
\\
\mathbf{O}_{[t]}
&=
\overleftarrow{\mathbf{Q}_{[t]}}
\mathbf{S}_{[t-1]}^{C \top}  
+ \left( 
\overleftarrow{\mathbf{Q}_{[t]}}
\overrightarrow{\mathbf{K}_{[t]}}^\top
\odot \mathbf{M} 
\right)  
\mathbf{V}_{[t],new}
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



### For $\delta \mathbf{U}_{[t]}$

The gradient with respect to $\mathbf{U}_{[t]}$ is:

$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\delta \mathbf{V}_{[t],new}
=
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
\end{aligned}$$


### For $\delta \mathbf{W}_{[t]}$

From the definition of $\mathbf{V}_{[t],new}$, we further obtain:

$$\begin{aligned}
\delta \mathbf{W}_{[t]}
= - \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\end{aligned}$$


### For $\delta \mathbf{S}_{[t]}^C$

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{V}_{[t],new}}
&=
-\delta \mathbf{U}_{[t]}^\top
\mathbf{W}_{[t]}
\\
\\
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{S}_{[t]}^{C} \text{w/} \mathbf{O}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
&=
\delta \mathbf{S}_{[t]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+ \delta \mathbf{O}_{[t]}^\top  
\overleftarrow{\mathbf{Q}_{[t]}}
\end{aligned}$$

Combining the two terms, we obtain:

$$\begin{aligned}
\delta \mathbf{S}_{[t]}^{C}
&=
\delta \mathbf{S}_{[t+1]}^{C}
\text{Diag}(\boldsymbol{\gamma}_{[t+1]}^C)
+ \delta \mathbf{O}_{[t+1]}^\top  
\overleftarrow{\mathbf{Q}_{[t+1]}}
-\delta \mathbf{U}_{[t+1]}^\top
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
\mathbf{V}_{[t],new}^\top \odot \mathbf{M}\right) \overrightarrow{\mathbf{K}_{[t]}}
\\
\\
\delta \mathbf{Q}_{[t]}
&=
\delta \overleftarrow{\mathbf{Q}_{[t]}}
\odot \mathbf{\Gamma}_{[t]}
\end{aligned}$$

### For $\delta \mathbf{V}_{[t]}$

$$\begin{aligned}
\delta \mathbf{V}_{[t]}
=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{U}_{[t]}
\end{aligned}$$

### For $\delta \widetilde{\mathbf{A}}_{[t]}$ $\delta \widetilde{\mathbf{X}}_{[t]}$

First, for `A_[t]`, we have:

$$\begin{aligned}
\delta \widetilde{\mathbf{A}}_{[t]} 
&=
\delta \mathbf{W}_{[t]}
\overleftarrow{\mathbf{K}_{[t]}}^\top
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
+
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\end{aligned}$$

Then, using the differential formula for the matrix inverse, we obtain:

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

### For $\delta \mathbf{K}_{[t]}$

There are mainly two `K` in the forward process, `K_left` and `K_right`. We derive the gradients respectively.

For `K_left`, we have:


$$\begin{aligned}
\left.\delta \overleftarrow{\mathbf{K}_{[t]}}\right|_{\text{from } \mathbf{W}_{[t]} \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
\\
\\
\left.\delta \overleftarrow{\mathbf{K}_{[t]}}\right|_{\text{from } \widetilde{\mathbf{A}}_{[t]} }
&=
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

so we get

$$\begin{aligned}
\delta \overleftarrow{\mathbf{K}_{[t]}}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
+
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$


For `K_right`, we have:



$$\begin{aligned}
\left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text{from } \mathbf{S}_{[t]}^C \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
&=
\mathbf{V}_{[t],new}
\delta \mathbf{S}_{[t]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\\
\\
\left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text{from } \mathbf{O}_{[t]}^C \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
&=
\left(
\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],new}^\top
\odot \mathbf{M}
\right)^\top
\overleftarrow{\mathbf{Q}_{[t]}}
\\
\\
\left.\delta \overrightarrow{\mathbf{K}_{[t]}}\right|_{\text{from } \widetilde{\mathbf{A}}_{[t]} }
&=
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)^\top
\overleftarrow{\mathbf{K}_{[t]}}
\end{aligned}$$


so we get

$$\begin{aligned}
\delta \overrightarrow{\mathbf{K}_{[t]}}
&=
\mathbf{V}_{[t],new}
\delta \mathbf{S}_{[t]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
+
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)^\top
\overleftarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

finally we merge those two path

$$\begin{aligned}
\delta \mathbf{K}_{[t]}
&=
\delta \overleftarrow{\mathbf{K}_{[t]}}
\odot \mathbf{\Gamma}_{[t]}
+
\delta \overrightarrow{\mathbf{K}_{[t]}}
\oslash \mathbf{\Gamma}_{[t]}
\\&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{W}_{[t]}
\odot \mathbf{\Gamma}_{[t]}
+
\mathbf{V}_{[t],new}
\delta \mathbf{S}_{[t]}^C
\text{Diag}(\boldsymbol{\gamma}_{[t]}^C)
\oslash \mathbf{\Gamma}_{[t]}
\\&+
\left(
\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],new}^\top
\odot \mathbf{M}
\right)^\top
\overleftarrow{\mathbf{Q}_{[t]}}
\oslash \mathbf{\Gamma}_{[t]}
\\&+
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)
\overrightarrow{\mathbf{K}_{[t]}}
\odot \mathbf{\Gamma}_{[t]}
+
\left(
\text{Diag}(\boldsymbol{\beta}_{[t]})
\delta \widetilde{\mathbf{X}}_{[t]}
\odot \mathbf{M}_{-1}
\right)^\top
\overleftarrow{\mathbf{K}_{[t]}}
\oslash \mathbf{\Gamma}_{[t]}
\end{aligned}$$

### For $\delta \boldsymbol{\beta}_{[t]}$

Next, the gradient with respect to $\boldsymbol{\beta}_{[t]}$ can be written as:

$$\begin{aligned}
\delta \text{Diag}(\boldsymbol{\beta}_{[t]})
&=
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \mathbf{W}_{[t]}
\overleftarrow{\mathbf{K}_{[t]}}^\top
+
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \mathbf{U}_{[t]}
\mathbf{V}_{[t]}^\top
+
\delta \widetilde{\mathbf{X}}_{[t]}
\left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)^\top
\\&=
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \widetilde{\mathbf{A}}_{[t]}
\text{Diag}(\boldsymbol{\beta}_{[t]})^{-1}
+
\delta \widetilde{\mathbf{X}}_{[t]}
\left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)^\top
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
\end{aligned}$$

### For $\delta \mathbf{\Gamma}_{[t]}$


$$\begin{aligned}
\delta \boldsymbol{\gamma}_{[t]}^C
&=
\text{diag}\left(
\left(
\mathbf{S}_{[t-1]}^C
+
\mathbf{V}_{[t],new}^\top
\overrightarrow{\mathbf{K}_{[t]}} 
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
\delta \overleftarrow{\mathbf{Q}_{[t]}} 
\odot \mathbf{Q}_{[t]}
+
\delta \overleftarrow{\mathbf{K}_{[t]}} 
\odot \mathbf{K}_{[t]}
-
\left(
\delta \overrightarrow{\mathbf{K}_{[t]}} 
\odot \mathbf{K}_{[t]}
\right)
\oslash
\left(
\mathbf{\Gamma}_{[t]}
\odot \mathbf{\Gamma}_{[t]}
\right)
\\&=
\delta \overleftarrow{\mathbf{Q}_{[t]}} 
\odot \mathbf{\Gamma}_{[t]}
\odot \mathbf{Q}_{[t]}
\oslash \mathbf{\Gamma}_{[t]}
+
\left(
\delta \overleftarrow{\mathbf{K}_{[t]}} 
\odot \mathbf{\Gamma}_{[t]}
\odot \mathbf{K}_{[t]}
-
\delta \overrightarrow{\mathbf{K}_{[t]}} 
\oslash \mathbf{\Gamma}_{[t]}
\odot \mathbf{K}_{[t]}
\right)
\oslash
\mathbf{\Gamma}_{[t]}
\\&=
\left(
\delta \mathbf{Q}_{[t]}
\odot \mathbf{Q}_{[t]}
+
\left(
\delta \overleftarrow{\mathbf{K}_{[t]}} 
\odot \mathbf{\Gamma}_{[t]}
\right)
\odot \mathbf{K}_{[t]}
-
\left(
\delta \overrightarrow{\mathbf{K}_{[t]}} 
\oslash \mathbf{\Gamma}_{[t]}
\right)
\odot \mathbf{K}_{[t]}
\right)
\oslash
\mathbf{\Gamma}_{[t]}
\end{aligned}$$

specially:


$$\begin{aligned}
\delta \log \mathbf{\Gamma}_{[t]}
&=
\delta \mathbf{\Gamma}_{[t]}
\odot
\mathbf{\Gamma}_{[t]}
\\&=
\delta \mathbf{Q}_{[t]}
\odot \mathbf{Q}_{[t]}
+
\left(
\left(
\delta \overleftarrow{\mathbf{K}_{[t]}} 
\odot \mathbf{\Gamma}_{[t]}
\right)
\odot \mathbf{K}_{[t]}
-
\left(
\delta \overrightarrow{\mathbf{K}_{[t]}} 
\oslash \mathbf{\Gamma}_{[t]}
\right)
\odot \mathbf{K}_{[t]}
\right)
\\&+
[0,0,...,\text{diag}\left(
\mathbf{S}_{[t]}^{C\top}
\delta \mathbf{S}_{[t]}^C 
\right)]
\end{aligned}$$


