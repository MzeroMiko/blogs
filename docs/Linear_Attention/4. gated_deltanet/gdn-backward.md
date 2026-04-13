# Reading Notes: Backward Pass for Gated DeltaNet

### Revisiting the Forward Pass

Let us first collect the forward equations used in the backward derivation:

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

### Gradient with Respect to `U_[t]`

Starting from the forward equations, we obtain the gradient with respect to `U_[t]` as:

$$\begin{aligned}
\delta \mathbf{U}_{[t]}
=
\gamma_{[t]}^C \overrightarrow{\mathbf{K}_{[t]}} \delta \mathbf{S}_{[t]}^{C \top}
+
\left( \overleftarrow{\mathbf{Q}_{[t]}}\overrightarrow{ \mathbf{K}_{[t]}}^\top \odot \mathbf{M}\right)^\top  \delta \mathbf{O}_{[t]}
\end{aligned}$$

### Gradient with Respect to `W_left_[t]`

Next, for `W_left_[t]`, we have:

$$\begin{aligned}
\delta \overleftarrow{\mathbf{W}_{[t]}}
=
- \delta \mathbf{U}_{[t]} \mathbf{S}_{[t-1]}^{C}
\end{aligned}$$


### Gradient with Respect to `S_[t]^C`

We now turn to the chunk state.  
First, the contribution coming from `S_[t]^C` itself is:

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{S}_{[t]}^{C}}
&=
\gamma_{[t]}^C \delta \mathbf{S}_{[t]}^{C}
-
\gamma_{[t]}^C  \delta \mathbf{S}_{[t]}^{C} \overrightarrow{\mathbf{K}_{[t]}}^\top  \overleftarrow{\mathbf{W}_{[t]}} 
\end{aligned}$$

Meanwhile, the contribution coming from `O_[t]` is:

$$\begin{aligned}
\left.\delta \mathbf{S}_{[t-1]}^{C} \right|_{\text{from } \mathbf{O}_{[t]}}
&=
\delta \mathbf{O}_{[t]}^\top  \overleftarrow{\mathbf{Q}_{[t]}}
-
\delta \mathbf{O}_{[t]}^\top
\left( \overleftarrow{\mathbf{Q}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M} \right)  
\overleftarrow{\mathbf{W}_{[t]}}
\end{aligned}$$

Combining the two terms gives:

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

### Gradient with Respect to `Q_[t]`

Next, the gradient with respect to `Q_[t]` can be written as:

$$\begin{aligned}
\delta \mathbf{Q}_{[t]}
=
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \delta \mathbf{O}_{[t]} \mathbf{S}_{[t-1]}^C
+
\text{Diag}(\boldsymbol{\gamma}_{[t]}) \left(\delta \mathbf{O}_{[t]}
\mathbf{V}_{[t],new}^\top \odot \mathbf{M}\right) \overrightarrow{\mathbf{K}_{[t]}}
\end{aligned}$$

### Gradient with Respect to `V_[t]`

Since `U_[t]` is expressed as a linear transform of `V_[t]`, it follows that:

$$\begin{aligned}
\delta \mathbf{V}_{[t]}
&=
\text{Diag}(\boldsymbol{\beta}_{[t]})
\widetilde{\mathbf{A}}_{[t]}^\top 
\delta \mathbf{U}_{[t]}
\end{aligned}$$

### Gradients with Respect to `A_[t]` and `X_[t]`


First, for `A_[t]`, we have:

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

### Gradient with Respect to `K_[t]` 

The gradient with respect to `K_[t]` receives contributions from several paths.

First, the part coming from `X_[t]` is:

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

Next, the part coming from `S_[t]` without passing through `V_[t],new` is:

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
&=
\gamma_{[t]}^C 
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1} 
\mathbf{V}_{[t],new} 
\delta \mathbf{S}_{[t]}^{C}
\end{aligned}$$

Similarly, the part coming from `O_[t]` without passing through `V_[t],new` is:

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

In addition, the part coming from `W_left_[t]` without passing through `T_[t]` is:

$$\begin{aligned}
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \overleftarrow{\mathbf{W}_{[t]}}  \text{ w/o } \mathbf{T}_{[t]} }
&=
\text{Diag}(\boldsymbol{\beta}_{[t]}) 
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}}
\end{aligned}$$

Therefore, after collecting all these contributions, we get:

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


### Gradient with Respect to `beta_[t]`


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


> **Comment**: Further simplifications here can accelerate FLA under `register` constraints.
> 
> $$\begin{aligned}
> \delta \boldsymbol{\beta}_{[t]} 
> &=
> \text{diag}\left( 
> \widetilde{\mathbf{A}}_{[t]}^\top 
> \delta \overleftarrow{\mathbf{W}_{[t]}} 
> \mathbf{K}_{[t]}^\top 
> \text{Diag}(\boldsymbol{\gamma}_{[t]}) 
> + \widetilde{\mathbf{A}}_{[t]}^\top 
> \delta \mathbf{U}_{[t]} 
> \mathbf{V}_{[t]}^\top \right) 
> \\& 
> + \text{diag}\left( 
> \left( 
> \delta \widetilde{\mathbf{X}}_{[t]} 
> \odot \mathbf{M}_{-1} 
> \right) 
> \overrightarrow{\mathbf{K}_{[t]}} 
> \overleftarrow{\mathbf{K}_{[t]}}^\top
> \right)
> \\&=
> \text{diag}\left( 
> \widetilde{\mathbf{A}}_{[t]}^\top 
> \delta \widetilde{\mathbf{A}}_{[t]}
> \right) 
> \odot \boldsymbol{\beta}_{[t]}^{-1} 
> + \text{diag}\left( 
> \left( 
> \delta \widetilde{\mathbf{X}}_{[t]} 
> \odot \mathbf{M}_{-1} 
> \right) 
> \overrightarrow{\mathbf{K}_{[t]}} 
> \overleftarrow{\mathbf{K}_{[t]}}^\top
> \right)
> \end{aligned}$$



### Gradient with Respect to `gamma_[t]`


We first have:

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
\left(\boldsymbol{\gamma}_{[t]}^C\right)^{-1} 
\text{Tr}\left(
\delta \mathbf{S}_{[t]}^C \mathbf{S}_{[t]}^{C \top}
\right)
\end{aligned}$$

> **Comment**: In practice, replacing two matrix multiplications with an extra memory load may not be beneficial.

Next, the contribution from `S_[t]^C` without passing through `V_[t],new` is:

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

Meanwhile, the contribution from `O_[t]` without passing through `V_[t],new` is:

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
\left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{V}_{[t],new} }
\mathbf{K}_{[t]}^\top
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\end{aligned}$$


> **Comment**: We can further simplify this part.
> 
> $$\begin{aligned}
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
> &=
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \delta \mathbf{Q}_{[t]}
> \mathbf{Q}_{[t]}^\top
> -
> \left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{O}_{[t]} \text{w/o } \mathbf{V}_{[t],new} }
> \mathbf{K}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\
> \\ 
> \Rightarrow
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/ } \mathbf{O}_{[t]} \text{w/o} \mathbf{V}_{[t],new}}
> &=
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \delta \mathbf{Q}_{[t]}
> \mathbf{Q}_{[t]}^\top
> -
> \left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/ } \mathbf{O}_{[t]} \text{w/o } \mathbf{V}_{[t],new} }
> \mathbf{K}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \end{aligned}$$


In the same way, the contribution from `U_[t]` together with `W_[t]`, but without passing through `A_[t]`, is:

$$\begin{aligned}
\left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{U}_{[t]} \text{w/ } \mathbf{W}_{[t]} \text{w/o} \widetilde{\mathbf{A}}_{[t]} }
&=
\widetilde{\mathbf{A}}_{[t]}^\top
\delta \overleftarrow{\mathbf{W}_{[t]}} 
\mathbf{K}_{[t]}^\top 
\text{Diag}(\boldsymbol{\beta}_{[t]})
\end{aligned}$$

The contribution from `A_[t]`, is:

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


> **Comment**: Further simplification is possible here, but it requires a fresh derivation from the ground up following a different roadmap.
> 
> $$\begin{aligned}
> \mathbf{\widetilde{X}}_{[t]} &= \mathbf{I} + \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \overleftarrow{\mathbf{K}_{[t]}} \overrightarrow{\mathbf{K}_{[t]}}^\top \odot \mathbf{M}_{-1}  \right)
> =
> \text{Diag}(\boldsymbol{\gamma}_{[t]}) 
> \mathbf{X}_{[t]}
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\
> \\
> \Rightarrow
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]} }
> &=
> \delta \widetilde{\mathbf{X}}_{[t]}
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \mathbf{X}_{[t]}^\top
> -
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \mathbf{X}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})
> \delta \widetilde{\mathbf{X}}_{[t]}
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\ &=
> \delta \widetilde{\mathbf{X}}_{[t]}
> \widetilde{\mathbf{X}}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> -
> \widetilde{\mathbf{X}}_{[t]}^\top
> \delta \widetilde{\mathbf{X}}_{[t]}
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\ &=
> \left(
> - \widetilde{\mathbf{A}}_{[t]}^\top
> \delta \widetilde{\mathbf{A}}_{[t]}
> +
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> \right)
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\
> \\
> \Rightarrow
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{U}_{[t]} \text{ w/ } \mathbf{W}_{[t]}}
> &= 
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \mathbf{U}_{[t]} \text{ w/ } \mathbf{W}_{[t]} \text{ w/o } \widetilde{\mathbf{A}}_{[t]}}
> +
> \left.\delta \text{Diag}(\boldsymbol{\gamma}_{[t]})\right|_{\text{from } \widetilde{\mathbf{X}}_{[t]}}
> \\ &=
> - \widetilde{\mathbf{A}}_{[t]}^\top
> \left(
> \delta \widetilde{\mathbf{A}}_{[t]}
> -
> \delta \overleftarrow{\mathbf{W}_{[t]}} 
> \mathbf{K}_{[t]}^\top 
> \text{Diag}(\boldsymbol{\beta}_{[t]})
> \text{Diag}(\boldsymbol{\gamma}_{[t]})
> \right)
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\&+
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\ &=
> - \widetilde{\mathbf{A}}_{[t]}^\top
> \left(
> \delta \mathbf{U}_{[t]}
> \mathbf{V}_{[t]}^\top
> \text{Diag}(\boldsymbol{\beta}_{[t]}) 
> \right)
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> +
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\ &=
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> -
> \text{Diag}(\boldsymbol{\beta}_{[t]})^{-1}
> \left(
> \delta \mathbf{V}_{[t]}
> \mathbf{V}_{[t]}^\top
> \right)
> \text{Diag}(\boldsymbol{\beta}_{[t]}) 
> \text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
> \\
> \\
> \Rightarrow
> \left.\delta \boldsymbol{\gamma}_{[t]}\right|_{\text{from } \mathbf{U}_{[t]} \text{ w/ } \mathbf{W}_{[t]}}
> &=
> \text{diag}\left( 
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> -
> \delta \mathbf{V}_{[t]}
> \mathbf{V}_{[t]}^\top
> \right)
> \odot \boldsymbol{\gamma}_{[t]}^{-1}
> \end{aligned}$$

Putting everything together, we arrive at:

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

> **Comment**: Alternatively, consolidating all simplifications into a single approach leads to the formulation below.
> 
> $$\begin{aligned}
> \delta \boldsymbol{\gamma}_{[t]} 
> &= 
> \text{diag}\left(
> \delta \mathbf{Q}_{[t]}
> \mathbf{Q}_{[t]}^\top
> \right)
> \odot \boldsymbol{\gamma}_{[t]}^{-1}
> \\&-
> \text{diag}\left(
> \left.\delta \mathbf{K}_{[t]}\right|_{\text{from } \mathbf{S}_{[t]} \text{ w/ } \mathbf{O}_{[t]} \text{ w/o } \mathbf{V}_{[t],new}}
> \mathbf{K}_{[t]}^\top 
> \right)
> \odot \boldsymbol{\gamma}_{[t]}^{-1}
> \\&+ 
> \text{diag}\left( 
> \delta \widetilde{\mathbf{A}}_{[t]}
> \widetilde{\mathbf{A}}_{[t]}^\top
> -
> \delta \mathbf{V}_{[t]}
> \mathbf{V}_{[t]}^\top
> \right)
> \odot \boldsymbol{\gamma}_{[t]}^{-1}
> \\ &+
> [0, 0, ..., \delta \boldsymbol{\gamma}_{[t]}^C]^\top
> \end{aligned}$$


### Gradient with Respect to `alpha_[t]`

$$\begin{aligned}
\delta \log \boldsymbol{\alpha}_{[t]}
&=
\text{suffix\_cumsum}(\delta \log \mathbf{\gamma}_{[t]})
\end{aligned}$$

### Comments

Someone may notice that, since

$$\begin{aligned}
\mathbf{X}_{[t]} 
&= 
\mathbf{I} 
+ \text{Diag}(\boldsymbol{\beta}_{[t]}) 
\left( 
\mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top 
\odot \mathbf{M}_{-1}  
\right)
\\
\\
\mathbf{\widetilde{X}}_{[t]} 
&= 
\text{Diag}(\boldsymbol{\gamma}_{[t]}) 
\mathbf{X}_{[t]}
\text{Diag}(\boldsymbol{\gamma}_{[t]})^{-1}
\\
\\
\mathbf{A}_{[t]}
&=
\mathbf{X}_{[t]}^{-1}
\end{aligned}$$

materializing `A` or `Diag{\gamma}A` may lead to a more convenient formulation that reuses the original intermediate results.

However, this can be numerically risky when `Diag(\gamma)^{-1}` appears in isolation while computing certain terms, because it may introduce extremely large values and amplify instability. Keep in mind the **secondary chunking** principle proposed in `GLA`.


