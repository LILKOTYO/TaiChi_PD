# TaiChi_PD
try a PD soft body simulation and here is my notes

### 2022-05-20 notes
in the projective dynamics paper. the newton method is : 
$$
\nabla^2F({\bf q}) \Delta{\bf q}=-\nabla F({\bf q})
$$
The gradient of potential energy W is :
$$
\begin{align*}
W&=\sum_i\frac{\omega_i}{2}\left\| M_i^{\frac{1}{2}}(S_i{\bf q}-{\bf p}_i)\right\|_F^2+\delta_{C_i}({\bf p_i})\\
\nabla W&=\sum_i\omega_iS_i^T(M_i^{\frac{1}{2}})^TM_i^{\frac{1}{2}}(S_i{\bf q}-{\bf p_i})
\end{align*}
$$
This may not be completely accurate physically, but it is more reasonable in formula derivation.

But in GAMES 103, Mr. Wang use a more physical way to calculate $\nabla W$:
$$
\nabla W = - F_{int}({\bf q})
$$
and in this way, it seems like we don't even need projection function in the code implementation.

So strange...

