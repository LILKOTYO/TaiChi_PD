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

### 2022-05-20 notes

the difference between games103 slides and pd paper is because the pd is analyze a nonlinear FEM question while in the games 103, we analyze a cloth simulation(no volume) question.

### 2022-06-03 notes

the deformation gradient F and J is computed multiple times 

### 2022-06-04 notes

some bugs:

- Forget to initialize the sparse mass matrix. As a result, the b ($Ax=b$) is a zero vector in the first iteration and the units in the $\Delta x$ is a invalid value
- Calculate `LameMu` and `LameLa` before the `YoungsModulus` and `PoissonsRatio`  are assigned.

### 2022-06-05 notes

Record the idea of projective dynamics backpropagation here:

1. There is a optimal question which has a loss function $L$, we want to compute $\frac{\partial L}{\partial \bf y}$ from $\frac{\partial L}{\partial \bf x}$(both of them are row vectors, and ${\bf y} = {\bf x}_n + h{\bf v}_n+h^2M^{-1}{\bf f}_{ext}$). 
2. As $\bf x$ and $\bf y$ are implicitly constrained by $\nabla g({\bf x})=0$,  we can differentiate it with respect to $\bf y$ and so we can solve $\frac{\partial \bf x}{\partial \bf y}$ .
3. Use the chain rule to obtain the $\frac{\partial L}{\partial \bf y}$(a row vector): $\frac{\partial L}{\partial \bf y}=\frac{1}{h^2}{\bf z}^TM$. 
4. ${\bf z}$ can be solved from the linear system: $\nabla^2g({\bf x}){\bf z}=(\frac{\partial  L}{\partial\bf x})^T$
5. According to the 3. and 4. , backpropagation within one time step can be done

### 2022-06-07 notes

Record the idea of " PD forward simulation with contact" here:

1. local step do not change at all

2. the right-hand side vector b:
   
   1. before: ${\bf b} = \frac{M}{h^2}{\bf s}_n + \omega{\bf p}$
   
   2. new: ${\bf b} = \frac{M}{h^2}{\bf s}_n + \omega({\bf p}-{\bf x}_{n,C=x^*,\bar C=0})$

3. global step becomes to: ${\bf x}_{\bar C}=(A_{\bar C \times \bar C})^{-1}{\bf b}_{\bar C}$
   
   1. something new to the precomputing:  precompute $A^{-1}I_{\mathcal I \times C}$ using a maximum possible $C$ (all surface nodes)
   
   2. Separate colliding and non-colliding nodes: $\left(\begin{array}{cc}\mathrm{A}_{\mathrm{C\times \mathrm{C}}}  & 0 \\ 0 & \mathrm{~A}_{\overline{\mathrm{C}}\times \overline{\mathrm{C}}} \end{array}\right)$, so we can solve the global step question

4. calculate the contact force ${\bf r} = \frac{M}{h^2}({\bf x}-{\bf s}_n)-{\bf f}_{int}({\bf x})$

5. update $C$
