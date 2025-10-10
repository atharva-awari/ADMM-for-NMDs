# ADMM for Nonlinear Matrix Decompositions (NMD)

Given a data matrix $X \in \mathbb{R}^{m \times n}$ and a rank $r \ll \min(m,n)$,  
we solve the nonlinear matrix decomposition problem:

$$
\min_{T,\,W,\,H} \; d(X, f(T))
\quad \text{s.t.} \quad
T = WH
$$

where:
- $f(\cdot)$ is an elementwise nonlinear function,
- $d(\cdot,\cdot)$ is the loss,
- $T \in \mathbb{R}^{m \times n}$ is a rank-$r$ matrix,
- $W \in \mathbb{R}^{m \times r}$ and $H \in \mathbb{R}^{r \times n}$ are factors.






