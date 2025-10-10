# ADMM for NMDs

Given a data matrix $X \in \mathbb R^ {m \times n}$ and a factorization rank $r\ll min(m,n)$, ADMM aims to solve the following
 nonlinear matrix decomposition problem

 $$ \;\; min_{T,W,H} d(X,f(T)) \quad \text{such that} \quad T = WH$$

 where -$f$ is an elementwise nonlinear function,
       -$d$ is a loss function,
       -$T \in \mathbb R^ {m \times n}$ is a rank-$r$ matrix representing a low-rank approximation of X, 
       -$W \in \mathbb R^ {m \times r}$  and $H \in \mathbb R^ {r \times n}$ are the rank-r factors.

The proposed ADMM algorithm minimizes the Augmented Lagrangian of the NMD problem $$L(W,H,T,\Lambda) = d(X, f(T)) + \langle  T - WH \,, \Lambda \rangle + \frac{\rho}{2} \|T - WH\|_F^2$$

with respect to the primal variables $T$, $W$ and $H$ and the dual variable $\Lambda$ in an alternating sequence.  