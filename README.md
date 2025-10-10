# ADMM for NMDs

Given a data matrix $X \in \mathbb R^ {m \times n}$ and a factorization rank $r\ll min(m,n)$, ADMM aims to solve the following
 nonlinear matrix decomposition problem

 $$ \;\; min_{T,W,H} d(X,f(T)) \quad \text{such that} \quad T = WH$$

 where $f$ is an elementwise nonlinear function,
       $d$ is the loss function,
       $T \in \mathbb R^ {m \times n}$ is a rank-$r$ matrix representing a low-rank approximation of $X$, 
       $W \in \mathbb R^ {m \times r}$  and $H \in \mathbb R^ {r \times n}$ are the rank-r factors.

# Nonlinearities and Loss functions handled by ADMM
Our proposed ADMM algorithm for NMD can handle 12 different combinations of nonlinearities and loss functions. They are listed as follows. 

Nonlinear functions $f(T)$ : 

    - \textbf{ReLU:} $f(T) = \max(0, T)$, 
    - \textbf{CSF:} $f(T) = T^{.2}$,
    - \textbf{Min--Max:} $f(T) = \min(b, \max(a, T))$,
    - \textbf{Modulus:} $f(T) = |T|$.  

Loss functions $d(X, f(T))$:  

    - \textbf{Frobenius norm:} \[\|X - f(T)\|_F = \sqrt{ \sum_{i=1}^m \sum_{j=1}^n \big( X_{ij} - [f(T)]_{ij} \big)^2 },\]  
    - \textbf{$\ell_1$ norm:} \[\|X - f(T)\|_1 = \sum_{i=1}^m \sum_{j=1}^n \left|\, X_{ij} - [f(T)]_{ij} \,\right|,\]  
    - \textbf{Kullback--Leibler (KL) divergence:}  Given two nonnegative scalars $x$ and $y$, the Kullback-Leibler divergence between $x$ and $y$ is defined as :
    $
        d_{\text{KL}}(x,y) = 
        \begin{cases}
            x \log \!\left( \tfrac{x}{y} \right) - x + y & \text{if } x > 0, \\
            y & \text{if } x = 0,
        \end{cases}
    $
    with $
        KL(X , f(WH)) = \sum_{i,j} d_{\text{KL}} (X_{ij}, f(T)_{ij}) .
    $  
\end{itemize} 

The proposed ADMM algorithm minimizes the Augmented Lagrangian of the NMD problem $$L(W,H,T,\Lambda) = d(X, f(T)) + \langle  T - WH \,, \Lambda \rangle + \frac{\rho}{2} \|T - WH\|_F^2$$

with respect to the primal variables $T$, $W$ and $H$ and the dual variable $\Lambda$ in an alternating sequence. 