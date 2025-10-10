# ADMM for NMDs

Given a data matrix $X \in \mathbb R^ {m \times n}$ and a factorization rank $r\ll min(m,n)$, ADMM aims to solve the following
 nonlinear matrix decomposition problem

 \[ min_{T,W,H} d(X,f(T)) \quad \text{such that} \quad T = WH]