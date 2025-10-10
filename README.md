# ADMM for Nonlinear Matrix Decompositions (NMD)

Given a data matrix \( X \in \mathbb{R}^{m \times n} \) and a factorization rank \( r \ll \min(m,n) \), we solve the **nonlinear matrix decomposition (NMD)**

\[
\min_{T,\,W,\,H} \; d(X, f(T))
\quad \text{such that} \quad
T = WH
\]

where:
- \( f(\cdot) \) is an elementwise nonlinearity,
- \( d(\cdot,\cdot) \) is a data-fidelity loss,
- \( T \in \mathbb{R}^{m \times n} \) is rank-\(r\),
- \( W \in \mathbb{R}^{m \times r} \), \( H \in \mathbb{R}^{r \times n} \).

---

## Nonlinearities and Losses

**Nonlinearities \( f(T) \)** (elementwise):
- **ReLU:** \( f(T) = \max(0, T) \)  
- **CSF:**  \( f(T) = T^{.2} \)  (elementwise square)  
- **Min–Max (clipping):** \( f(T) = \min(b, \max(a, T)) \)  
- **Modulus:** \( f(T) = |T| \)

**Losses \( d(X, f(T)) \):**
- **Frobenius norm:**  
  \[
  \|X - f(T)\|_F = \sqrt{ \sum_{i=1}^m \sum_{j=1}^n \big(X_{ij} - [f(T)]_{ij}\big)^2 }
  \]
- **\(\ell_1\) norm:**  
  \[
  \|X - f(T)\|_1 = \sum_{i=1}^m \sum_{j=1}^n \left| X_{ij} - [f(T)]_{ij} \right|
  \]
- **Kullback–Leibler (KL) divergence** (for nonnegative \(x,y\)):
  \[
  d_{\text{KL}}(x,y) =
  \begin{cases}
    x \log \!\left( \tfrac{x}{y} \right) - x + y, & x>0\\
    y, & x=0
  \end{cases}
  \quad\Rightarrow\quad
  KL(X,f(T)) = \sum_{i,j} d_{\text{KL}}\!\big(X_{ij}, f(T)_{ij}\big)
  \]

We support the **12 combinations** from the 4 nonlinearities × 3 losses above.

---

## Augmented Lagrangian and ADMM

We minimize the augmented Lagrangian
\[
L(W,H,T,\Lambda)
= d(X,f(T)) + \langle T - WH, \Lambda \rangle
+ \frac{\rho}{2}\,\|T - WH\|_F^2,
\]
where \( \Lambda \) is the dual variable and \( \rho>0 \) the penalty.



