import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_s_curve

N = 1000  
D = 3     
errors = []  
m_values = []  

X, t = make_s_curve(N, noise=0.1)
X = X[t.argsort()]

p_index = np.random.choice(N) 
p = X[p_index].reshape(1, -1)

K_ = rbf_kernel(X, p, gamma=1).flatten()

for m in range(2, 1000, 10): 
    W = np.random.normal(loc=0, scale=1, size=(m, D))
    b = np.random.uniform(0, 2 * np.pi, size=m)
    
    Z_cos = np.sqrt(2 / m) * np.cos(W @ X.T + b[:, np.newaxis])
    Z_sin = np.sqrt(2 / m) * np.sin(W @ X.T + b[:, np.newaxis])
    
    ZZ = Z_cos.T @ Z_cos + Z_sin.T @ Z_sin
    
    error = np.linalg.norm(K_ - ZZ) / (N ** 2)
    errors.append(error)
    m_values.append(m)

plt.figure(figsize=(10, 6))
plt.plot(m_values, errors, marker='o')
plt.title('Approximation Error of Random Fourier Features', fontdict={'fontsize': 18})
plt.xlabel('Number of dimensions (m)', fontdict={'fontsize': 14})
plt.ylabel('Approximation Error |K(x, p) - ⟨φ̂(x), φ̂(p)⟩|', fontdict={'fontsize': 14})
plt.yscale('log')  # Log scale for better visualization of error
plt.axhline(y=0.01, color='r', linestyle='--', label='Target Error = 0.01')
plt.legend()
plt.grid()
plt.show()
