#------------------------------------------------ Approximation ---------------------

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


#---------------- Visualization ----------------------------

import matplotlib.pyplot as plt
import numpy as np
from   sklearn.metrics.pairwise import rbf_kernel
from   sklearn.datasets import make_s_curve


# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 7)
fig.set_size_inches(20, 4)
font = {'fontname': 'arial', 'fontsize': 9}

N    = 1000
D    = 3
X, t = make_s_curve(N, noise=0.1)
X    = X[t.argsort()]
# The RBF kernel is the Gaussian kernel if we let \gamma = 1 / (2 \sigma^2).
K    = rbf_kernel(X, gamma=1)

axes[0].imshow(K, cmap=plt.cm.Blues)
axes[0].set_title(r'$K(x, p) = exp(-||x - p||^2)$', **font)
axes[0].set_xticks([])
axes[0].set_yticks([])

for R, ax in zip([1, 10, 100, 1000, 10000, 100000], axes[1:]):
    W    = np.random.normal(loc=0, scale=1, size=(R, D))
    b    = np.random.uniform(0, 2*np.pi, size=R)
    B    = np.repeat(b[:, np.newaxis], N, axis=1)
    norm = 1./ np.sqrt(R)
    Z    = norm * np.sqrt(2) * np.cos(W @ X.T + B)
    ZZ   = Z.T@Z

    ax.imshow(ZZ, cmap=plt.cm.Blues)
    ax.set_title(r'$K(x, p) = exp(-||x - p||^2)$, $R=%s$' % R, **font)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
