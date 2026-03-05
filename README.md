# 📊 Principal Component Analysis (PCA)

> **Dimensionality Reduction — Linear Method**  
> A principal Component Analysis: theory, math, and practical examples in Python.

<img width="1626" height="579" alt="image" src="https://github.com/user-attachments/assets/10294bb2-431d-47bd-8aa1-bc389f21a9af" />

---

## Mathematical Foundation

### 1. Data Centering (Standardization)

Before applying PCA, the data must be **centered** (and ideally standardized). Let $\mathbf{X}$ be an $n \times p$ data matrix with $n$ samples and $p$ features.

**Mean of each feature:**

$$\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$$

**Centered data matrix:**

$$\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top$$

where $\mathbf{1}$ is an $n$-dimensional vector of ones and $\bar{\mathbf{x}}$ is the mean vector.

**Standardized (Z-score) version** — recommended when features have different scales:

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{\sigma_j}$$

where $\sigma_j$ is the standard deviation of feature $j$.

---

### 2. Covariance Matrix

The **covariance matrix** $\mathbf{C}$ captures the pairwise linear relationships between all features. It is a $p \times p$ symmetric positive semi-definite matrix.

$$\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$$

The entry $C_{jk}$ represents the covariance between feature $j$ and feature $k$:

$$C_{jk} = \frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)$$

Properties:
- $C_{jj} = \sigma_j^2$ (variance of feature $j$ on the diagonal)
- $C_{jk} = C_{kj}$ (symmetric)
- All eigenvalues $\geq 0$ (positive semi-definite)

---

### 3. Eigendecomposition

PCA solves the eigenvalue problem on the covariance matrix:

$$\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

where:
- $\lambda_k$ — the **eigenvalue** (amount of variance explained by the $k$-th PC)
- $\mathbf{v}_k$ — the **eigenvector** (direction of the $k$-th PC), with $\|\mathbf{v}_k\| = 1$

The full decomposition is:

$$\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top$$

where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_p]$ — matrix of eigenvectors (columns)
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_p)$ — diagonal matrix of eigenvalues

Eigenvalues are sorted in **descending order**:

$$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$$

> **Why eigenvectors?** The eigenvectors of the covariance matrix define the directions of maximum variance. The corresponding eigenvalue tells us *how much* variance exists along that direction.

---

### 4. Selecting Principal Components

Choose the top $k$ eigenvectors corresponding to the $k$ largest eigenvalues. These form the **projection matrix**:

$$\mathbf{W}_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k] \in \mathbb{R}^{p \times k}$$

The value of $k$ is typically chosen so that the cumulative explained variance exceeds a threshold (e.g., 95%):

$$k =  left\{ k' : \frac{\sum_{i=1}^{k'} \lambda_i}{\sum_{i=1}^{p} \lambda_i} \geq \text{threshold}    right\}$$

---

### 5. Projection

Project the centered data onto the $k$ principal components:

$$\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{W}_k$$

where:
- $\tilde{\mathbf{X}} \in \mathbb{R}^{n \times p}$ — centered data
- $\mathbf{W}_k \in \mathbb{R}^{p \times k}$ — projection matrix (top $k$ eigenvectors)
- $\mathbf{Z} \in \mathbb{R}^{n \times k}$ — **reduced data** (PCA output)

Each row of $\mathbf{Z}$ is the **score** of one sample in the new $k$-dimensional space. Each column is called a **principal component score**.

The $i$-th sample's score on the $j$-th PC is:

$$z_{ij} = \tilde{\mathbf{x}}_i^\top \mathbf{v}_j = \sum_{l=1}^{p} \tilde{x}_{il} \cdot v_{lj}$$

---

### 6. Explained Variance Ratio

The **proportion of total variance** explained by the $k$-th principal component:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}$$

The **cumulative explained variance** of the first $k$ components:

$$\text{Cumulative EVR} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

This is used to build the **Scree Plot** and determine the optimal $k$.

---

### 7. Reconstruction

The original data can be approximately **reconstructed** from the reduced representation:

$$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{W}_k^\top + \mathbf{1}\bar{\mathbf{x}}^\top$$

The **reconstruction error** (mean squared error) is:

$$\text{MSE}_{\text{reconstruction}} = \frac{1}{n} \left\| \mathbf{X} - \hat{\mathbf{X}} \right\|_F^2 = \sum_{i=k+1}^{p} \lambda_i$$

where $\|\cdot\|_F$ is the Frobenius norm. This equals the sum of the **discarded** eigenvalues — the variance we lost by keeping only $k$ components.

---

## Step-by-Step Algorithm

```
Input:  X ∈ ℝⁿˣᵖ  (n samples, p features)
Output: Z ∈ ℝⁿˣᵏ  (n samples, k components)

Step 1:  Standardize each feature (zero mean, unit variance)
         X̃ = StandardScaler().fit_transform(X)

Step 2:  Compute the covariance matrix
         C = (1 / n-1) * X̃ᵀ X̃

Step 3:  Compute eigenvalues and eigenvectors of C
         C vₖ = λₖ vₖ

Step 4:  Sort eigenvectors by descending eigenvalue
         λ₁ ≥ λ₂ ≥ ... ≥ λₚ

Step 5:  Select top k eigenvectors → form W_k = [v₁, v₂, ..., vₖ]

Step 6:  Project data  →  Z = X̃ · W_k
```

---

## PCA from Scratch in Python

```python
np.random.seed(42)
mean = [2, 3]
cov = [[3, 2], [2, 2]] 
X = np.random.multivariate_normal(mean, cov, 200)
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered.T)
print('Covariance Matrix:')
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f'\nEigenvalues: {eigenvalues}')
print(f'Eigenvectors:\n{eigenvectors}')
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]
X_pca_manual = X_centered @ eigenvectors[:, :1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, color='steelblue', label='Original Data')
origin = np.mean(X, axis=0)
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    axes[0].annotate('', xy=origin + val*vec, xytext=origin,
                     arrowprops=dict(arrowstyle='->', color=['red','green'][i], lw=2))
    axes[0].text(*(origin + val*vec), f'  PC{i+1}', fontsize=12,
                 color=['red','green'][i], fontweight='bold')
axes[0].set_title('Original Data + Principal Components', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_pca_manual, np.zeros_like(X_pca_manual), alpha=0.5, color='coral')
axes[1].set_title('After PCA → 1 Dimension (PC1)', fontsize=13)
axes[1].set_xlabel('PC1')
axes[1].set_yticks([])
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA Implementation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
print('PCA completed!')
```
<img width="1445" height="640" alt="image" src="https://github.com/user-attachments/assets/6062c73b-5036-46ea-9a0d-1f0b9779bd76" />

---

## PCA with Scikit-Learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load data 
iris         = load_iris()
X, y         = iris.data, iris.target        # shape: (150, 4)
target_names = iris.target_names

# Standardize
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (2 components)
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape : {X.shape}")          # (150, 4)
print(f"Reduced shape  : {X_pca.shape}")      # (150, 2)
print(f"EVR per PC     : {pca.explained_variance_ratio_}")
print(f"Total variance : {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Visualize 
colors  = ['#e74c3c', '#2ecc71', '#3498db']
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, color, marker) in enumerate(zip(target_names, colors, markers)):
    idx = y == i
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, marker=marker,
               label=name, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA on Iris Dataset (4D → 2D)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
```
<img width="1519" height="591" alt="image" src="https://github.com/user-attachments/assets/d025b25c-0ccf-4dca-b57d-67d29675e151" />

---

## Choosing the Optimal Number of Components

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load Digits dataset (64 features)
digits         = load_digits()
X_scaled       = StandardScaler().fit_transform(digits.data)

# Fit PCA with all components 
pca_full       = PCA().fit(X_scaled)
cumvar         = np.cumsum(pca_full.explained_variance_ratio_)

n_95 = np.argmax(cumvar >= 0.95) + 1     # components for 95% variance
n_99 = np.argmax(cumvar >= 0.99) + 1     # components for 99% variance

print(f"Components for 95% variance : {n_95}  (out of 64)")
print(f"Components for 99% variance : {n_99}  (out of 64)")
print(f"Feature reduction (95%)     : {(1 - n_95/64)*100:.1f}%")

# Plot 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, 21), pca_full.explained_variance_ratio_[:20] * 100,
            color='steelblue', edgecolor='black')
axes[0].set_xlabel('Component Number')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Scree Plot')

# Cumulative variance
axes[1].plot(range(1, len(cumvar)+1), cumvar * 100, 'b-', lw=2)
axes[1].axhline(y=95, color='red',   ls='--', label=f'95% → {n_95} components')
axes[1].axhline(y=99, color='green', ls='--', label=f'99% → {n_99} components')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance (%)')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.suptitle('Optimal Number of PCA Components — Digits Dataset', fontweight='bold')
plt.tight_layout()
plt.show()
```
<img width="1529" height="533" alt="image" src="https://github.com/user-attachments/assets/047e87b1-be58-49b0-9649-4a9e14f9f640" />

---

## PCA on Image Data — Eigenfaces

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load face dataset 
faces   = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data                      # shape: (400, 4096)  ← 64×64 pixels

# Apply PCA 
pca_faces       = PCA(n_components=100, whiten=True, random_state=42)
X_faces_reduced = pca_faces.fit_transform(X_faces)
X_reconstructed = pca_faces.inverse_transform(X_faces_reduced)

total_var = pca_faces.explained_variance_ratio_.sum() * 100
print(f"Variance explained by 100 components: {total_var:.2f}%")
print(f"Compression ratio: 4096 → 100  ({(1 - 100/4096)*100:.1f}% smaller)")

# Show original vs reconstructed 
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i in range(8):
    axes[0, i].imshow(X_faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title('Original', pad=4)

    axes[1, i].imshow(X_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title('Reconstructed', pad=4)

plt.suptitle(f'Eigenfaces: Original vs Reconstructed ({total_var:.1f}% variance)',
             fontweight='bold')
plt.tight_layout()
plt.show()
```
<img width="1593" height="996" alt="image" src="https://github.com/user-attachments/assets/811107a0-fc7d-42af-a681-692c10856b56" />

---

## PCA as Preprocessing Before Classification

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

# Without PCA 
pipe_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=1000, random_state=42))
])

# With PCA (retain 95% variance)
pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=0.95)),
    ('clf',    LogisticRegression(max_iter=1000, random_state=42))
])

pipe_baseline.fit(X_train, y_train)
pipe_pca.fit(X_train, y_train)

acc_base = pipe_baseline.score(X_test, y_test)
acc_pca  = pipe_pca.score(X_test, y_test)
n_comp   = pipe_pca.named_steps['pca'].n_components_

print(f"Without PCA | 64 features  | Accuracy: {acc_base*100:.2f}%")
print(f"With PCA    | {n_comp} features  | Accuracy: {acc_pca*100:.2f}%")
print(f"Feature reduction: {(1 - n_comp/64)*100:.1f}%")
```
<img width="805" height="611" alt="image" src="https://github.com/user-attachments/assets/a3cde24e-3d39-42b7-a8fa-6620375cf2bb" />

---

## PCA 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris     = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

pca_3d    = PCA(n_components=3)
X_iris_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')

colors = ['#e74c3c', '#2ecc71', '#3498db']
for i, (name, color) in enumerate(zip(iris.target_names, colors)):
    idx = iris.target == i
    ax.scatter(X_iris_3d[idx, 0], X_iris_3d[idx, 1], X_iris_3d[idx, 2],
               c=color, label=name, s=50, alpha=0.8, edgecolors='black', linewidth=0.3)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_title('PCA 3D — Iris Dataset', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
```
<img width="758" height="710" alt="image" src="https://github.com/user-attachments/assets/53651623-a4f5-4d86-8b10-33e0873eee04" />

---

## Summary & Practical Tips

### Algorithm Summary

| Step | Operation | Formula |
|:---:|---|---|
| 1 | Standardize data | $z = (x - \mu) / \sigma$ |
| 2 | Covariance matrix | $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ |
| 3 | Eigendecomposition | $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$ |
| 4 | Sort by $\lambda$ descending | $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$ |
| 5 | Select top $k$ eigenvectors | $\mathbf{W}_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$ |
| 6 | Project data | $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{W}_k$ |

---

## References

- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 12.
- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press. Chapter 10.
- [Scikit-Learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [A Tutorial on Principal Component Analysis — Jonathon Shlens (arXiv:1404.1100)](https://arxiv.org/abs/1404.1100)

---

<div align="center">
