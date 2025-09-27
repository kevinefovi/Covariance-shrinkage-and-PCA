# Covariance Shrinkage and PCA on a high dimensional dataset (Arcene 700x10,000)

The goal was to show the bias-variance tradeoff for PCA on a high dimensional dataset. We compare the PCA built on the raw covariance to PCA built on a shrunk covariance with 2 independent knobs:

- Correlation shrink (alpha_corr) - dampens sample correlations towards their identity
- Variance shrink (alpha_var) - flatten sample variances towards their mean

In T << N, the sample covariance has at least N - (T-1) eigenvalues at exactly 0 (9,301 for our dataset). This means we have a huge nullspace where directions are completely undetermined by the data. Even in our signal subspace, the finite sample noise inflates 
some eigenvalues and deflates others, and the associated eigenvectors are unstable (especially when eigengaps are small).

We tune (alpha_corr, alpha_var, K) (top K PCs) by 5 fold CV and calculate the OOS reconstruction MSE and explained variance EVR. We found a small correlation shrinkage (alpha_corr = 0.2, alpha_var = 0.0, K=100) gave the global minimum OOS reconstruction MSE. In theory, this is because of the PCs rotating towards a more stable and 
less noisy region. The variance reduction outweighs the added bias at this point, however we haven't tested the statistical significance of this result here.

We plotted the results (alpha_corr, alpha_var, K) for K=100.

<img width="640" height="480" alt="mse_mean_plot1" src="https://github.com/user-attachments/assets/6b998fea-b25e-470d-ac42-faaba01334d8" />

Along alpha_corr, we see the classical bias-variance tradeoff with an extremely shallow U shape minimum at 0.2 As we increase alpha_corr beyond 0.2, we start to kill the true structure 
of the data, as evident by the steep increase of OOS reconstruction MSE towards alpha_corr ~ 0.8. Along alpha_var, we see a monotonic rise.
