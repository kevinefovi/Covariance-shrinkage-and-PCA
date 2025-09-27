import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, ArpackError
from sklearn.model_selection import KFold
import os
import json

results = "tuning_results.json"

def save(obj):
    with open(results, "w") as f:
        json.dump(obj, f ,indent=2)

def load():
    if os.path.exists(results):
        with open(results) as f:
            return json.load(f)
    return None

# data
X_train = np.loadtxt("data/arcene_train.data", dtype=np.float64)
X_valid = np.loadtxt("data/arcene_valid.data", dtype=np.float64)
X_test = np.loadtxt("data/arcene_test.data", dtype=np.float64)

"""
Motivations for this project was to showcase the bias-variance tradeoff with a real world high dimensional dataset (700x10,000 - UCI ML repo Arcene)
To do this, we shrink the covariance towards a target using 2 orthogonal knobs: 1 for dampening correlations and the other to flatten variance
We then do PCA on this shrunk covariance and test the OOS MSE (and EVR for insight)

The U shaped OOS error along the correlation dampening strength is extremely shallow, with the minimum 
observed MSE on (0.2, 0.0) for (alpha_corr, alpha_var) indicating slightly scaling down correlations rotates the top K PCs towards a more 
stable estimate (lower variance) without overly biasing it however, we haven't tested the statistical signficance of this result here
"""

def sample_cov(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (len(X) - 1)

def shrink_cov(S, alpha_corr, alpha_var):
    T, N = S.shape
    d = np.diag(S)
    d_c = np.clip(d, 1e-15, None)
    m = d.mean()

    # it was crucial to not build the full dense diagonal matrix since it would be 10,000x10,000
    # instead we used broadcasting to do diagonal operations on the covariance matrix
    s = np.sqrt(d_c)
    C = (S / s) / s[:, None] # (S / s) is (N, N) / (N,) so each column j is divided by s[j]
                             # / s[:, None] is shape (N, 1) so each row i is divided be s[i]

    Dv = (1 - alpha_var) * d + alpha_var * m
    Dv = np.clip(Dv, 1e-12, None)
    t = np.sqrt(Dv)
    Cc = (1 - alpha_corr) * C + alpha_corr * np.eye(N)

    sig = (t[:, None] * Cc) * t[None, :] # multiplies row i by t[i] (left multiply) 
                                         # multiplies column j by t[j] (right multiply)
    sig += 1e-12 * m * np.eye(N)
    return sig # result without forming dense diagonal matrices

def pca_cov(S, K):
    N = S.shape[0]
    K = int(min(K, N-1))
    if K < 1:
        raise ValueError("error")
    
    S = 0.5 * (S + S.T)
    try:
        # added ncv to increase subspace and seperate clustered values 
        lam, V = eigsh(S, k=K, which="LA", ncv=min(N, max(2*K+1, K+40)), maxiter=40000, tol=1e-3)
    except ArpackError: 
        # sometimes with ill conditioned matrices the top K eigenvalues are very close and eigsh cant 
        # converge - fallback to eigh as the robust solver (slow)
        lam, V = eigh(S)
        lam, V = lam[::-1][:K], V[:, ::-1][:, :K]

    order = np.argsort(lam)[::-1]
    lam_K = lam[order]
    V_K = V[:, order]

    return V_K, lam_K

def explained_var(X_val, mean_tr, V_K):
    Xc = X_val - mean_tr
    Z = Xc @ V_K
    num = Z.var(axis=0, ddof=1).sum()
    den = Xc.var(axis=0, ddof=1).sum()
    return (num / den) if den > 0 else 0.0

def project_scores(X, mean, V_K):
    return (X - mean) @ V_K

def reconstruction(Z, mean, V_K):
    return Z @ V_K.T + mean

def tune_hyperparameters(X_train, X_valid,
                         alpha_corr=(0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0),
                         alpha_var=(0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0),
                         K = (10, 20, 30, 40, 50, 60, 80, 100),
                         n_splits = 5):
    
    Xtr_val = np.vstack([X_train, X_valid])
    T = Xtr_val.shape[0]
    kfolds = KFold(n_splits=n_splits)

    res = []
    best = None

    for c in alpha_corr:
        for v in alpha_var:
            for k in K:
                print(f"K:{k} alpha_corr:{c} alpha_var:{v}")
                mses = []
                evrs = []

                for idx_tr, idx_val in kfolds.split(Xtr_val):
                    X_tr = Xtr_val[idx_tr]
                    X_val = Xtr_val[idx_val]
                    mean_tr = X_tr.mean(axis=0, keepdims=True)

                    S = sample_cov(X_tr)
                    Ss = shrink_cov(S, c, v)

                    k_min = min(k, X_tr.shape[0]-1, S.shape[0]-1)
                    if k_min < 1:
                        continue

                    V_K, _ = pca_cov(Ss, k_min)
                    Z_val = project_scores(X_val, mean_tr, V_K)
                    Xhat_val = reconstruction(Z_val, mean_tr, V_K)
                    mse = np.mean((X_val - Xhat_val)**2)
                    evr = explained_var(X_val, mean_tr, V_K)
                    mses.append(mse)
                    evrs.append(evr)

                mse_mean = float(np.mean(mses))
                mse_std = float(np.std(mses, ddof=1))

                evr_mean = float(np.mean(evrs))
                evr_std = float(np.std(evrs, ddof=1))

                res.append({
                    "alpha_var":float(v),
                    "alpha_corr":float(c),
                    "K":int(k),
                    "evr_mean":evr_mean,
                    "evr_std":evr_std,
                    "mse_mean":mse_mean,
                    "mse_std":mse_std
                })
                print(res[-1])

                if best is None or mse_mean < best[2]:
                    best = (evr_mean, evr_std, mse_mean, evr_mean, v, c, k)

    evr_mean, evr_std, mse_mean, evr_mean, v_hat, c_hat, k_hat = best
    best_params = {
        "alpha_var":float(v_hat),
        "alpha_corr":float(c_hat),
        "K":int(k_hat),
        "evr_mean":evr_mean,
        "evr_std":evr_std,
        "mse_mean":mse_mean,
        "evr_mean":evr_mean
    }

    return best_params, res

#best_params, res = tune_hyperparameters(X_train, X_valid)

# We plotted the 3d surface fixed on K_hat (K=100) 
d = load()

rows = [i for i in d if i["K"]==100 and i["alpha_corr"]]

ac = sorted({float(row["alpha_corr"]) for row in rows})
av = sorted({float(row["alpha_var"]) for row in rows})

i_var = {v:i for i, v in enumerate(av)}
j_corr = {c:j for j, c in enumerate(ac)}
Z = np.full((len(av), len(ac)), np.nan)

for row in rows:
    i = i_var[float(row["alpha_var"])]
    j = j_corr[float(row["alpha_corr"])]
    Z[i, j] = float(row["mse_mean"])

AC, AV = np.meshgrid(ac, av) # columns x rows

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(AC, AV, Z, cmap="viridis", linewidth=0, antialiased=True)
ax.set_xlabel("alpha_corr")
ax.set_ylabel("alpha_var")
ax.set_zlabel("MSE")
plt.tight_layout()
plt.show()