import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler

# data
X_train = np.loadtxt("data/arcene_train.data", dtype=np.float64)
X_valid = np.loadtxt("data/arcene_valid.data", dtype=np.float64)
X_test = np.loadtxt("data/arcene_test.data", dtype=np.float64)

# define matirix helper functions

def sample_cov(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (len(X) - 1)

def shrink_cov(S, alpha, target="identity"):
    p = S.shape[0]
    if target == "identity":
        F = (np.trace(S) / p) * np.eye(p)
    elif target == "diagonal":
        F = np.diag(np.diag(S))
    else:
        raise ValueError("Pick identity or diagonal")
    return (1 - alpha) * S + alpha * F

def nll_gaussian(X, mu, Sigma):
    Xc = X - mu
    c, lower = cho_factor(Sigma, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    sol = cho_solve((c, lower), Xc.T, check_finite=False).T
    quad = np.einsum("ij,ij->i", Xc, sol).mean()
    p = X.shape[1]
    return 0.5 * (logdet + quad + p * np.log(2 * np.pi))

# 

def find_alpha_star():
# fit scaler on train dataset only, then transform all splits
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_valid)
    # Xte = scaler.transform(X_test)

    # train statistics
    mu_tr = Xtr.mean(axis=0)
    S_tr = sample_cov(Xtr)

    alphas = np.linspace(0.05, 1.0, 41)
    target = "identity"
    nlls = []
    for a in alphas:
        Sig = shrink_cov(S_tr, a, target)
        nlls.append(nll_gaussian(Xva, mu_tr, Sig))

    # pick optimal alpha by validation NLL
    best_i = int(np.argmin(nlls))
    best_alpha = float(alphas[best_i])

    plt.plot(alphas, nlls, marker="o", linewidth=1)
    plt.axvline(best_alpha, linestyle="--")
    plt.scatter([best_alpha], [nlls[best_i]])
    plt.xlabel("alpha")
    plt.ylabel("Validation NLL")
    plt.title(f"NLL vs alpha")
    plt.tight_layout()
    plt.show()

    return best_alpha

"""
We fix the optimal hyperparameter alpha, retrain the 
model with that alpha (train + val) to get better estimates
and report NLL on test data
"""

def test_score(alpha_star, target="identity"):
    # refit scalar on train+valid 
    scaler = StandardScaler().fit(np.vstack([X_train, X_valid]))
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_valid)
    Xte = scaler.transform(X_test)

    # re estimate mean and covariance 
    Xtrva = np.vstack([Xtr, Xva])
    mu_trva = Xtrva.mean(axis=0)
    S_trva = sample_cov(Xtrva)

    # build shrunk covariance with fixed alpha
    Sig_star = shrink_cov(S_trva, alpha_star, target)

    test_nll = nll_gaussian(Xte, mu_trva, Sig_star)
    return test_nll, mu_trva, Sig_star, scaler

# calculate the uncertainty on test nll via confidence interval

def bootstrap_test_nll(Xte, mu_final, Sig_final, B=300, seed=0):
    # bootstrap for test nll
    rng = np.random.default_rng(seed)
    n = Xte.shape[0]

    # point estimate on original test set
    point = nll_gaussian(Xte, mu_final, Sig_final)

    # bootstrap resamples
    boots = np.empty(B, dtype=float)
    for b in range(B):
        t0 = time.perf_counter()
        # sample w/ replacement to mimic a new test set from the same population
        idx = rng.integers(0, n, size=n)
        Xb = Xte[idx]
        boots[b] = nll_gaussian(Xb, mu_final, Sig_final)

        dt = time.perf_counter() - t0
        print(dt*1000)

    # 95% confidence interval
    lo, hi = np.percentile(boots, [2.5, 97.5])

    plt.figure()
    plt.hist(boots, alpha=0.8, edgecolor="black")
    plt.axvline(point, linestyle="--", linewidth=1, label="point")
    plt.axvline(lo, linestyle=":", linewidth=1, label="lo")
    plt.axvline(hi, linestyle=":", linewidth=1, label="hi")
    plt.xlabel("Test NLL")
    plt.ylabel("Bootstrap frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return point, (lo, hi), boots

alpha_star = find_alpha_star()
point_score, final_mean, final_sigma, scaler = test_score(alpha_star)
print(point_score)
Xte = scaler.transform(X_test)
point, CI, boots = bootstrap_test_nll(Xte, final_mean, final_sigma)