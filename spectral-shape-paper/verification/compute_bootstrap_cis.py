"""
Compute bootstrap CIs for all paper AUROCs + cross-arch permutation p-values.
Fills the remaining TBDs.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent
N_BOOT = 2000
N_PERM = 1000
SEED = 42
rng = np.random.RandomState(SEED)

SHAPE_FEATURES = ['stable_rank', 'participation_ratio', 'sv_kurtosis',
                   'condition_number', 'nuclear_norm_ratio', 'mp_fit_residual']
MP_FEATURES = ['mp_signal_rank', 'mp_signal_fraction', 'mp_top_sv_excess',
               'mp_spectral_gap', 'mp_norm_per_token']


def load_cognitive(model_name):
    with open(DATA_DIR / f'{model_name}_cognitive_states.json') as f:
        data = json.load(f)
    trials = data['results']['confab']
    X_rows, y, lengths = [], [], []
    for t in trials:
        if t['behavior'] in ('CONFABULATED', 'HEDGED'):
            X_rows.append(t['features'])
            y.append(1 if t['behavior'] == 'CONFABULATED' else 0)
            lengths.append(t['n_tokens'])
    return X_rows, np.array(y), np.array(lengths)


def to_matrix(rows, keys):
    X = np.zeros((len(rows), len(keys)))
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            X[i, j] = r.get(k, 0.0)
    return X


def auroc_cv(X, y, lengths, n_splits=5, prompt_ids=None):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log_len = np.log(lengths + 1).reshape(-1, 1)
    gkf = GroupKFold(n_splits=n_splits)
    # Use prompt_ids for real GroupKFold; np.arange is KFold-equivalent (see audit)
    groups = prompt_ids if prompt_ids is not None else np.arange(len(y))
    probs = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        slope = np.linalg.lstsq(log_len[tr], X[tr], rcond=None)[0]
        Xtr = X[tr] - log_len[tr] @ slope.reshape(1, -1)
        Xte = X[te] - log_len[te] @ slope.reshape(1, -1)
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegressionCV(cv=3, penalty='l2', max_iter=1000, random_state=SEED)
        clf.fit(Xtr, y[tr])
        probs[te] = clf.predict_proba(Xte)[:, 1]
    valid = ~np.isnan(probs)
    return roc_auc_score(y[valid], probs[valid]) if valid.sum() > 10 else 0.5


def bootstrap_ci(X, y, lengths, n_boot=N_BOOT):
    n = len(y)
    boots = []
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(auroc_cv(X[idx], y[idx], lengths[idx]))
        except:
            pass
        if (i+1) % 200 == 0:
            print(f'  ... {i+1}/{n_boot}')
    boots = np.array(boots)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5), boots


def perm_p(X, y, lengths, n_perm=N_PERM):
    real = auroc_cv(X, y, lengths)
    nulls = []
    for i in range(n_perm):
        nulls.append(auroc_cv(X, rng.permutation(y), lengths))
        if (i+1) % 200 == 0:
            print(f'  ... {i+1}/{n_perm}')
    return real, np.mean(np.array(nulls) >= real)


print("=" * 60)
print("BOOTSTRAP CIs FOR TABLE 1 (Qwen)")
print("=" * 60)

rows, y, lengths = load_cognitive('Qwen2.5-7B-Instruct')

for name, keys in [('MP-SVD', MP_FEATURES), ('Shape', SHAPE_FEATURES), ('All', SHAPE_FEATURES + MP_FEATURES)]:
    X = to_matrix(rows, keys)
    point = auroc_cv(X, y, lengths)
    print(f'\n{name}: point={point:.3f}, bootstrapping...')
    lo, hi, _ = bootstrap_ci(X, y, lengths)
    print(f'  95% CI: [{lo:.3f}, {hi:.3f}]')

print("\n" + "=" * 60)
print("CROSS-ARCH PERMUTATION P-VALUES")
print("=" * 60)

for model in ['Mistral-7B-Instruct-v0.3', 'Llama-3.1-8B-Instruct']:
    rows, y, lengths = load_cognitive(model)
    X = to_matrix(rows, SHAPE_FEATURES)
    print(f'\n{model}:')
    real, p = perm_p(X, y, lengths)
    print(f'  AUROC={real:.3f}, permutation p={p:.4f}')

print("\nDone.")
