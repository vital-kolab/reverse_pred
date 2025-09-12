from scipy.stats import shapiro, wilcoxon, ttest_ind, ttest_rel, ranksums, spearmanr, pearsonr
import numpy as np

def test_normality(d1, d2):
    stats1, p1 = shapiro(d1)
    stats2, p2 = shapiro(d2)
    if p1 < 0.05 or p2 < 0.05:
        return 0 #not normal
    else:
        return 1
    
def print_paired_test(d1, d2, alternative="two-sided", verbose=True):
    if test_normality(d1, d2):
        stat, p = ttest_rel(d1, d2, alternative=alternative)
        dof = len(d1) - 1
        if verbose:
            if p < 0.001:
                print(f"t({dof}) = {stat}, p < 0.001")
            else:
                print(f"t({dof}) = {stat:.3f}, p = {p:.3f}")

    else:
        stat, p = wilcoxon(d1, d2, alternative=alternative)
        if verbose:
            if p < 0.001:
                print(f"z = {stat}, p < 0.001")
            else:
                print(f"z = {stat:.3f}, p = {p:.3f}")
    return p

def print_unpaired_test(d1, d2, alternative="two-sided", verbose=True):
    if test_normality(d1, d2):
        stat, p = ttest_ind(d1, d2, alternative=alternative)
        dof = len(d1) + len(d2) - 2
        if verbose:
            if p < 0.001:
                print(f"t({dof}) = {stat}, p < 0.001")
            else:
                print(f"t({dof}) = {stat:.3f}, p = {p:.3f}")

    else:
        stat, p = ranksums(d1, d2, alternative=alternative)
        if verbose:
            if p < 0.001:
                print(f"z = {stat:.3f}, p < 0.001")
            else:
                print(f"z = {stat:.3f}, p = {p:.3f}")

    return p

def permutation_test(
    x,
    y,
    stat_func=None,
    n_permutations=10_000,
    alternative="two-sided",
    random_state=None,
    nan_policy="omit",
    verbose=True
):
    x = np.asarray(x)
    y = np.asarray(y)

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    if nan_policy == "omit":
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
    elif np.isnan(x).any() or np.isnan(y).any():
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "null_dist": np.full(n_permutations, np.nan),
            "alternative": alternative,
            "n_permutations": n_permutations,
        }

    if stat_func is None:
        stat_func = lambda a, b: np.mean(a) - np.mean(b)

    # observed statistic
    stat_obs = float(stat_func(x, y))

    # RNG
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # build null by shuffling labels
    pooled = np.concatenate([x, y])
    nx = x.size
    null = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        perm = rng.permutation(pooled)
        x_perm = perm[:nx]
        y_perm = perm[nx:]
        null[i] = stat_func(x_perm, y_perm)

    # p-value with +1 correction to avoid zeros
    if alternative == "two-sided":
        extreme = np.sum(np.abs(null) >= abs(stat_obs))
    elif alternative == "greater":
        extreme = np.sum(null >= stat_obs)
    elif alternative == "less":
        extreme = np.sum(null <= stat_obs)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")

    pval = (extreme + 1) / (n_permutations + 1)

    if verbose:
        if pval < 0.001:
                print(f"D = {stat_obs:.3f}, p < 0.001")
        else:
            print(f"D = {stat_obs:.3f}, p = {pval:.3f}")

    return {
        "stat": stat_obs,
        "pvalue": pval,
        "null_dist": null,
        "alternative": alternative,
        "n_permutations": n_permutations,
    }
        


def permutation_test_corr_diff(d1, d2, d3, n_permutations=10000, seed=None, verbose=True):
    rng = np.random.default_rng(seed)

    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    d3 = np.asarray(d3)

    # observed difference
    corr1 = pearsonr(d3, d1)[0]
    corr2 = pearsonr(d3, d2)[0]
    observed_diff = corr1 - corr2

    N = len(d1)
    perm_diffs = np.empty(n_permutations)

    for i in range(n_permutations):
        
        choice_mask1 = rng.integers(0, 2, size=N).astype(bool)
        choice_mask2 = rng.integers(0, 2, size=N).astype(bool)
        perm_d1 = np.where(choice_mask1, d1, d2)
        perm_d2 = np.where(choice_mask2, d2, d1)

        c1 = pearsonr(d3, perm_d1)[0]
        c2 = pearsonr(d3, perm_d2)[0]
        perm_diffs[i] = c1 - c2

    # two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= abs(observed_diff))

    if verbose:
        if p_value < 0.001:
                print(f"D = {observed_diff:.3f}, p < 0.001")
        else:
            print(f"D = {observed_diff:.3f}, p = {p_value:.3f}")

    return p_value, observed_diff, perm_diffs


def nancorr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)
    
    # Mask NaNs from either array
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if method == "pearson":
        r, p = pearsonr(x_clean, y_clean)
        if p < 0.001:
            print(f"r({N-1}) = {r:.2f}, p = < 0.001")
        else:
            print(f"r({N-1}) = {r:.2f}, p = {p:.3f}")
    elif method == "spearman":
        r, p = spearmanr(x_clean, y_clean)
        if p < 0.001:
            print(f"r({N-1}) = {r:.2f}, p = < 0.001")
        else:
            print(f"r({N-1}) = {r:.2f}, p = {p:.3f}")
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
    return r,p
