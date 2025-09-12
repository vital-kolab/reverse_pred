from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import median_abs_deviation as mad

def nanmad(var):
    return mad(np.nan_to_num(var))

def print_correlation(var1, var2):
    r, p =pearsonr(var1, var2)
    d = var1.shape[0]-2
    if p < 0.001:
        print(f"r ({d}) = {r:.4f}, p < 0.001")
    else:
        print(f"r ({d}) = {r:.4f}, p = {p:.4f}")

def print_correlation_with_nans(x, y):
    # Remove nan values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Compute Pearson correlation
    print_correlation(x_clean, y_clean)

def permutation_test(population1, population2, num_permutations=10000, alternative='two-sided'):
    """
    Perform a permutation test to determine if two populations are statistically different.

    Parameters:
        population1 (array-like): First population data.
        population2 (array-like): Second population data.
        num_permutations (int): Number of permutations to perform (default=10000).
        alternative (str): Type of test to perform ('two-sided', 'greater', 'less').

    Returns:
        p_value (float): P-value indicating the probability of observing the result under the null hypothesis.
    """
    # Combine the two populations
    combined = np.concatenate([population1, population2])
    
    # Observed difference in means
    observed_diff = np.mean(population1) - np.mean(population2)

    # Initialize a counter for the number of extreme cases
    extreme_count = 0

    # Generate permutations
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        permuted_pop1 = combined[:len(population1)]
        permuted_pop2 = combined[len(population1):]
        permuted_diff = np.mean(permuted_pop1) - np.mean(permuted_pop2)

        if alternative == 'two-sided':
            if abs(permuted_diff) >= abs(observed_diff):
                extreme_count += 1
        elif alternative == 'greater':
            if permuted_diff >= observed_diff:
                extreme_count += 1
        elif alternative == 'less':
            if permuted_diff <= observed_diff:
                extreme_count += 1
        else:
            raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'greater', or 'less'.")

    # Calculate p-value
    p_value = extreme_count / num_permutations

    # Print APA-style report
    direction = "greater than" if alternative == 'greater' else "less than" if alternative == 'less' else "different from"
    if p_value < 0.001:
        print(f"The observed difference ({observed_diff:.3f}) was {direction} expected under the null hypothesis (p < 0.001).")
    else:
        print(f"The observed difference ({observed_diff:.3f}) was {direction} expected under the null hypothesis (p = {p_value:.4f}).")

    return p_value

def journal_figure(do_save=False, filename='figure.eps', dpi=300, size_inches=(2.16, 2.16), linewidth=1):
    """
    Adjusts the current matplotlib figure to make it look publication-worthy.
    
    Parameters:
    - do_save: bool, whether to save the figure to an EPS file.
    - filename: str, the name of the file to save the figure as.
    - dpi: int, the resolution of the figure in dots per inch.
    - size_inches: tuple, the size of the figure in inches.
    - linewidth: float, the line width for the plot elements.
    """
    ax = plt.gca()  # Get the current axes
    
    # Adjust tick direction and length
    ax.tick_params(direction='out', length=10, width=linewidth)
    
    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=2)
    ax.set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
    # Set font size and type
    plt.xticks(fontsize=12) #, fontname='Times New Roman')
    plt.yticks(fontsize=12) #, fontname='Times New Roman')
    
    if do_save:
        # Save the figure
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='eps', linewidth=linewidth)