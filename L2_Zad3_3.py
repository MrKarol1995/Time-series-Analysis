import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
import statsmodels.api as sm

# Parameters for the simulation
beta_0 = 2.0  # True intercept
beta_1 = 3.0  # True slope
sigma = 1.0  # Standard deviation of the error term
n = 50  # Sample size
num_simulations = 1000  # Number of Monte Carlo simulations

# Containers to store the results of each simulation
beta_0_estimates = []
beta_1_estimates = []
t_beta_0_estimates = []
t_beta_1_estimates = []

# Monte Carlo simulation
for _ in range(num_simulations):
    # Generate random x and epsilon
    x = np.random.uniform(0, 10, n)
    epsilon = np.random.normal(0, sigma, n)

    # Generate Y based on the linear model
    y = beta_0 + beta_1 * x + epsilon

    # Fit the OLS regression model
    X = sm.add_constant(x)  # Adds intercept term to the model
    model = sm.OLS(y, X).fit()

    # Save the estimates of beta_0 and beta_1
    beta_0_hat = model.params[0]
    beta_1_hat = model.params[1]
    beta_0_estimates.append(beta_0_hat)
    beta_1_estimates.append(beta_1_hat)

    # Save the studentized (t-statistics) estimates
    t_beta_0 = (beta_0_hat - beta_0) / model.bse[0]
    t_beta_1 = (beta_1_hat - beta_1) / model.bse[1]
    t_beta_0_estimates.append(t_beta_0)
    t_beta_1_estimates.append(t_beta_1)

# Plotting the results

# 1. Empirical vs. Theoretical Distribution of beta_0 and beta_1
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(beta_0_estimates, kde=True, stat="density", ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title(r'Empirical Distribution of $\hat{\beta}_0$')
x_vals = np.linspace(min(beta_0_estimates), max(beta_0_estimates), 100)
axes[0, 0].plot(x_vals, norm.pdf(x_vals, loc=beta_0, scale=sigma / np.sqrt(n)), color='red', linestyle='--',
                label="Theoretical")
axes[0, 0].legend()

sns.histplot(beta_1_estimates, kde=True, stat="density", ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title(r'Empirical Distribution of $\hat{\beta}_1$')
x_vals = np.linspace(min(beta_1_estimates), max(beta_1_estimates), 100)
axes[0, 1].plot(x_vals, norm.pdf(x_vals, loc=beta_1, scale=sigma / np.sqrt(np.sum((x - np.mean(x)) ** 2))), color='red',
                linestyle='--', label="Theoretical")
axes[0, 1].legend()

# 2. Empirical vs. Theoretical Distribution of Studentized beta_0 and beta_1
sns.histplot(t_beta_0_estimates, kde=True, stat="density", ax=axes[1, 0], color='salmon')
axes[1, 0].set_title(r'Studentized Distribution of $\hat{\beta}_0$')
x_vals = np.linspace(-4, 4, 100)
axes[1, 0].plot(x_vals, t.pdf(x_vals, df=n - 2), color='blue', linestyle='--', label="Theoretical")
axes[1, 0].legend()

sns.histplot(t_beta_1_estimates, kde=True, stat="density", ax=axes[1, 1], color='purple')
axes[1, 1].set_title(r'Studentized Distribution of $\hat{\beta}_1$')
x_vals = np.linspace(-4, 4, 100)
axes[1, 1].plot(x_vals, t.pdf(x_vals, df=n - 2), color='blue', linestyle='--', label="Theoretical")
axes[1, 1].legend()

plt.tight_layout()
plt.show()