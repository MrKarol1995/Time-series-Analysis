import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Parameters
n = 1000  # Sample size
beta_0_true = 2  # True intercept
beta_1_true = 5  # True slope
sigma = 2  # Standard deviation of errors
num_simulations = 1000  # Number of simulations for Monte Carlo

# Containers for storing estimates
beta_0_estimates = []
beta_1_estimates = []

# Monte Carlo Simulation
for _ in range(num_simulations):
    # Generate X and epsilon
    X = np.random.uniform(0, 10, n)
    epsilon = np.random.normal(0, sigma ** 2, n)  # Use sigma instead of sigma^2 as it's SD

    # Generate Y using the linear model
    Y = beta_0_true + beta_1_true * X + epsilon

    # Estimate beta_0 and beta_1 using OLS
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # OLS formula for beta_1 and beta_0
    beta_1_hat = np.sum(X * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
    beta_0_hat = Y_mean - beta_1_hat * X_mean

    # Store the estimates
    beta_0_estimates.append(beta_0_hat)
    beta_1_estimates.append(beta_1_hat)

# Convert lists to arrays for analysis
beta_0_estimates = np.array(beta_0_estimates)
beta_1_estimates = np.array(beta_1_estimates)

# Calculate the sample mean and standard deviation of the estimates
beta_0_mean = np.mean(beta_0_estimates)
beta_0_std = np.std(beta_0_estimates)
beta_1_mean = np.mean(beta_1_estimates)
beta_1_std = np.std(beta_1_estimates)

# Create plots to compare empirical and theoretical distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set(style="whitegrid")

# Plot for beta_0 estimates: Empirical CDF vs. Theoretical CDF
sns.ecdfplot(beta_0_estimates, ax=axes[0, 0], label='Empirical CDF')
x = np.linspace(beta_0_mean - 3 * beta_0_std, beta_0_mean + 3 * beta_0_std, 1000)
axes[0, 0].plot(x, norm.cdf(x, beta_0_mean, beta_0_std), 'r--', label='Theoretical CDF')
axes[0, 0].set_title(r'$\beta_0$: Empirical vs Theoretical CDF')
axes[0, 0].legend()

# Plot for beta_1 estimates: Empirical CDF vs. Theoretical CDF
sns.ecdfplot(beta_1_estimates, ax=axes[0, 1], label='Empirical CDF')
x = np.linspace(beta_1_mean - 3 * beta_1_std, beta_1_mean + 3 * beta_1_std, 1000)
axes[0, 1].plot(x, norm.cdf(x, beta_1_mean, beta_1_std), 'r--', label='Theoretical CDF')
axes[0, 1].set_title(r'$\beta_1$: Empirical vs Theoretical CDF')
axes[0, 1].legend()

# Plot for beta_0 estimates: Empirical Density vs. Theoretical Density
sns.kdeplot(beta_0_estimates, ax=axes[1, 0], label='Empirical Density')
x = np.linspace(beta_0_mean - 3 * beta_0_std, beta_0_mean + 3 * beta_0_std, 1000)
axes[1, 0].plot(x, norm.pdf(x, beta_0_mean, beta_0_std), 'r--', label='Theoretical Density')
axes[1, 0].set_title(r'$\beta_0$: Empirical vs Theoretical Density')
axes[1, 0].legend()

# Plot for beta_1 estimates: Empirical Density vs. Theoretical Density
sns.kdeplot(beta_1_estimates, ax=axes[1, 1], label='Empirical Density')
x = np.linspace(beta_1_mean - 3 * beta_1_std, beta_1_mean + 3 * beta_1_std, 1000)
axes[1, 1].plot(x, norm.pdf(x, beta_1_mean, beta_1_std), 'r--', label='Theoretical Density')
axes[1, 1].set_title(r'$\beta_1$: Empirical vs Theoretical Density')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

#zad 2

# Parametry symulacji
np.random.seed(0)
beta_true = 2.0  # Założona prawdziwa wartość beta1
num_simulations = 1000  # Liczba powtórzeń Monte Carlo
sigma_values = range(5, 101, 5)  # Zakres wartości sigma
n_values = range(50, 1001, 50)  # Różne wielkości próby n


# Funkcja do przeprowadzenia symulacji
def monte_carlo_simulation(n, sigma):
    betas_estimates = []

    for _ in range(num_simulations):
        # Generowanie danych losowych x i błędów epsilon
        x = np.random.uniform(0, 10, n)
        epsilon = np.random.normal(0, sigma, n)
        y = beta_true * x + epsilon  # Generowanie zmiennej zależnej y

        # Estymacja beta1 przy użyciu wzoru na współczynnik regresji
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        beta_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        betas_estimates.append(beta_hat)

    # Obliczenie wartości oczekiwanej i wariancji estymatora beta1
    mean_beta = np.mean(betas_estimates)
    var_beta = np.var(betas_estimates)
    return mean_beta, var_beta


# Przeprowadzenie symulacji dla różnych wartości n i sigma
results = {}
for n in n_values:
    results[n] = {"sigma": [], "mean_beta": [], "var_beta": []}
    for sigma in sigma_values:
        mean_beta, var_beta = monte_carlo_simulation(n, sigma)
        results[n]["sigma"].append(sigma)
        results[n]["mean_beta"].append(mean_beta)
        results[n]["var_beta"].append(var_beta)

# Tworzenie wykresu
plt.figure(figsize=(16, 8))

# Tworzenie wykresów dla każdego n
for n in n_values:
    plt.plot(results[n]["sigma"], results[n]["var_beta"], label=f'n = {n}', marker='o')

# Opisy osi i tytuł
plt.xlabel('Wariancja błędu σ', fontsize=14)
plt.ylabel('Wariancja estymatora β̂1', fontsize=14)
plt.title('Wariancja estymatora β̂1 w zależności od σ dla różnych wartości n', fontsize=16)
plt.legend(title="Rozmiar próby n")
plt.grid(True)
plt.show()