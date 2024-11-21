import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Parametry
beta0 = 2
beta1 = 5
sigma_values = [0.01, 0.5, 1]
n_values = range(10, 101, 5)
M = 1000  # Liczba symulacji
alpha = 0.05
alpha2 = alpha * 100

# Funkcja do symulacji (znana lub nieznana sigma)
def monte_carlo_simulation(beta0, beta1, sigma, n, M, known_sigma=True):
    coverage_beta0 = 0
    coverage_beta1 = 0
    lengths_beta0 = []
    lengths_beta1 = []

    for _ in range(M):
        # dane
        x = np.arange(1, n + 1)
        epsilon = np.random.normal(0, sigma, n)
        y = beta0 + beta1 * x + epsilon

        # Wyznaczanie parametrów estymacji
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        b1_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        b0_hat = y_mean - b1_hat * x_mean

        # Wariancje i długości przedziałów
        if known_sigma:
            se_b0 = np.sqrt(sigma**2 * (1 / n + x_mean**2 / np.sum((x - x_mean) ** 2)))
            se_b1 = np.sqrt(sigma**2 / np.sum((x - x_mean) ** 2))
            z_critical = stats.norm.ppf(1 - alpha / 2)
            ci_b0 = (b0_hat - z_critical * se_b0, b0_hat + z_critical * se_b0)
            ci_b1 = (b1_hat - z_critical * se_b1, b1_hat + z_critical * se_b1)
        else:
            residuals = y - (b0_hat + b1_hat * x)
            sigma_squared = np.sum(residuals ** 2) / (n - 2)
            se_b0 = np.sqrt(sigma_squared * (1 / n + x_mean**2 / np.sum((x - x_mean) ** 2)))
            se_b1 = np.sqrt(sigma_squared / np.sum((x - x_mean) ** 2))
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 2)
            ci_b0 = (b0_hat - t_critical * se_b0, b0_hat + t_critical * se_b0)
            ci_b1 = (b1_hat - t_critical * se_b1, b1_hat + t_critical * se_b1)

        # Długości przedziałów
        lengths_beta0.append(ci_b0[1] - ci_b0[0])
        lengths_beta1.append(ci_b1[1] - ci_b1[0])

        # Sprawdzamy, czy prawdziwe beta0 i beta1 są w przedziałach
        if ci_b0[0] <= beta0 <= ci_b0[1]:
            coverage_beta0 += 1
        if ci_b1[0] <= beta1 <= ci_b1[1]:
            coverage_beta1 += 1

    # Wyliczamy procent pokrycia i średnie długości przedziałów
    avg_length_b0 = np.mean(lengths_beta0)
    avg_length_b1 = np.mean(lengths_beta1)
    return (coverage_beta0 / M) * 100, (coverage_beta1 / M) * 100, avg_length_b0, avg_length_b1

# Wyniki symulacji
results = []

for sigma in sigma_values:
    for n in n_values:
        coverage_b0_known, coverage_b1_known, avg_len_b0_known, avg_len_b1_known = monte_carlo_simulation(
            beta0, beta1, sigma, n, M, known_sigma=True
        )
        coverage_b0_unknown, coverage_b1_unknown, avg_len_b0_unknown, avg_len_b1_unknown = monte_carlo_simulation(
            beta0, beta1, sigma, n, M, known_sigma=False
        )
        results.append({
            "Sigma": sigma,
            "n": n,
            "Coverage Beta0 (Known)": coverage_b0_known,
            "Coverage Beta1 (Known)": coverage_b1_known,
            "Coverage Beta0 (Unknown)": coverage_b0_unknown,
            "Coverage Beta1 (Unknown)": coverage_b1_unknown,
            "Avg Length Beta0 (Known)": avg_len_b0_known,
            "Avg Length Beta1 (Known)": avg_len_b1_known,
            "Avg Length Beta0 (Unknown)": avg_len_b0_unknown,
            "Avg Length Beta1 (Unknown)": avg_len_b1_unknown,
        })

# Tworzymy DataFrame z wynikami
df = pd.DataFrame(results)

# Wykresy pokrycia dla Beta0 i Beta1
plt.figure(figsize=(12, 6))
for sigma in sigma_values:
    subset = df[df["Sigma"] == sigma]
    plt.plot(subset["n"], subset["Coverage Beta0 (Known)"], label=f"Beta0 (Known Sigma), Sigma={sigma}", linestyle="-")
    plt.plot(subset["n"], subset["Coverage Beta0 (Unknown)"], label=f"Beta0 (Unknown Sigma), Sigma={sigma}", linestyle="--")

plt.axhline(100 - alpha2, color="red", linestyle="--", label="95% poziom")
plt.title("Pokrycie Beta0 - Znana vs Nieznana Sigma")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Pokrycie (%)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
for sigma in sigma_values:
    subset = df[df["Sigma"] == sigma]
    plt.plot(subset["n"], subset["Coverage Beta1 (Known)"], label=f"Beta1 (Known Sigma), Sigma={sigma}", linestyle="-")
    plt.plot(subset["n"], subset["Coverage Beta1 (Unknown)"], label=f"Beta1 (Unknown Sigma), Sigma={sigma}", linestyle="--")

plt.axhline(100 - alpha2, color="red", linestyle="--", label="95% poziom")
plt.title("Pokrycie Beta1 - Znana vs Nieznana Sigma")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Pokrycie (%)")
plt.legend()
plt.grid()
plt.show()

# Średnie długości przedziałów dla Beta0 i Beta1
plt.figure(figsize=(12, 6))
for sigma in sigma_values:
    subset = df[df["Sigma"] == sigma]
    plt.plot(subset["n"], subset["Avg Length Beta0 (Known)"], label=f"Beta0 (Known Sigma), Sigma={sigma}", linestyle="-")
    plt.plot(subset["n"], subset["Avg Length Beta0 (Unknown)"], label=f"Beta0 (Unknown Sigma), Sigma={sigma}", linestyle="--")

plt.title("Średnia długość przedziału - Beta0")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Średnia długość przedziału")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
for sigma in sigma_values:
    subset = df[df["Sigma"] == sigma]
    plt.plot(subset["n"], subset["Avg Length Beta1 (Known)"], label=f"Beta1 (Known Sigma), Sigma={sigma}", linestyle="-")
    plt.plot(subset["n"], subset["Avg Length Beta1 (Unknown)"], label=f"Beta1 (Unknown Sigma), Sigma={sigma}", linestyle="--")

plt.title("Średnia długość przedziału - Beta1")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Średnia długość przedziału")
plt.legend()
plt.grid()
plt.show()

