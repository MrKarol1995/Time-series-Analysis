import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt

# Parametry modelu AR(2)
phi1 = 0.2
phi2 = 0.4
sigma2 = 1

# Współczynniki AR i MA
ar = np.array([1, -phi1, -phi2])
ma = np.array([1])

# Tworzenie procesu AR(2)
arma_process = ArmaProcess(ar, ma)

# Generowanie próbek
n_samples = 500
np.random.seed(42)
X = arma_process.generate_sample(nsample=n_samples, scale=np.sqrt(sigma2))

# Funkcja obliczająca empiryczną autokowariancję
def empirical_autocovariance(series, lag):
    n = len(series)
    mean = np.mean(series)
    return np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / n

# Obliczenie empirycznej ACVF
h_max = 20
empirical_gamma = [empirical_autocovariance(X, lag) for lag in range(h_max + 1)]

# Teoretyczna autokowariancja
gamma_0 = sigma2 / (1 - phi1**2 - phi2**2 - 2 * phi1 * phi2)
gamma_1 = phi1 * gamma_0 + phi1 * phi2 * gamma_0
gamma_2 = phi2 * gamma_0

theoretical_gamma = [gamma_0, gamma_1, gamma_2]

# Wyświetlenie wyników
print("Empiryczna ACVF:")
for lag, gamma in enumerate(empirical_gamma[:3]):  # Porównanie do lag 2
    print(f"Lag {lag}: {gamma:.4f}")

print("\nTeoretyczna ACVF:")
for lag, gamma in enumerate(theoretical_gamma):
    print(f"Lag {lag}: {gamma:.4f}")

# Wykres porównawczy
plt.figure(figsize=(10, 6))

# Dodanie empirycznych punktów
plt.stem(range(h_max + 1), empirical_gamma, linefmt="b-", markerfmt="bo", basefmt=" ", label="Empiryczna ACVF")

# Dodanie teoretycznych punktów z przesunięciem -0.1
offset_lags = np.arange(3) - 0.1
plt.stem(offset_lags, theoretical_gamma, linefmt="r-", markerfmt="rs", basefmt=" ", label="Teoretyczna ACVF")

# Tytuł, legenda i osie
plt.title("Porównanie empirycznej i teoretycznej ACVF")
plt.xlabel("Opóźnienie (lag)")
plt.ylabel("Autokowariancja")
plt.xticks(range(h_max + 1))
plt.legend()
plt.grid()
plt.show()