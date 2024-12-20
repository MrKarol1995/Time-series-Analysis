import numpy as np
import matplotlib.pyplot as plt

# Zad 1
# Parametry modelu
phi = 0.2
sigma2 = 0.4
sigma = np.sqrt(sigma2)
X0 = 0
n = 1000  # długość trajektorii
h_max = 50  # maksymalna liczba lagów

# Symulacja trajektorii AR(1)
Z = np.random.normal(0, sigma, size=n)
X = np.zeros(n)
X[0] = X0

for t in range(1, n):
    X[t] = phi * X[t - 1] + Z[t]

# Obliczanie empirycznej funkcji autokowariancji
def empirical_autocovariance(series, lag):
    n = len(series)
    mean = np.mean(series)
    return np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / n

empirical_gamma = [empirical_autocovariance(X, lag) for lag in range(h_max + 1)]

# Obliczanie teoretycznej funkcji autokowariancji
theoretical_gamma = [(sigma2 * phi**np.abs(lag)) / (1 - phi**2) for lag in range(h_max + 1)]

# Obliczanie empirycznej i teoretycznej autokorelacji
empirical_rho = [g / empirical_gamma[0] for g in empirical_gamma]
theoretical_rho = [phi**np.abs(lag) for lag in range(h_max + 1)]

# Wykresy porównawcze
lags = np.arange(h_max + 1)


# ZAD 2
n_sim = 1000  # liczba symulacji

# Symulacja Monte Carlo
autocovariances_sim = np.zeros((n_sim, h_max + 1))
autocorrelations_sim = np.zeros((n_sim, h_max + 1))


for sim in range(n_sim):
    # Generowanie trajektorii AR(1)
    Z = np.random.normal(0, sigma, size=n)
    X = np.zeros(n)
    X[0] = X0
    for t in range(1, n):
        X[t] = phi * X[t - 1] + Z[t]

    # Obliczanie autokowariancji i autokorelacji
    gamma = [empirical_autocovariance(X, lag) for lag in range(h_max + 1)]
    rho = [g / gamma[0] for g in gamma]

    autocovariances_sim[sim, :] = gamma
    autocorrelations_sim[sim, :] = rho

# Obliczanie przedziałów ufności (95%)
lower_bound_cov = np.percentile(autocovariances_sim, 5, axis=0)
upper_bound_cov = np.percentile(autocovariances_sim, 95, axis=0)

lower_bound_corr = np.percentile(autocorrelations_sim, 5, axis=0)
upper_bound_corr = np.percentile(autocorrelations_sim, 95, axis=0)


# Wykresy porównawcze
lags = np.arange(h_max + 1)

plt.figure(figsize=(14, 7))
# Autokowariancja
plt.subplot(1, 2, 1)
plt.fill_between(lags, lower_bound_cov, upper_bound_cov, color='lightblue', alpha=0.5, label='95% przedział ufności')
plt.plot(lags, empirical_gamma, 'o-', label='Empiryczna autokowariancja')
plt.plot(lags, theoretical_gamma, 'x-', label='Teoretyczna autokowariancja')
plt.title('Porównanie autokowariancji')
plt.xlabel('Lag (h)')
plt.ylabel('Autokowariancja')
plt.grid(True)
plt.legend()

# Autokorelacja
plt.subplot(1, 2, 2)
plt.fill_between(lags, lower_bound_corr, upper_bound_corr, color='lightblue', alpha=0.5, label='95% przedział ufności')
plt.plot(lags, empirical_rho, 'o-', label='Empiryczna autokorelacja')
plt.plot(lags, theoretical_rho, 'x-', label='Teoretyczna autokorelacja')
plt.title('Porównanie autokorelacji')
plt.xlabel('Lag (h)')
plt.ylabel('Autokorelacja')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# cześc druga

# Parametry modelu
np.random.seed(42)
n = 500  # liczba obserwacji
a0, a1 = 2, 0.5  # współczynniki wielomianu m(t) = a0 + a1 * t
omega = 2 * np.pi / 50  # częstotliwość dla s(t)
theta = 0.6  # parametr MA(1)
sigma = 1.0  # odchylenie standardowe szumu

# Generowanie modelu deterministycznego
t = np.arange(n)
m_t = a0 + a1 * t
s_t = np.sin(omega * t)  # przykład funkcji okresowej

# Generowanie szeregu MA(1)
epsilon = np.random.normal(0, sigma, n)
Y_t = epsilon[1:] + theta * epsilon[:-1]
Y_t = np.concatenate([[0], Y_t])  # dopasowanie rozmiaru

# Model X_t
X_t = m_t + s_t + Y_t

# Funkcja autokowariancji
def autocovariance(series, lag):
    mean = np.mean(series)
    return np.mean((series[:-lag] - mean) * (series[lag:] - mean)) if lag > 0 else np.var(series)

lags = np.arange(20)
autocov_X = [autocovariance(X_t, lag) for lag in lags]

# Usuwanie składowych deterministycznych
X_detrended = X_t - m_t - s_t

# Empiryczna funkcja autokowariancji dla X_detrended
autocov_detrended = [autocovariance(X_detrended, lag) for lag in lags]

# Teoretyczna funkcja autokowariancji dla MA(1)
autocov_MA1 = [sigma**2 * (1 + theta**2) if lag == 0 else sigma**2 * theta if lag == 1 else 0 for lag in lags]

# Wizualizacja
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, X_t, label="X_t")
plt.plot(t, m_t, label="m(t)")
plt.plot(t, s_t, label="s(t)")
plt.legend()
plt.grid(True)
plt.title("Model X_t")

plt.subplot(2, 1, 2)
plt.plot(lags, autocov_detrended, label="Empiryczna autokowariancja X_detrended")
plt.stem(lags, autocov_MA1, linefmt='r-', markerfmt='ro', basefmt=' ', label="Teoretyczna autokowariancja MA(1)")
plt.xlabel("Lag")
plt.ylabel("Autokowariancja")
plt.legend()
plt.title("Porównanie autokowariancji")
plt.tight_layout()
plt.grid(True)
plt.show()