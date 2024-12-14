import numpy as np
import matplotlib.pyplot as plt

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
plt.title("Model X_t")

plt.subplot(2, 1, 2)
plt.plot(lags, autocov_detrended, label="Empiryczna autokowariancja X_detrended")
plt.stem(lags, autocov_MA1, linefmt='r-', markerfmt='ro', basefmt=' ', label="Teoretyczna autokowariancja MA(1)")
plt.xlabel("Lag")
plt.ylabel("Autokowariancja")
plt.legend()
plt.title("Porównanie autokowariancji")
plt.tight_layout()
plt.show()