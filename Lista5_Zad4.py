import numpy as np
import matplotlib.pyplot as plt

def generate_ma1_process(theta, sigma, n):
    """
    Generuje szereg MA(1) o zadanym współczynniku theta i wariancji sigma^2.
    """
    epsilon = np.random.normal(0, sigma, n+1)  # n+1, aby uwzględnić Y_{t-1}
    Y = np.zeros(n)
    for t in range(1, n+1):
        Y[t-1] = epsilon[t] + theta * epsilon[t-1]
    return Y

def generate_time_series(a, b, A, f, theta, sigma, n):
    """
    Generuje szereg czasowy X_t = m(t) + s(t) + Y_t.
    - m(t) = a + b * t (trend liniowy),
    - s(t) = A * sin(2 * pi * f * t) (funkcja sinusoidalna),
    - Y_t to proces MA(1).
    """
    t = np.arange(n)
    m_t = a + b * t                     # Składowa liniowa
    s_t = A * np.sin(2 * np.pi * f * t) # Składowa sinusoidalna
    Y_t = generate_ma1_process(theta, sigma, n)  # Szum MA(1)
    X_t = m_t + s_t + Y_t
    return t, X_t, m_t, s_t, Y_t

def remove_deterministic_components(X_t, m_t, s_t):
    """
    Usuwa składowe deterministyczne m(t) i s(t) ze szeregu X_t.
    """
    return X_t - m_t - s_t

def autocovariance(x, lag):
    """
    Oblicza autokowariancję dla zadanego szeregu x i opóźnienia lag.
    """
    n = len(x)
    x_mean = np.mean(x)
    return np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / n

def compute_autocovariance_function(x, max_lag):
    """
    Oblicza funkcję autokowariancji dla opóźnień od 0 do max_lag.
    """
    return [autocovariance(x, lag) for lag in range(max_lag+1)]

# Parametry modelu
n = 500           # Długość szeregu
a, b = 0, 0.01    # Parametry trendu liniowego m(t) = a + b*t
A, f = 1, 0.05    # Parametry funkcji sinusoidalnej s(t) = A * sin(2*pi*f*t)
theta = 0.5       # Parametr MA(1)
sigma = 1.0       # Odchylenie standardowe szumu

# Generowanie szeregu czasowego
t, X_t, m_t, s_t, Y_t = generate_time_series(a, b, A, f, theta, sigma, n)

# Usunięcie składowych deterministycznych
X_t_detrended = remove_deterministic_components(X_t, m_t, s_t)

# Obliczenie funkcji autokowariancji
max_lag = 20
autocov_X = compute_autocovariance_function(X_t_detrended, max_lag)
autocov_Y = compute_autocovariance_function(Y_t, max_lag)

# Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, X_t, label="X(t)")
plt.plot(t, m_t + s_t, label="Deterministic Components m(t) + s(t)")
plt.title("Szereg czasowy X(t) i składowe deterministyczne")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, X_t_detrended, label="X(t) - m(t) - s(t)", color='green')
plt.title("Szereg czasowy po usunięciu składowych deterministycznych")
plt.legend()
plt.tight_layout()
plt.show()

# Porównanie funkcji autokowariancji
lags = np.arange(max_lag+1)
plt.figure(figsize=(8, 6))
plt.stem(lags, autocov_X, linefmt='b-', markerfmt='bo', basefmt='k-', label='Empiryczna ACV X_t')
plt.stem(lags, autocov_Y, linefmt='r--', markerfmt='ro', basefmt='k-', label='ACV MA(1)')
plt.xlabel("Lag")
plt.ylabel("Autokowariancja")
plt.title("Porównanie funkcji autokowariancji")
plt.legend()
plt.show()