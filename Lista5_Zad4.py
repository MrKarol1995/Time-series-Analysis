import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parametry modelu
t0, t1000 = 0, 10  # Czas od 0 do 10 z krokiem 0.01
t = np.linspace(t0, t1000, 1001)  # Punkty czasu
a1, a2 = 1.5, 5                 # Parametry trendu liniowego m(t)
b1, b2 = 2, 3                   # Parametry sezonowości s(t)
theta = 0.5                     # Parametr MA(1)
sigma2 = 1                      # Wariancja szumu
sigma = np.sqrt(sigma2)         # Odchylenie standardowe szumu
n = len(t)                      # Długość szeregu czasowego

# Funkcja generująca proces MA(1)
def generate_ma1_process(theta, sigma, n):
    epsilon = np.random.normal(0, sigma, n+1)  # n+1 dla epsilon[t-1]
    X = np.zeros(n)
    for i in range(1, n+1):
        X[i-1] = epsilon[i] + theta * epsilon[i-1]
    return X

# Generowanie składowych deterministycznych
m_t = a1 * t + a2                     # Trend liniowy
s_t = b1 * np.sin(b2 * t)             # Sezonowość
X_t = generate_ma1_process(theta, sigma, n)  # Proces MA(1)
Y_t = m_t + s_t + X_t                 # Szereg czasowy

# Usuwanie trendu deterministycznego za pomocą polyfit
coeffs1 = np.polyfit(t, Y_t, 1)  # Dopasowanie trendu liniowego (1. stopnia)
a1_hat, a2_hat = coeffs1
Y_prime_t = Y_t - (a1_hat * t + a2_hat)  # Usunięcie trendu

# Dopasowanie parametrów sezonowości za pomocą curve_fit
def sinusoidal_model(t, b1_hat, b2_hat):
    return b1_hat * np.sin(b2_hat * t)

# Dopasowanie b1_hat i b2_hat
popt, _ = curve_fit(sinusoidal_model, t, Y_prime_t, p0=[2, 3])  # Początkowe zgadywane wartości
b1_hat, b2_hat = popt

# Usuwanie sezonowości z \( Y'(t) \)
Y_double_prime_t = Y_prime_t - sinusoidal_model(t, b1_hat, b2_hat)

# Funkcja do obliczenia autokorelacji
def autocorrelation(x, lag):
    n = len(x)
    x_mean = np.mean(x)
    return np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / np.sum((x - x_mean) ** 2)

# Obliczanie autokorelacji dla Y" i MA(1)
max_lag = 100
autocorr_Y_double_prime = [autocorrelation(Y_double_prime_t, h) for h in range(1, max_lag+1)]
autocorr_MA1 = [(theta**h) / (1 + theta**2) for h in range(1, max_lag+1)]

# Wizualizacje
plt.figure(figsize=(12, 8))

# Wykres szeregu czasowego i jego składowych
plt.subplot(3, 1, 1)
plt.plot(t, Y_t, label="Y(t)")
plt.plot(t, a1_hat * t + a2_hat, label="Linia regresji (trend m(t))", color='green', linestyle=':')
plt.plot(t, m_t + s_t, label="Składowe deterministyczne m(t) + s(t)", linestyle='--')
plt.title("Szereg czasowy Y(t) i składowe deterministyczne")
plt.legend()
plt.grid(True)

# Wykres po usunięciu trendu
plt.subplot(3, 1, 2)
plt.plot(t, Y_prime_t, label="Y'(t)", color='green')
plt.title("Szereg czasowy po usunięciu trendu liniowego")
plt.legend()
plt.grid(True)

# Wykres po usunięciu sezonowości
plt.subplot(3, 1, 3)
plt.plot(t, Y_double_prime_t, label="Y''(t)", color='purple')
plt.title("Szereg czasowy po usunięciu trendu i sezonowości")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Wykres porównania autokorelacji
lags = np.arange(1, max_lag+1)
plt.figure(figsize=(8, 6))
plt.stem(lags, autocorr_Y_double_prime, linefmt='b-', basefmt='k-', label='Autokorelacja Y"(t)')
plt.stem(lags, autocorr_MA1, linefmt='r--', basefmt='k-', label='Teoretyczna AC MA(1)')
plt.xlabel("Lag")
plt.ylabel("Autokorelacja")
plt.title("Porównanie autokorelacji")
plt.legend()
plt.grid(True)
plt.show()

def autokowariancja(h, x):
    n = len(x)
    mean_x = np.mean(x)
    return 1/n * np.sum((x[:n-abs(h)] - mean_x) * (x[abs(h):] - mean_x))
def autokorelacja(h, x):
    return autokowariancja(h, x) / autokowariancja(0,x)


def ma1_teo_acf(h, theta):
    if h == 0:
        return 1
    elif h == 1 or h == -1:
        return theta / (1 + theta**2)
    else:
        return 0

sigma, theta = 0.5, 1
n = 1000
t = np.arange(1,11,0.01)
a1, a2 = 1.5, 5
b1, b2 = 2, 3
def ma1_sample(n, sigma, theta):
    z = np.random.normal(0, sigma, n+1)
    x = np.zeros(n)
    for i in range(1,n+1):
        x[i-1] = z[i] + theta * z[i-1]
    return x
x = ma1_sample(n, sigma, theta)
m_t = lambda t: a1 * t + a2
def s_t(t, b1, b2):
    return b1 * np.sin(b2 * t)
y = m_t(t) + s_t(t, b1, b2) + x
y = m_t(t) + s_t(t, b1, b2) + x
a1_dash, a2_dash = np.polyfit(t,y, 1)
y_dash = lambda t: a1_dash * t + a2_dash
y_star = y - y_dash(t)

b1_dash, b2_dash = curve_fit(s_t, t, y_star,[max(y_star), 2*np.pi/2])[0]
y2_dash = lambda t: b1_dash * np.sin(b2_dash * t)
y_two_stars = y_star - y2_dash(t)

hs = np.arange(0,101,1)

autokor_teo = [ma1_teo_acf(h, theta) for h in hs]
autokor_emp = [autokorelacja(h, y_two_stars) for h in hs]

plt.plot(hs, autokor_teo,"ro", label='Autokorelacja Y"(t)')
plt.plot(hs, autokor_emp,label='Teoretyczna AC MA(1)')
plt.xlabel("Lag")
plt.ylabel("Autokorelacja")
plt.title("Porównanie autokorelacji")
plt.grid(True)
plt.show()