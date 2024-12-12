import numpy as np
import matplotlib.pyplot as plt

# Parametry
n = 1000
sigma_X = 1
sigma_W = 0.5
theta = 2


# Funkcje pomocnicze
def autokowariancja(h, x):
    n = len(x)
    mean_x = np.mean(x)
    return 1 / n * np.sum((x[:n - abs(h)] - mean_x) * (x[abs(h):] - mean_x))


def autokorelacja(h, x):
    return autokowariancja(h, x) / autokowariancja(0, x)


def x_sample(n, sigma, theta):
    z = np.random.normal(0, sigma, n + 1)
    x = np.zeros(n)
    for i in range(1, n + 1):
        x[i - 1] = z[i] + theta * z[i - 1]
    return x


def y_sample(n, sigma_z, theta, sigma_w):
    x = x_sample(n, sigma_z, theta)
    w = np.random.normal(0, sigma_w, n)
    return x + w


def teo_autokow(h, sigma_z, sigma_w, theta):
    if h == 0:
        return sigma_z ** 2 * (theta ** 2 + 1) + sigma_w ** 2
    elif h == 1 or h == -1:
        return theta * sigma_z ** 2
    else:
        return 0


def teo_autCOR(h, sigma_z, sigma_w, theta):
    if h == 0:
        return 1
    elif h == 1 or h == -1:
        return (sigma_z ** 2 * theta) / (sigma_z ** 2 * (theta ** 2 + 1) + sigma_w ** 2)
    else:
        return 0


# Przygotowanie danych
hs = np.arange(0, 11, 1)
autokow_emp, autokow_teo = [], []
autocorr_emp, autocorr_teo = [], []

y = y_sample(n, sigma_X, theta, sigma_W)

for h in hs:
    # Empiryczne autokowariancje i autokorelacje
    autokow_emp.append(autokowariancja(h, y))
    autocorr_emp.append(autokorelacja(h, y))

    # Teoretyczne autokowariancje i autokorelacje
    autokow_teo.append(teo_autokow(h, sigma_X, sigma_W, theta))
    autocorr_teo.append(teo_autCOR(h, sigma_X, sigma_W, theta))

# Wykresy
plt.figure(figsize=(12, 6))

# Wykres autokowariancji
plt.subplot(2, 1, 1)
plt.plot(hs, autokow_teo, "ro", label="Teoretyczna autokowariancja")
plt.plot(hs, autokow_emp, label="Empiryczna autokowariancja")
plt.title("Funkcja autokowariancji")
plt.xlabel("Opóźnienie (lag)")
plt.ylabel("Autokowariancja")
plt.legend(loc="best")
plt.grid(True)

# Wykres autokorelacji
plt.subplot(2, 1, 2)
plt.plot(hs, autocorr_teo, "ro", label="Teoretyczna autokorelacja")
plt.plot(hs, autocorr_emp, label="Empiryczna autokorelacja")
plt.title("Funkcja autokorelacji")
plt.xlabel("Opóźnienie (lag)")
plt.ylabel("Autokorelacja")
plt.legend(loc="best")
plt.grid(True)

plt.tight_layout()
plt.show()