import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parametry modelu
n = 1000  # liczba obserwacji
b0 = 5    # wyraz wolny
b1 = 2    # współczynnik nachylenia
x = np.linspace(0, 10, 1000)  # wartości zmiennej niezależnej x

# Funkcja estymacji współczynnika nachylenia b1
def beta_1(x, y):
    return np.sum(x * y) / np.sum(x ** 2)

# Parametry symulacji Monte Carlo
mc = 5000  # liczba iteracji
sigmas = np.linspace(1, 50, 50)  # różne wartości odchyleń standardowych

# Funkcja estymacji współczynnika b1 dla różnych wartości sigma
def estymacja_mc3_sigma(mc, n, sigmas, b1):
    b1s = []  # estymacje b1 dla różnych sigm
    b1s_teo_var = []  # teoretyczna wariancja estymatora b1
    for sigma in sigmas:
        b1_mc = []
        for i in range(mc):
            blad = np.random.normal(0, sigma, n)  # losowe błędy
            y = b1 * x + blad  # wartości y zależne od x, b1 i błędu
            b1_e = beta_1(x, y)  # estymator b1
            b1_mc.append(b1_e)
        b1s_teo_var.append(sigma**2 / np.sum(x**2))  # teoretyczna wariancja b1
        b1s.append(b1_mc)
    b1_mean = [np.mean(b1_) for b1_ in b1s]  # średnie estymaty b1
    b1_var = [np.var(b1_) for b1_ in b1s]  # wariancja estymatów b1
    return b1_mean, b1_var, b1s_teo_var

# Wywołanie funkcji estymacji dla różnych sigma
b1_mean, b1_var, b1s_teo_var = estymacja_mc3_sigma(mc, n, sigmas, b1)

# Wykres średnich estymacji b1 w funkcji sigma
plt.plot(sigmas, b1_mean, label="Średnia estymowana b1")
plt.hlines(b1, 0, 50, colors="red", linestyles="dashed", label="Wartość teoretyczna b1")
plt.xlabel("Sigma (odchylenie standardowe)")
plt.ylabel("Średnia estymacji b1")
plt.title("Średnia estymacji b1 w funkcji sigma")
plt.legend()
plt.show()

# Wykres wariancji estymacji b1 oraz teoretycznej wariancji w funkcji sigma
plt.plot(sigmas, b1_var, label="Estymowana wariancja b1")
plt.plot(sigmas, b1s_teo_var, label="Teoretyczna wariancja b1", linestyle="dashed")
plt.xlabel("Sigma (odchylenie standardowe)")
plt.ylabel("Wariancja estymacji b1")
plt.title("Wariancja estymacji b1 w funkcji sigma")
plt.legend()
plt.show()

# Parametry symulacji Monte Carlo dla różnych rozmiarów próby
sigma = 5  # odchylenie standardowe błędów
ns = np.linspace(100, 1000, 18)  # różne rozmiary próby

# Funkcja estymacji współczynnika b1 dla różnych rozmiarów próby
def estymacja_mc3_n(mc, ns, sigma, b1):
    b1s = []
    b1s_teo_var = []
    for n in ns:
        b1_mc = []
        for i in range(mc):
            x = np.linspace(0, 10, int(n))  # wartości x dla danej wielkości próby
            blad = np.random.normal(0, sigma, int(n))  # losowe błędy
            y = b1 * x + blad  # wartości y zależne od x, b1 i błędu
            b1_e = beta_1(x, y)  # estymator b1
            b1_mc.append(b1_e)
        b1s_teo_var.append(sigma**2 / np.sum(x**2))  # teoretyczna wariancja b1
        b1s.append(b1_mc)
    b1_mean = [np.mean(b1_) for b1_ in b1s]  # średnie estymaty b1
    b1_var = [np.var(b1_) for b1_ in b1s]  # wariancja estymatów b1
    return b1_mean, b1_var, b1s_teo_var

# Wywołanie funkcji estymacji dla różnych rozmiarów próby
b1_mean, b1_var, b1s_teo_var = estymacja_mc3_n(mc, ns, sigma, b1)

# Wykres średnich estymacji b1 w funkcji rozmiaru próby n
plt.plot(ns, b1_mean, label="Średnia estymowana b1")
plt.hlines(b1, min(ns), max(ns), colors="red", linestyles="dashed", label="Wartość teoretyczna b1")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Średnia estymacji b1")
plt.title("Średnia estymacji b1 w funkcji rozmiaru próby")
plt.legend()
plt.show()

# Wykres wariancji estymacji b1 oraz teoretycznej wariancji w funkcji rozmiaru próby n
plt.plot(ns, b1_var, label="Estymowana wariancja b1")
plt.plot(ns, b1s_teo_var, label="Teoretyczna wariancja b1", linestyle="dashed")
plt.xlabel("Rozmiar próby (n)")
plt.ylabel("Wariancja estymacji b1")
plt.title("Wariancja estymacji b1 w funkcji rozmiaru próby")
plt.legend()
plt.show()