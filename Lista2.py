import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.linear_model import LinearRegression


# Zad 1
# Parametry regresji
b0 = 5  # Wyraz wolny
beta1 = 2  # Współczynnik kierunkowy
n = 1000  # Liczba próbek
df = 5  # Liczba stopni swobody (v)

# Zmienna do przechowywania wartości estymowanych b_0 i b_1
b0_hats = []
b1_hats = []

# Liczba symulacji
num_simulations = 1000

# Symulacja
for _ in range(num_simulations):
    # Generowanie epsilon (błędów) z rozkładu t-Studenta
    epsilon = t.rvs(df, size=n)

    # Generowanie wartości x i y
    x = np.linspace(0, 10, num=n).reshape(-1, 1)  # Wartości x od 0 do 10
    y = b0 + beta1 * x + epsilon.reshape(-1, 1)

    # Dopasowanie modelu regresji liniowej
    model = LinearRegression()
    model.fit(x, y)

    # Wyznaczanie estymowanych wartości b_0_hat i b_1_hat
    b0_hats.append(model.intercept_[0])
    b1_hats.append(model.coef_[0][0])

# Wizualizacja danych (x_i, y_i) dla ostatniej symulacji
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label='Dane (x, y)')
plt.plot(x, model.predict(x), color='red', label='Regresja Liniowa')
plt.title('Wykres danych (x, y) z linią regresji')
plt.xlabel('x (zmienna objaśniająca)')
plt.ylabel('y (zmienna objaśniana)')
plt.legend()
plt.grid()
plt.show()

# Boxplot dla b_0_hat i b_1_hat
plt.figure(figsize=(10, 6))
plt.boxplot([b0_hats, b1_hats], labels=['$\\hat{b}_0$', '$\\hat{b}_1$'])
plt.title('Boxplot dla estymacji $\\hat{b}_0$ i $\\hat{b}_1$')
plt.ylabel('Wartości estymatorów')
plt.grid(True)
plt.show()

# Obliczenie i wyświetlenie średnich wartości b_0_hat i b_1_hat
mean_b0_hat = np.mean(b0_hats)
mean_b1_hat = np.mean(b1_hats)
print(f"Średnia wartość beta_0: {mean_b0_hat:.4f}")
print(f"Średnia wartość beta_1: {mean_b1_hat:.4f}")


# Zad 2


# Funkcja do symulacji Monte Carlo
def monte_carlo_simulation(n, sigma, num_simulations=1000):
    beta1 = 2  # Prawdziwa wartość współczynnika beta_1
    beta1_estimates = []

    for _ in range(num_simulations):
        x = np.linspace(0, 10, n).reshape(-1, 1)  # Wartości zmiennej objaśniającej
        epsilon = np.random.normal(0, sigma, n)  # Błędy o rozkładzie N(0, sigma)
        y = beta1 * x.flatten() + epsilon  # Model liniowy

        # Dopasowanie modelu regresji
        model = LinearRegression()
        model.fit(x, y)
        beta1_estimates.append(model.coef_[0])  # Zapamiętaj estymator beta1

    # Obliczenie wartości oczekiwanej i wariancji estymatora beta1
    beta1_mean = np.mean(beta1_estimates)
    beta1_variance = np.var(beta1_estimates)

    return beta1_mean, beta1_variance


# Parametry symulacji
sigma_values = [0.5, 1, 2, 5]  # Różne wartości sigma
n_values = [50, 100, 500, 1000]  # Różne wielkości próby
num_simulations = 1000  # Liczba powtórzeń Monte Carlo

# Przechowywanie wyników
results = {}

# Symulacje dla różnych sigma i n
for sigma in sigma_values:
    for n in n_values:
        beta1_mean, beta1_variance = monte_carlo_simulation(n, sigma, num_simulations)
        results[(n, sigma)] = (beta1_mean, beta1_variance)
        print(f'n={n}, sigma={sigma}: E[β1_hat]={beta1_mean:.4f}, Var[β1_hat]={beta1_variance:.4f}')

# Wizualizacja wyników
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Wartość oczekiwana estymatora β1
means = [results[(n, sigma)][0] for sigma in sigma_values for n in n_values]
ax[0].plot(np.repeat(n_values, len(sigma_values)), means, 'o-')
ax[0].set_title('Wartość oczekiwana estymatora β1')
ax[0].set_xlabel('n')
ax[0].set_ylabel('E[β1_hat]')
ax[0].grid(True)

# Wariancja estymatora β1
variances = [results[(n, sigma)][1] for sigma in sigma_values for n in n_values]
ax[1].plot(np.repeat(n_values, len(sigma_values)), variances, 'o-')
ax[1].set_title('Wariancja estymatora β1')
ax[1].set_xlabel('n')
ax[1].set_ylabel('Var[β1_hat]')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Zad 3

# Parametry modelu regresji
beta0 = 5  # Wyraz wolny
beta1 = 2  # Współczynnik kierunkowy
n = 100  # Liczba próbek
sigma = 1  # Odchylenie standardowe błędu (epsilon)

# Liczba symulacji Monte Carlo
num_simulations = 1000

# Zmienne do przechowywania estymatorów
b0_hats = []
b1_hats = []

# Symulacje Monte Carlo
for _ in range(num_simulations):
    # Generowanie losowych wartości x
    x = np.linspace(0, 10, n).reshape(-1, 1)

    # Generowanie losowych błędów epsilon z rozkładu N(0, sigma)
    epsilon = np.random.normal(0, sigma, size=n).reshape(-1, 1)

    # Obliczanie wartości y zgodnie z modelem
    y = beta0 + beta1 * x + epsilon

    # Dopasowanie modelu regresji liniowej
    model = LinearRegression()
    model.fit(x, y)

    # Zapisanie estymatorów beta0 i beta1
    b0_hats.append(model.intercept_[0])
    b1_hats.append(model.coef_[0][0])

# Wizualizacja wyników
# Boxplot dla b0_hat i b1_hat
plt.figure(figsize=(10, 6))
plt.boxplot([b0_hats, b1_hats], labels=['$\\hat{b}_0$', '$\\hat{b}_1$'])
plt.title('Boxplot dla estymacji $\\hat{b}_0$ i $\\hat{b}_1$')
plt.ylabel('Wartości estymatorów')
plt.grid(True)
plt.show()

# Obliczenie średnich wartości estymatorów
mean_b0_hat = np.mean(b0_hats)
mean_b1_hat = np.mean(b1_hats)

# Porównanie z wartościami teoretycznymi
print(f"Średnia wartość $\\hat{{b}}_0$: {mean_b0_hat:.4f} (wartość teoretyczna: {beta0})")
print(f"Średnia wartość $\\hat{{b}}_1$: {mean_b1_hat:.4f} (wartość teoretyczna: {beta1})")

# Histogram rozkładów estymatorów
plt.figure(figsize=(10, 6))

# Histogram dla b0_hat
plt.subplot(1, 2, 1)
plt.hist(b0_hats, bins=30, color='blue', alpha=0.7)
plt.axvline(beta0, color='red', linestyle='dashed', linewidth=2, label='Teoretyczne $b_0$')
plt.title('Histogram estymatorów $\\hat{b}_0$')
plt.legend()
plt.grid()

# Histogram dla b1_hat
plt.subplot(1, 2, 2)
plt.hist(b1_hats, bins=30, color='green', alpha=0.7)
plt.axvline(beta1, color='red', linestyle='dashed', linewidth=2, label='Teoretyczne $b_1$')
plt.title('Histogram estymatorów $\\hat{b}_1$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Zad 4

# Parametry modelu regresji
beta0 = 5
beta1 = 2
n = 100
sigma = 1
num_simulations = 1000
alpha = 0.05

# Zmienne do przechowywania estymatorów
b0_hats = []
b1_hats = []
b0_se = []  # Standard error for beta_0
b1_se = []  # Standard error for beta_1

# Symulacje Monte Carlo
for _ in range(num_simulations):
    x = np.linspace(0, 10, n).reshape(-1, 1)
    epsilon = np.random.normal(0, sigma, size=n).reshape(-1, 1)
    y = beta0 + beta1 * x + epsilon

    model = LinearRegression()
    model.fit(x, y)

    b0_hats.append(model.intercept_[0])
    b1_hats.append(model.coef_[0][0])

    # Obliczenie standard error
    residuals = y - model.predict(x)
    residual_std = np.std(residuals, ddof=2)
    x_mean = np.mean(x)
    s_xx = np.sum((x - x_mean) ** 2)

    se_b0 = residual_std * np.sqrt(1 / n + (x_mean ** 2) / s_xx)
    se_b1 = residual_std / np.sqrt(s_xx)

    b0_se.append(se_b0)
    b1_se.append(se_b1)

# Studentyzacja estymatorów (znormalizowane wartości estymatorów)
b0_t_stats = [(b0 - beta0) / se for b0, se in zip(b0_hats, b0_se)]
b1_t_stats = [(b1 - beta1) / se for b1, se in zip(b1_hats, b1_se)]

# Porównanie z rozkładem teoretycznym t-Studenta
df = n - 2  # Stopnie swobody
t_values = t.ppf([alpha / 2, 1 - alpha / 2], df)

# Histogram dla znormalizowanych estymatorów
plt.figure(figsize=(10, 6))
plt.hist(b0_t_stats, bins=30, color='blue', alpha=0.7, label='$\\hat{b}_0$')
plt.hist(b1_t_stats, bins=30, color='green', alpha=0.7, label='$\\hat{b}_1$')
plt.axvline(t_values[0], color='red', linestyle='dashed', linewidth=2, label=f'α={alpha} granice teoretyczne')
plt.axvline(t_values[1], color='red', linestyle='dashed', linewidth=2)
plt.title('Rozkład studentyzowanych estymatorów $\\hat{b}_0$ i $\\hat{b}_1$')
plt.legend()
plt.grid()
plt.show()
