import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm


def monte_carlo_confidence_intervals(n, sigma, alpha, num_simulations=1000):
    """
    Funkcja przeprowadza symulacje Monte Carlo w celu wyznaczenia przedziałów ufności
    dla parametrów beta_0 i beta_1 modelu regresji liniowej na danym poziomie ufności alpha.

    Parametry:
    - n: rozmiar próby (int)
    - sigma: odchylenie standardowe błędu (float)
    - alpha: poziom istotności dla przedziału ufności (float)
    - num_simulations: liczba symulacji Monte Carlo (int)

    Zwraca:
    - pokrycie_beta_0: prawdopodobieństwo pokrycia przedziału dla beta_0
    - pokrycie_beta_1: prawdopodobieństwo pokrycia przedziału dla beta_1
    """
    # Prawdziwe wartości parametrów
    beta_0 = 2.0
    beta_1 = 3.0

    # Przechowywanie wyników symulacji
    beta_0_in_interval = 0
    beta_1_in_interval = 0

    for _ in range(num_simulations):
        # Generowanie losowych danych x i epsilon
        x = np.arange(1, n + 1)  # x_i = i dla i = 1, 2, ..., n
        epsilon = np.random.normal(0, sigma, n)

        # Generowanie wartości Y według modelu regresji
        y = beta_0 + beta_1 * x + epsilon

        # Dopasowanie modelu OLS
        X = sm.add_constant(x)  # Dodanie wyrazu wolnego
        model = sm.OLS(y, X).fit()

        # Oszacowania beta_0 i beta_1 oraz ich błędy standardowe
        beta_0_hat = model.params[0]
        beta_1_hat = model.params[1]
        se_beta_0 = model.bse[0]
        se_beta_1 = model.bse[1]

        # Wyznaczanie przedziałów ufności dla beta_0 i beta_1
        z_value = norm.ppf(1 - alpha / 2)  # wartość krytyczna z rozkładu normalnego
        ci_beta_0 = (beta_0_hat - z_value * se_beta_0, beta_0_hat + z_value * se_beta_0)
        ci_beta_1 = (beta_1_hat - z_value * se_beta_1, beta_1_hat + z_value * se_beta_1)

        # Sprawdzenie, czy prawdziwe beta_0 i beta_1 leżą w wyznaczonych przedziałach
        if ci_beta_0[0] <= beta_0 <= ci_beta_0[1]:
            beta_0_in_interval += 1
        if ci_beta_1[0] <= beta_1 <= ci_beta_1[1]:
            beta_1_in_interval += 1

    # Obliczenie prawdopodobieństwa pokrycia
    pokrycie_beta_0 = beta_0_in_interval / num_simulations
    pokrycie_beta_1 = beta_1_in_interval / num_simulations

    return pokrycie_beta_0, pokrycie_beta_1


# Parametry do analizy
n_values = [20, 50, 100]  # różne rozmiary próby
sigma_values = [0.01, 0.5, 1.0]  # różne odchylenia standardowe
alpha_values = [0.01, 0.05]  # poziomy istotności

# Przeprowadzenie symulacji i wyświetlenie wyników
results = []

for n in n_values:
    for sigma in sigma_values:
        for alpha in alpha_values:
            pokrycie_beta_0, pokrycie_beta_1 = monte_carlo_confidence_intervals(n, sigma, alpha)
            results.append((n, sigma, alpha, pokrycie_beta_0, pokrycie_beta_1))
            print(f"Rozmiar próby: {n}, Sigma: {sigma}, Alpha: {alpha}")
            print(f"Prawdopodobieństwo pokrycia dla beta_0: {pokrycie_beta_0:.2f}")
            print(f"Prawdopodobieństwo pokrycia dla beta_1: {pokrycie_beta_1:.2f}")
            print("---------------------------------------------------")

# Opcjonalne: Wizualizacja wyników
# Można wykorzystać matplotlib do rysowania wyników pokrycia dla różnych parametrów.