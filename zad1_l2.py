import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Ustawienia
beta0 = 5
beta1 = 2
n = 100  # liczba próbek
iterations = 1000  # liczba iteracji

# Zakres stopni swobody
v_values = np.arange(3, 30, 2)

# Przechowywanie wyników
results = {v: {'beta0': [], 'beta1': []} for v in v_values}

for v in v_values:
    for _ in range(iterations):
        # Generowanie danych
        x = np.random.rand(n)
        epsilon = np.random.standard_t(v, size=n)
        Y = beta0 + beta1 * x + epsilon

        # Dodanie stałej do macierzy X
        X = sm.add_constant(x)

        # Dopasowanie modelu regresji
        model = sm.OLS(Y, X).fit()

        # Przechowywanie estymatorów
        results[v]['beta0'].append(model.params[0])
        results[v]['beta1'].append(model.params[1])

# Obliczanie wartości oczekiwanej i wariancji empirycznej i teoretycznej
expected_values = {}
variances_empirical = {}
variances_theoretical = {}

for v in v_values:
    expected_values[v] = {
        'beta0': np.mean(results[v]['beta0']),
        'beta1': np.mean(results[v]['beta1'])
    }
    variances_empirical[v] = {
        'beta0': np.var(results[v]['beta0']),
        'beta1': np.var(results[v]['beta1'])
    }
    # Teoretyczna wariancja dla rozkładu t Studenta
    variances_theoretical[v] = {
        'beta0': 1 / (n - 2) * (v / (v - 2)),  # Wariancja dla beta0
        'beta1': 1 / (n - 2) * (v / (v - 2)) * np.var(x)  # Wariancja dla beta1
    }

# Wykresy porównawcze
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Wykresy wartości oczekiwanej
axs[0, 0].plot(v_values, [expected_values[v]['beta0'] for v in v_values], label='Beta0 - Oczekiwana', marker='o')
axs[0, 0].plot(v_values, [expected_values[v]['beta1'] for v in v_values], label='Beta1 - Oczekiwana', marker='o')
axs[0, 0].axhline(y=beta0, color='r', linestyle='--', label='Beta0 - Teoretyczna')
axs[0, 0].axhline(y=beta1, color='g', linestyle='--', label='Beta1 - Teoretyczna')
axs[0, 0].set_title('Wartości oczekiwane estymatorów')
axs[0, 0].set_xlabel('Stopnie swobody')
axs[0, 0].set_ylabel('Wartość oczekiwana')
axs[0, 0].legend()

# Wykresy wariancji empirycznej
axs[0, 1].plot(v_values, [variances_empirical[v]['beta0'] for v in v_values], label='Beta0 - Empiryczna', marker='o')
axs[0, 1].plot(v_values, [variances_empirical[v]['beta1'] for v in v_values], label='Beta1 - Empiryczna', marker='o')
axs[0, 1].plot(v_values, [variances_theoretical[v]['beta0'] for v in v_values], label='Beta0 - Teoretyczna',
               linestyle='--')
axs[0, 1].plot(v_values, [variances_theoretical[v]['beta1'] for v in v_values], label='Beta1 - Teoretyczna',
               linestyle='--')
axs[0, 1].set_title('Wariancje estymatorów')
axs[0, 1].set_xlabel('Stopnie swobody')
axs[0, 1].set_ylabel('Wariancja')
axs[0, 1].legend()

# Wykresy porównawcze dla Beta0
axs[1, 0].bar(v_values - 0.2, [variances_empirical[v]['beta0'] for v in v_values], width=0.4,
              label='Beta0 - Empiryczna')
axs[1, 0].bar(v_values + 0.2, [variances_theoretical[v]['beta0'] for v in v_values], width=0.4,
              label='Beta0 - Teoretyczna', alpha=0.5)
axs[1, 0].set_title('Porównanie wariancji Beta0')
axs[1, 0].set_xlabel('Stopnie swobody')
axs[1, 0].set_ylabel('Wariancja')
axs[1, 0].legend()

# Wykresy porównawcze dla Beta1
axs[1, 1].bar(v_values - 0.2, [variances_empirical[v]['beta1'] for v in v_values], width=0.4,
              label='Beta1 - Empiryczna')
axs[1, 1].bar(v_values + 0.2, [variances_theoretical[v]['beta1'] for v in v_values], width=0.4,
              label='Beta1 - Teoretyczna', alpha=0.5)
axs[1, 1].set_title('Porównanie wariancji Beta1')
axs[1, 1].set_xlabel('Stopnie swobody')
axs[1, 1].set_ylabel('Wariancja')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Zad 1

# Parametry modelu regresji
beta0 = 5  # Wyraz wolny
beta1 = 2  # Współczynnik kierunkowy
n = 1000  # Liczba próbek
sigma = 1  # Odchylenie standardowe błędu (epsilon)
num_simulations = 1000  # Zwiększona liczba symulacji Monte Carlo
alpha = 0.05  # Poziom istotności
df_range = np.arange(10, 101, 5)  # Stopnie swobody od 3 do 100

# Zmienne do przechowywania wyników
empirical_means_b0 = []
empirical_means_b1 = []
empirical_vars_b0 = []
empirical_vars_b1 = []
theoretical_means_b0 = []
theoretical_means_b1 = []
theoretical_vars_b0 = []
theoretical_vars_b1 = []

# Symulacje Monte Carlo dla różnych stopni swobody
for df in df_range:
    b0_hats = []
    b1_hats = []

    # Monte Carlo simulation
    for _ in range(num_simulations):
        x = np.linspace(0, 10, n).reshape(-1, 1)

        # Błędy z rozkładu t-Studenta, normalizowane w zależności od df
        epsilon = t.rvs(df, size=n) / np.sqrt(df / (df - 2)) if df > 2 else np.zeros(n)  # Normalizacja wariancji do 1
        y = beta0 + beta1 * x + epsilon.reshape(-1, 1)

        # Dopasowanie modelu regresji liniowej
        model = LinearRegression()
        model.fit(x, y)

        # Zapisanie estymatorów
        b0_hats.append(model.intercept_[0])
        b1_hats.append(model.coef_[0][0])

    # Obliczanie wartości oczekiwanej i wariancji estymatorów empirycznych
    empirical_means_b0.append(np.mean(b0_hats))
    empirical_means_b1.append(np.mean(b1_hats))

    # Wyliczenia związane z x
    mean_x = np.mean(x)
    sum_sq_diff_x = np.sum((x - mean_x) ** 2)

    # Zmodyfikowana empiryczna wariancja b0
    var_b0 = np.var(b0_hats)
    empirical_vars_b0.append(var_b0)

    # Zmodyfikowana empiryczna wariancja b1
    var_b1 = np.var(b1_hats)
    empirical_vars_b1.append(var_b1)

    # Teoretyczne wartości oczekiwane
    theoretical_means_b0.append(beta0)
    theoretical_means_b1.append(beta1)

    # Teoretyczna wariancja błędów przy danej liczbie stopni swobody
    theoretical_var_epsilon = df / (df - 2) if df > 2 else np.inf  # Wariancja t-Studenta
    theoretical_vars_b0.append((1 / n + (mean_x ** 2) / sum_sq_diff_x) * theoretical_var_epsilon)
    theoretical_vars_b1.append((1 / sum_sq_diff_x) * theoretical_var_epsilon)

# Wykresy porównujące empiryczne i teoretyczne wartości oczekiwane oraz wariancje

# Wartość oczekiwana estymatora beta_0 i beta_1
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df_range, empirical_means_b0, 'o-', label="Empiryczne $\\hat{b}_0$")
plt.plot(df_range, theoretical_means_b0, 's-', label="Teoretyczne $\\hat{b}_0$")
plt.title("Porównanie wartości oczekiwanej $\\hat{b}_0$")
plt.xlabel("Stopnie swobody (df)")
plt.ylabel("Wartość oczekiwana")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(df_range, empirical_means_b1, 'o-', label="Empiryczne $\\hat{b}_1$")
plt.plot(df_range, theoretical_means_b1, 's-', label="Teoretyczne $\\hat{b}_1$")
plt.title("Porównanie wartości oczekiwanej $\\hat{b}_1$")
plt.xlabel("Stopnie swobody (df)")
plt.ylabel("Wartość oczekiwana")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Wariancja estymatora beta_0 i beta_1
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df_range, empirical_vars_b0, 'o-', label="Empiryczne $\\hat{b}_0$")
plt.plot(df_range, theoretical_vars_b0, 's-', label="Teoretyczne $\\hat{b}_0$")
plt.title("Porównanie wariancji $\\hat{b}_0$")
plt.xlabel("Stopnie swobody (df)")
plt.ylabel("Wariancja")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(df_range, empirical_vars_b1, 'o-', label="Empiryczne $\\hat{b}_1$")
plt.plot(df_range, theoretical_vars_b1, 's-', label="Teoretyczne $\\hat{b}_1$")
plt.title("Porównanie wariancji $\\hat{b}_1$")
plt.xlabel("Stopnie swobody (df)")
plt.ylabel("Wariancja")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 2 Część

# Parametry modelu regresji
df = 10  # Stała liczba stopni swobody
sigma = 1  # Odchylenie standardowe błędu (epsilon)
alpha = 0.05  # Poziom istotności
n_values = np.arange(50, 1001, 50)  # Liczba próbek od 50 do 1000

# Zmienne do przechowywania wyników
empirical_means_b0 = []
empirical_means_b1 = []
empirical_vars_b0 = []
empirical_vars_b1 = []
theoretical_means_b0 = []
theoretical_means_b1 = []
theoretical_vars_b0 = []
theoretical_vars_b1 = []

# Symulacje Monte Carlo dla różnych liczebności próbek
for n in n_values:
    b0_hats = []
    b1_hats = []

    # Monte Carlo simulation
    for _ in range(num_simulations):
        x = np.linspace(0, 10, n).reshape(-1, 1)
        epsilon = t.rvs(df, size=n)  # Błędy z rozkładu t-Studenta o stopniach swobody df=10
        y = beta0 + beta1 * x + epsilon.reshape(-1, 1)

        # Dopasowanie modelu regresji liniowej
        model = LinearRegression()
        model.fit(x, y)

        # Zapisanie estymatorów
        b0_hats.append(model.intercept_[0])
        b1_hats.append(model.coef_[0][0])

    # Obliczanie wartości oczekiwanej i wariancji estymatorów empirycznych
    empirical_means_b0.append(np.mean(b0_hats))
    empirical_means_b1.append(np.mean(b1_hats))
    empirical_vars_b0.append(np.var(b0_hats))
    empirical_vars_b1.append(np.var(b1_hats))

    # Teoretyczne wartości oczekiwane i wariancje (zakładając asymptotyczne rozkłady)
    theoretical_means_b0.append(beta0)
    theoretical_means_b1.append(beta1)

    # Teoretyczna wariancja błędów przy danej liczbie próbek
    theoretical_var_epsilon = df / (df - 2) if df > 2 else np.inf  # Wariancja t-Studenta
    theoretical_vars_b0.append(theoretical_var_epsilon / n)
    theoretical_vars_b1.append(theoretical_var_epsilon / n)

# Wykresy porównujące empiryczne i teoretyczne wartości oczekiwane oraz wariancje

# Wartość oczekiwana estymatora beta_0 i beta_1
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(n_values, empirical_means_b0, 'o-', label="Empiryczne $\\hat{b}_0$")
plt.plot(n_values, theoretical_means_b0, 's-', label="Teoretyczne $\\hat{b}_0$")
plt.title("Porównanie wartości oczekiwanej $\\hat{b}_0$")
plt.xlabel("Liczba próbek (n)")
plt.ylabel("Wartość oczekiwana")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(n_values, empirical_means_b1, 'o-', label="Empiryczne $\\hat{b}_1$")
plt.plot(n_values, theoretical_means_b1, 's-', label="Teoretyczne $\\hat{b}_1$")
plt.title("Porównanie wartości oczekiwanej $\\hat{b}_1$")
plt.xlabel("Liczba próbek (n)")
plt.ylabel("Wartość oczekiwana")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Wariancja estymatora beta_0 i beta_1
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(n_values, empirical_vars_b0, 'o-', label="Empiryczne $\\hat{b}_0$")
plt.plot(n_values, theoretical_vars_b0, 's-', label="Teoretyczne $\\hat{b}_0$")
plt.title("Porównanie wariancji $\\hat{b}_0$")
plt.xlabel("Liczba próbek (n)")
plt.ylabel("Wariancja")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(n_values, empirical_vars_b1, 'o-', label="Empiryczne $\\hat{b}_1$")
plt.plot(n_values, theoretical_vars_b1, 's-', label="Teoretyczne $\\hat{b}_1$")
plt.title("Porównanie wariancji $\\hat{b}_1$")
plt.xlabel("Liczba próbek (n)")
plt.ylabel("Wariancja")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()