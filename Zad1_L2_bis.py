import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.linear_model import LinearRegression


# Ustawienia parametrów
beta0 = 5
beta1 = 2
n = 100  # liczba obserwacji
v = 5    # stopnie swobody dla rozkładu t-Studenta

# Generowanie danych
np.random.seed(42)  # dla powtarzalności wyników
x = np.random.uniform(0, 10, n)  # generowanie wartości x
epsilon = np.random.standard_t(v, n)  # generowanie błędów z rozkładu t-Studenta
Y = beta0 + beta1 * x + epsilon  # generowanie wartości Y

# Tworzenie DataFrame do analizy
data = pd.DataFrame({'X': x, 'Y': Y})

# Dodanie stałej do modelu (intercept)
X = sm.add_constant(data['X'])

# Dopasowanie modelu regresji liniowej
model = sm.OLS(data['Y'], X).fit()

# Wyświetlenie wyników estymacji
# print(model.summary())


# Zad 1

# Parametry
b0 = 5  # Wyraz wolny
beta1 = 2  # Współczynnik kierunkowy
n = 1000  # Liczba próbek

# Zmienna do przechowywania wartości estymowanych b_0 i b_1
b0_hats_mean = []
b1_hats_mean = []
empirical_vars_b0 = []
empirical_vars_b1 = []
theoretical_means_b0 = []
theoretical_means_b1 = []
theoretical_vars_b0 = []
theoretical_vars_b1 = []

# Zakres stopni swobody
df_values = range(3, 20)  # Można dostosować zakres

# Liczba symulacji
num_simulations = 1000

# Symulacja dla różnych stopni swobody
for df in df_values:
    b0_hats = []
    b1_hats = []

    for _ in range(num_simulations):
        # Generowanie epsilon (błędów) z rozkładu t-Studenta
        epsilon = t.rvs(df, size=n)

        # Generowanie wartości x i y
        x = np.linspace(0, 10, num=n)  # Wartości x od 0 do 10
        y = b0 + beta1 * x + epsilon  # Generowanie wartości y

        # Przygotowanie macierzy X do metody najmniejszych kwadratów
        X = np.column_stack((np.ones(n), x))  # Dodajemy kolumnę jedynek dla wyrazu wolnego

        # Dopasowanie modelu regresji liniowej metodą najmniejszych kwadratów
        model = LinearRegression()
        model.fit(X, y)

        # Wyznaczanie estymowanych wartości b_0_hat i b_1_hat
        b0_hats.append(model.intercept_)  # Estymator b_0
        b1_hats.append(model.coef_[1])     # Estymator b_1 (indeks 1 dla beta1)

    # Obliczanie średnich wartości estymatorów
    b0_hats_mean.append(np.mean(b0_hats))
    b1_hats_mean.append(np.mean(b1_hats))

    # Obliczenia związane z x
    mean_x = np.mean(x)
    sum_sq_diff_x = np.sum((x - mean_x) ** 2)

    # Zmodyfikowana empiryczna wariancja b0
    var_b0 = np.var(b0_hats)
    empirical_vars_b0.append(var_b0)

    # Zmodyfikowana empiryczna wariancja b1
    var_b1 = np.var(b1_hats)
    empirical_vars_b1.append(var_b1)

    # Teoretyczne wartości oczekiwane
    theoretical_means_b0.append(b0)
    theoretical_means_b1.append(beta1)

    # Teoretyczna wariancja błędów przy danej liczbie stopni swobody
    theoretical_var_epsilon = df / (df - 2) if df > 2 else np.inf  # Wariancja t-Studenta
    theoretical_vars_b0.append(((1 / n) + (mean_x ** 2) / sum_sq_diff_x) * theoretical_var_epsilon)
    theoretical_vars_b1.append((1 / sum_sq_diff_x) * theoretical_var_epsilon)

# Wykres wartości oczekiwanych estymatorów
plt.figure(figsize=(14, 6))

# Wykres dla b0_hat
plt.subplot(2, 2, 1)
plt.plot(df_values, b0_hats_mean, marker='o', label='Empiryczna $\hat{b_0}$')
plt.axhline(y=b0, color='r', linestyle='--', label='Prawdziwe b0')
plt.xlabel('Stopnie swobody')
plt.ylabel('Oczekiwana wartość $\hat{b_0}$')
plt.title('Zależność $\hat{b_0}$ od stopni swobody')
plt.legend()

# Wykres dla b1_hat
plt.subplot(2, 2, 2)
plt.plot(df_values, b1_hats_mean, marker='o', label='Empiryczna $\hat{b_1}$')
plt.axhline(y=beta1, color='r', linestyle='--', label='Prawdziwe b1')
plt.xlabel('Stopnie swobody')
plt.ylabel('Oczekiwana wartość $\hat{b_1}$')
plt.title('Zależność $\hat{b_1}$ od stopni swobody')
plt.legend()

# Wykres empirycznych wariancji
plt.subplot(2, 2, 3)
plt.plot(df_values, empirical_vars_b0, marker='o', label='Empiryczna wariancja $\hat{b_0}$')
plt.plot(df_values, empirical_vars_b1, marker='o', label='Empiryczna wariancja $\hat{b_1}$')
plt.xlabel('Stopnie swobody')
plt.ylabel('Empiryczna wariancja')
plt.title('Empiryczne wariancje estymatorów')
plt.legend()

# Wykres teoretycznych wariancji
plt.subplot(2, 2, 4)
plt.plot(df_values, theoretical_vars_b0, marker='o', label='Teoretyczna wariancja $\hat{b_0}$')
plt.plot(df_values, theoretical_vars_b1, marker='o', label='Teoretyczna wariancja $\hat{b_1}$')
plt.xlabel('Stopnie swobody')
plt.ylabel('Teoretyczna wariancja')
plt.title('Teoretyczne wariancje estymatorów')
plt.legend()

plt.tight_layout()
plt.show()


# druga cześć
# Parametry
b0 = 5  # Wyraz wolny
beta1 = 2  # Współczynnik kierunkowy

# Zmienna do przechowywania wartości estymowanych b_0 i b_1
b0_hats_mean = []
b1_hats_mean = []
empirical_vars_b0 = []
empirical_vars_b1 = []
theoretical_means_b0 = []
theoretical_means_b1 = []
theoretical_vars_b0 = []
theoretical_vars_b1 = []

# Stała liczba stopni swobody
df = 10

# Zakres n od 10 do 100
n_values = range(10, 101, 10)  # Możesz zmienić krok na inny, jeśli chcesz

# Liczba symulacji
num_simulations = 1000

# Symulacja dla różnych wartości n
for n in n_values:
    b0_hats = []
    b1_hats = []

    for _ in range(num_simulations):
        # Generowanie epsilon (błędów) z rozkładu t-Studenta
        epsilon = t.rvs(df, size=n)

        # Generowanie wartości x i y
        x = np.linspace(0, 10, num=n)  # Wartości x od 0 do 10
        y = b0 + beta1 * x + epsilon  # Generowanie wartości y

        # Przygotowanie macierzy X do metody najmniejszych kwadratów
        X = np.column_stack((np.ones(n), x))  # Dodajemy kolumnę jedynek dla wyrazu wolnego

        # Dopasowanie modelu regresji liniowej metodą najmniejszych kwadratów
        model = LinearRegression()
        model.fit(X, y)

        # Wyznaczanie estymowanych wartości b_0_hat i b_1_hat
        b0_hats.append(model.intercept_)  # Estymator b_0
        b1_hats.append(model.coef_[1])     # Estymator b_1 (indeks 1 dla beta1)

    # Obliczanie średnich wartości estymatorów
    b0_hats_mean.append(np.mean(b0_hats))
    b1_hats_mean.append(np.mean(b1_hats))

    # Obliczenia związane z x
    mean_x = np.mean(x)
    sum_sq_diff_x = np.sum((x - mean_x) ** 2)

    # Zmodyfikowana empiryczna wariancja b0
    var_b0 = np.var(b0_hats)
    empirical_vars_b0.append(var_b0)

    # Zmodyfikowana empiryczna wariancja b1
    var_b1 = np.var(b1_hats)
    empirical_vars_b1.append(var_b1)

    # Teoretyczne wartości oczekiwane
    theoretical_means_b0.append(b0)
    theoretical_means_b1.append(beta1)

    # Teoretyczna wariancja błędów przy danej liczbie stopni swobody
    theoretical_var_epsilon = df / (df - 2)
    theoretical_vars_b0.append((1 / n + (mean_x ** 2) / sum_sq_diff_x) * theoretical_var_epsilon)
    theoretical_vars_b1.append((1 / sum_sq_diff_x) * theoretical_var_epsilon)

# Wykres wartości oczekiwanych estymatorów
plt.figure(figsize=(14, 6))

# Wykres dla b0_hat
plt.subplot(2, 2, 1)
plt.plot(n_values, b0_hats_mean, marker='o', label='Empiryczna $\hat{b_0}$')
plt.axhline(y=b0, color='r', linestyle='--', label='Prawdziwe b0')
plt.xlabel('Liczba próbek (n)')
plt.ylabel('Oczekiwana wartość $\hat{b_0}$')
plt.title('Zależność $\hat{b_0}$ od liczby próbek')
plt.legend()

# Wykres dla b1_hat
plt.subplot(2, 2, 2)
plt.plot(n_values, b1_hats_mean, marker='o', label='Empiryczna $\hat{b_1}$')
plt.axhline(y=beta1, color='r', linestyle='--', label='Prawdziwe b1')
plt.xlabel('Liczba próbek (n)')
plt.ylabel('Oczekiwana wartość $\hat{b_1}$')
plt.title('Zależność $\hat{b_1}$ od liczby próbek')
plt.legend()

# Wykres empirycznych wariancji
plt.subplot(2, 2, 3)
plt.plot(n_values, empirical_vars_b0, marker='o', label='Empiryczna wariancja $\hat{b_0}$')
plt.plot(n_values, empirical_vars_b1, marker='o', label='Empiryczna wariancja $\hat{b_1}$')
plt.xlabel('Liczba próbek (n)')
plt.ylabel('Empiryczna wariancja')
plt.title('Empiryczne wariancje estymatorów')
plt.legend()

# Wykres teoretycznych wariancji
plt.subplot(2, 2, 4)
plt.plot(n_values, theoretical_vars_b0, marker='o', label='Teoretyczna wariancja $\hat{b_0}$')
plt.plot(n_values, theoretical_vars_b1, marker='o', label='Teoretyczna wariancja $\hat{b_1}$')
plt.xlabel('Liczba próbek (n)')
plt.ylabel('Teoretyczna wariancja')
plt.title('Teoretyczne wariancje estymatorów')
plt.legend()

plt.tight_layout()
plt.show()