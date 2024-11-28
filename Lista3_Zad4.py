import math
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import t


def read_columns_from_file(filename: str) -> Tuple[List[float], List[float]]:
    column1 = []
    column2 = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) >= 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))
    return column1, column2


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Oblicza współczynniki regresji beta0_hat, beta1_hat oraz wariancję resztową s^2."""
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    beta1_hat = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / sum((xi - x_mean) ** 2 for xi in x)
    beta0_hat = y_mean - beta1_hat * x_mean

    # Obliczenie y_hat dla danych treningowych
    y_hat = [beta0_hat + beta1_hat * xi for xi in x]

    # Wariancja reszt
    s_squared = sum((yi - y_hat_i) ** 2 for yi, y_hat_i in zip(y, y_hat)) / (n - 2)
    return beta0_hat, beta1_hat, s_squared


def confidence_interval(x_train: List[float], y_train: List[float], x0: float, alpha: float) -> Tuple[
    float, Tuple[float, float]]:
    """Oblicza punktowy oraz przedziałowy prognozę dla y0(x0)."""
    n = len(x_train)
    beta0_hat, beta1_hat, s_squared = linear_regression(x_train, y_train)
    x_mean = sum(x_train) / n
    sum_sq_x = sum((xi - x_mean) ** 2 for xi in x_train)
    y_hat = beta0_hat + beta1_hat * x0

    # Współczynnik t-Studenta
    t_value = t.ppf(1 - alpha / 2, n - 2)

    margin = t_value * math.sqrt(
        s_squared * (1 + 1 / n + ((x0 - x_mean) ** 2) / sum_sq_x)
    )
    return y_hat, (y_hat - margin, y_hat + margin)


# Wczytanie danych
filename = 'zad4_lista1.txt'  # Podaj nazwę swojego pliku
X, Y = read_columns_from_file(filename)

# Podział danych
data = sorted(zip(X, Y), key=lambda pair: pair[0])
X_sorted, Y_sorted = zip(*data)
X_train, Y_train = X_sorted[:990], Y_sorted[:990]
X_test, Y_test = X_sorted[990:], Y_sorted[990:]

# Wyznaczenie współczynników regresji na podstawie danych treningowych
beta0_hat, beta1_hat, s_squared = linear_regression(X_train, Y_train)


# Obliczenie przedziałów ufności dla danych testowych
alpha = 0.05  # Poziom ufności
test_predictions = []
test_confidence_intervals = []
within_interval = []  # Lista wskazująca, czy punkty są w przedziale

for x0, y_actual in zip(X_test, Y_test):
    y_hat, (ci_lower, ci_upper) = confidence_interval(X_train, Y_train, x0, alpha)
    test_predictions.append(y_hat)
    test_confidence_intervals.append((ci_lower, ci_upper))
    within_interval.append(ci_lower <= y_actual <= ci_upper)

# Wykres regresji
plt.figure(figsize=(12, 8))

# Punkty treningowe
plt.scatter(X_train, Y_train, color='blue', label='Dane treningowe')

# Punkty testowe w przedziale ufności
X_test_in = [x for x, inside in zip(X_test, within_interval) if inside]
Y_test_in = [y for y, inside in zip(Y_test, within_interval) if inside]
plt.scatter(X_test_in, Y_test_in, color='green', label='Dane testowe (w przedziale)')

# Punkty testowe poza przedziałem ufności
X_test_out = [x for x, inside in zip(X_test, within_interval) if not inside]
Y_test_out = [y for y, inside in zip(Y_test, within_interval) if not inside]
plt.scatter(X_test_out, Y_test_out, color='red', label='Dane testowe (poza przedziałem)')

# Linia regresji przedłużona na całe dane
x_range_full = [min(X_sorted), max(X_sorted)]  # Zakres obejmujący wszystkie dane
y_range_full = [beta0_hat + beta1_hat * x for x in x_range_full]
plt.plot(x_range_full, y_range_full, color='red', label=f'Regresja: y = {beta0_hat:.2f} + {beta1_hat:.2f}x')

# Dodanie przedziałów ufności dla danych testowych
for x0, (ci_lower, ci_upper) in zip(X_test, test_confidence_intervals):
    plt.plot([x0, x0], [ci_lower, ci_upper], color='green', linestyle='--')

# Oznaczenia osi i legenda
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linia regresji z przedziałami ufności i kolorowaniem danych testowych")
plt.legend()
plt.grid()
plt.show()