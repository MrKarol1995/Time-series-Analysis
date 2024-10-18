import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats
from typing import Callable, List, Tuple
from sklearn.linear_model import LinearRegression

print('Zad 5')
# Krok 2: Wyznaczenie prostej regresji dla pierwszych 990 obserwacji
X_train = X[:990]
Y_train = Y[:990]

slope_train, intercept_train, r_value_train, p_value_train, std_err_train = stats.linregress(X_train, Y_train)
print(f'Współczynnik kierunkowy (dla 990 obserwacji): {slope_train}, wyraz wolny: {intercept_train}')

# Krok 3: Dokonanie predykcji dla obserwacji 991-1000
X_test = X[990:1000]  # Obserwacje 991-1000
Y_test = Y[990:1000]  # Oczekiwane wartości

# Obliczenie przewidywanych wartości
Y_pred_test = intercept_train + slope_train * X_test

# Krok 4: Wyznaczenie błędów predykcji
prediction_errors = Y_test - Y_pred_test

# Wyświetlenie wyników
for i in range(len(X_test)):
    print(
        f'Obserwacja {i + 991}: Rzeczywista wartość: {Y_test[i]}, Przewidywana wartość: {Y_pred_test[i]}'
        f', Błąd predykcji: {prediction_errors[i]}')

# Wizualizacja regresji i predykcji
plt.scatter(X_train, Y_train, label='Dane (treningowe)')
plt.scatter(X_test, Y_test, label='Dane (testowe)', color='orange')
plt.plot(X_train, intercept_train + slope_train * X_train, color='red', label='Regresja liniowa')
plt.scatter(X_test, Y_pred_test, label='Przewidywane wartości', marker='x', color='black')
plt.legend()
plt.title('Regresja liniowa i predykcje')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.show()
