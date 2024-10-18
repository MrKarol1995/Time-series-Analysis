import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats
from typing import Callable, List, Tuple
from sklearn.linear_model import LinearRegression

# Zad 4

print("Zad 4")
# Krok 1: Wczytanie danych z pliku
def read_columns_from_file(filename: str) -> Tuple[List[float], List[float]]:
    column1 = []
    column2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()  # Rozdzielenie linii na kolumny (domyślnie spacje)
            if len(values) >= 2:  # Sprawdzenie, czy są co najmniej dwie kolumny
                column1.append(float(values[0]))  # Pierwsza kolumna
                column2.append(float(values[1]))  # Druga kolumna

    return column1, column2

# Użycie funkcji do wczytania danych
filename = 'zad4_lista1.txt'  # Podaj nazwę swojego pliku
X, Y = read_columns_from_file(filename)

# Konwersja do tablic numpy
X = np.array(X)
Y = np.array(Y)

# Krok 2: Wyznaczenie prostej regresji metodą najmniejszych kwadratów
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print(f'Współczynnik kierunkowy: {slope}, wyraz wolny: {intercept}')

# Krok 3: Obliczenie residuów (błędów)
Y_pred = intercept + slope * X
residuals = Y - Y_pred

# Krok 4: Zidentyfikowanie obserwacji odstających
std_residuals = np.std(residuals) # odchylenie z residuum 
outliers = np.abs(residuals) > 2 * std_residuals  # Obserwacje odstające

print(f'Liczba obserwacji odstających: {np.sum(outliers)}')

# Wizualizacja prostej regresji i obserwacji odstających
plt.scatter(X, Y, label='Dane')
plt.plot(X, Y_pred, color='red', label='Regresja liniowa')
plt.scatter(X[outliers], Y[outliers], color='orange', label='Odstające', zorder=5)
plt.legend()
plt.title('Regresja liniowa z obserwacjami odstającymi')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.show()

# Krok 5: Usunięcie obserwacji odstających bez użycia ~outliers
X_clean = []
Y_clean = []

for i in range(len(X)):
    if not outliers[i]:  # Jeżeli obserwacja nie jest odstająca, dodajemy ją do X_clean i Y_clean
        X_clean.append(X[i])
        Y_clean.append(Y[i])

# Konwersja wyników do tablic numpy (opcjonalnie)
X_clean = np.array(X_clean)
Y_clean = np.array(Y_clean)


# Krok 6: Ponowne wyznaczenie prostej regresji po usunięciu obserwacji odstających
slope_clean, intercept_clean, _, _, _ = stats.linregress(X_clean, Y_clean)
print(f'Regresja liniowa (po usunięciu odstających): {slope_clean}x + {intercept_clean}')

# Wizualizacja prostej regresji po usunięciu odstających
Y_pred_clean = intercept_clean + slope_clean * X_clean
plt.scatter(X_clean, Y_clean, label='Dane po usunięciu odstających')
plt.plot(X_clean, Y_pred_clean, color='green', label='Regresja liniowa (po usunięciu odstających)')
plt.legend()
plt.title('Regresja liniowa po usunięciu obserwacji odstających')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.show()

# Krok 7: Wizualizacja residuów (różnica między rzeczywistymi wartościami a przewidywanymi)
plt.scatter(X, residuals, label='Residua', color='purple')
plt.axhline(y=0, color='red', linestyle='--', label='Linia zerowa')
plt.title('Wykres residuów')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Residua (Y - Y_pred)')
plt.legend()
plt.show()