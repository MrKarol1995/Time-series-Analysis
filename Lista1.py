import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats
from typing import Callable, List, Tuple
from sklearn.linear_model import LinearRegression


# Zad 1

# blad bezwzgledny
def oblicz_mse(y_prawdziwe, y_przewidywane):
    n = len(y_prawdziwe)
    mse = np.sum((y_prawdziwe - y_przewidywane) ** 2) / n
    return mse



def oblicz_r2(y_prawdziwe, y_przewidywane):
    ssr = np.sum((y_przewidywane - np.mean(y_prawdziwe)) ** 2)
    sst = np.sum((y_prawdziwe - np.mean(y_prawdziwe)) ** 2)
    r2 = (ssr / sst)
    return r2


def oblicz_mae(y_prawdziwe, y_przewidywane):
    n = len(y_prawdziwe)
    mae = np.sum(np.abs(y_prawdziwe - y_przewidywane)) / n
    return mae


# Funkcja do wczytania danych z pliku i dopasowania modelu
def wczytaj_i_rysuj_wykres(plik):
    # Wczytywanie danych z pliku
    dane = np.loadtxt(plik)

    # Pierwsza kolumna jako zmienna objaśniająca (X), druga jako objaśniana (Y)
    X = dane[:, 0]
    Y = dane[:, 1]

    # Tworzymy wykres rozproszenia danych
    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, label="Dane", color="blue")

    # Listy do przechowywania wyników błędów i R^2
    stopnie = list(range(1, 6))
    mse_values = []
    r2_values = []
    mae_values = []

    # Rysowanie dopasowania dla stopni od 1 do 5
    x_linspace = np.linspace(min(X), max(X), 500)  # Ciągły zakres dla gładkiego wykresu

    for stopien in stopnie:
        # Dopasowanie wielomianu o zadanym stopniu
        wspolczynniki = np.polyfit(X, Y, stopien)

        # Utworzenie funkcji wielomianowej
        poly_func = np.poly1d(wspolczynniki)

        # Wygenerowanie wartości Y na podstawie modelu (dla ciągłej linii)
        Y_fit = poly_func(x_linspace)

        # Obliczenie przewidywanych wartości Y dla danych X
        Y_pred = poly_func(X)

        # Obliczenie średniego błędu kwadratowego, R^2 i MAE
        mse = oblicz_mse(Y, Y_pred)
        r2 = oblicz_r2(Y, Y_pred)
        mae = oblicz_mae(Y, Y_pred)

        # Zapisanie wyników do porównania
        mse_values.append(mse)
        r2_values.append(r2)
        mae_values.append(mae)

        # Rysowanie dopasowania
        plt.plot(x_linspace, Y_fit, label=f'Stopień {stopien}')

    # Dodatkowe elementy wykresu
    plt.xlabel('Zmienna objaśniająca')
    plt.ylabel('Zmienna objaśniana')
    plt.title('Porównanie regresji wielomianowej od stopnia 1 do 5')
    plt.legend()
    plt.show()

    # Rysowanie porównania MSE, MAE i R² dla różnych stopni

    # Wykres 1: MSE
    plt.figure(figsize=(10, 6))
    plt.plot(stopnie, mse_values, 'r-', marker='o', label='MSE')
    plt.xlabel('Stopień wielomianu')
    plt.ylabel('MSE')
    plt.title('Porównanie MSE dla różnych stopni wielomianu')
    plt.grid(True)
    plt.show()

    # Wykres 2: MAE
    plt.figure(figsize=(10, 6))
    plt.plot(stopnie, mae_values, 'g-', marker='o', label='MAE')
    plt.xlabel('Stopień wielomianu')
    plt.ylabel('Bezwzgledny')
    plt.title('Porównanie MAE dla różnych stopni wielomianu')
    plt.grid(True)
    plt.show()

    # Wykres 3: R²
    plt.figure(figsize=(10, 6))
    plt.plot(stopnie, r2_values, 'b-', marker='o', label='R²')
    plt.xlabel('Stopień wielomianu')
    plt.ylabel('R²')
    plt.title('Porównanie R² dla różnych stopni wielomianu')
    plt.grid(True)
    plt.show()


# Przykład użycia funkcji z plikiem "zad1.txt"
wczytaj_i_rysuj_wykres('zad1_lista1.txt')


# Zad 2

# Funkcja do obliczania średniej ruchomej wg. zadanej formuły
def custom_moving_average(data: pd.Series, p: int) -> pd.Series:
    ma_values = []
    n = len(data)
    window_size = 2 * p + 1

    # Iterujemy przez dane, ale tylko od p do n - p
    for t in range(p, n - p):
        window_sum = 0

        # Sumujemy wartości od -p do p dla bieżącego punktu t
        for j in range(-p, p + 1):
            window_sum += data[t + j]

        # Obliczamy średnią dla tego punktu
        ma_value = window_sum / window_size
        ma_values.append(ma_value)

    # Uzupełniamy początkowe i końcowe wartości NaN
    ma_values = [None] * p + ma_values + [None] * p
    return pd.Series(ma_values)


# Wczytaj dane z pliku tekstowego
file_path = 'zad2_lista1.txt'  # Zmień na rzeczywistą ścieżkę
data = pd.read_csv(file_path, header=None, names=['values'])  # Odczyt danych z pliku

# Kolumna z danymi
values = data['values']

# Obliczanie średnich ruchomych dla różnych okien
p_11 = 5  # Średnia ruchoma 11 to p=5
ma_11 = custom_moving_average(values, p_11)

p_25 = 12  # Średnia ruchoma 25 to p=12
ma_25 = custom_moving_average(values, p_25)

# Podstawa 2p + 1, gdzie p jest dowolną liczbą wybraną przez użytkownika
p = 11
ma_2p1 = custom_moving_average(values, p)

# Wykres danych
plt.figure(figsize=(10, 6))
plt.plot(values, label='Oryginalne dane', color='blue')
plt.plot(ma_11, label='Średnia ruchoma (11)', color='orange')
plt.plot(ma_25, label='Średnia ruchoma (25)', color='green')
plt.plot(ma_2p1, label=f'Średnia ruchoma (2p+1) gdzie p={p}', color='red')
plt.legend()
plt.title('Porównanie średnich ruchomych')
plt.xlabel('Indeks')
plt.ylabel('Wartości')
plt.grid(True)
plt.show()

# Funkcja do obliczania prostej średniej ruchomej
def moving_average(data: pd.Series, window_size: int) -> pd.Series:
    return data.rolling(window=window_size, center=True).mean()

# Wczytaj dane z pliku tekstowego
file_path = 'zad2_lista1.txt'  # Zmień na rzeczywistą ścieżkę
data = pd.read_csv(file_path, header=None, names=['values'])  # Odczyt danych z pliku

# Kolumna z danymi
values = data['values']

# Obliczanie średnich ruchomych dla różnych okien
ma_11 = moving_average(values, 11)
ma_25 = moving_average(values, 25)

# Podstawa 2p + 1, gdzie p jest dowolną liczbą wybraną przez użytkownika
p = 11 # Przykładowo
ma_2p1 = moving_average(values, 2 * p + 1)

# Wykres danych
plt.figure(figsize=(10, 6))
plt.plot(values, label='Oryginalne dane', color='blue')
plt.plot(ma_11, label='Średnia ruchoma (11)', color='orange')
plt.plot(ma_25, label='Średnia ruchoma (25)', color='green')
plt.plot(ma_2p1, label=f'Średnia ruchoma (2p+1) gdzie p={p}', color='red')
plt.legend()
plt.title('Porównanie średnich ruchomych')
plt.xlabel('Indeks')
plt.ylabel('Wartości')
plt.grid(True)
plt.show()

# Zad 3

# Wczytaj dane
X = np.loadtxt('zad2_lista1.txt')
Y = np.loadtxt('zad3_lista1.txt')


# Funkcja do obliczenia współczynnika nachylenia (b1) i wyrazu wolnego (b0)
def oblicz_regresje(X, Y):
    # Obliczenie średnich X i Y
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # Obliczenie współczynnika b1
    s1 = np.sum((X - X_mean) * (Y - Y_mean))
    s2 = np.sum((X - X_mean) ** 2)
    b1 = s1 / s2

    # Obliczenie współczynnika b0
    b0 = Y_mean - b1 * X_mean

    return b0, b1


# Funkcja do obliczenia współczynnika R^2


# Wyznacz współczynniki regresji
b0, b1 = oblicz_regresje(X, Y)


# Wyświetlenie wyników
print(f"Współczynnik b0: {b0}")
print(f"Współczynnik b1: {b1}")

# Rysowanie wykresu
plt.scatter(X, Y, label='Dane', color='blue')
plt.plot(X, b0 + b1 * X, label='Regresja liniowa', color='red')
plt.legend()
plt.title(f'Regresja liniowa z R^2 = ')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.grid(True)
plt.show()
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

# Krok 4: Zidentyfikowanie obserwacji odstających (metoda 3-sigma)
std_residuals = np.std(residuals)
outliers = np.abs(residuals) > 3 * std_residuals  # Obserwacje odstające

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

# Krok 5: Usunięcie obserwacji odstających
X_clean = X[~outliers]
Y_clean = Y[~outliers]

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

# Zad 5


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
prediction_errors_test = Y_test - Y_pred_test

# Obliczenie wartości przewidywanych dla zbioru treningowego
Y_pred_train = intercept_train + slope_train * X_train

# Wyznaczenie błędów dla zbioru treningowego
prediction_errors_train = Y_train - Y_pred_train

# Obliczenie miar błędów dla zbioru treningowego
mae_train = np.mean(np.abs(prediction_errors_train))
mse_train = np.mean(prediction_errors_train ** 2)

# Obliczenie miar błędów dla zbioru testowego
mae_test = np.mean(np.abs(prediction_errors_test))
mse_test = np.mean(prediction_errors_test ** 2)

# Wyświetlenie miar błędów
# Przygotowanie danych do tabeli
data = {
    'Zbiór danych': ['Treningowy (MAE)', 'Treningowy (MSE)', 'Testowy (MAE)', 'Testowy (MSE)'],
    'Wartość': [mae_train, mse_train, mae_test, mse_test]
}

# Utworzenie DataFrame z danych
error_table = pd.DataFrame(data)

# Wyświetlenie tabeli
print('\nMiary błędów:')
print(error_table)


# Wyświetlenie wyników dla poszczególnych obserwacji w zbiorze testowym
for i in range(len(X_test)):
    print(
        f'Obserwacja {i + 991}: Rzeczywista wartość: {Y_test[i]}, Przewidywana wartość: {Y_pred_test[i]}'
        f', Błąd predykcji: {prediction_errors_test[i]}')

# Wizualizacja regresji i predykcji
plt.scatter(X_train, Y_train, label='Dane (treningowe)')
plt.scatter(X_test, Y_test, label='Dane (testowe)', color='orange')
plt.plot(X_train, intercept_train + slope_train * X_train, color='red', label='Regresja liniowa')
# plt.scatter(X_test, Y_pred_test, label='Przewidywane wartości', marker='x', color='black')
plt.legend()
plt.title('Regresja liniowa i predykcje')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.show()

# Wizualizacja błędów predykcji
plt.figure(figsize=(10, 5))

# Ustalamy osie X od 991 do 1000
observation_indices = np.arange(991, 1001)

plt.scatter(observation_indices, prediction_errors_test, color='blue', label='Błędy predykcji')
plt.axhline(0, color='red', linestyle='--', label='Brak błędu')
plt.title('Błędy predykcji dla zbioru testowego')
plt.xlabel('Indeks obserwacji (od 991 do 1000)')
plt.ylabel('Błąd predykcji (Rzeczywista wartość - Przewidywana wartość)')
plt.legend()
plt.grid()
plt.xticks(observation_indices)  # Ustalamy etykiety na osi X
plt.show()
# Zad 6

print('Zad 6')

# Użycie funkcji do wczytania danych
filename = 'zad6_lista1.txt'  # Podaj nazwę swojego pliku
X6, Y6 = read_columns_from_file(filename)

plt.scatter(X6, Y6, label='Dane', color='blue')
plt.legend()
plt.title('Dane z Zadanie 6')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Użycie funkcji do wczytania danych
filename = 'zad6_lista1.txt'  # Podaj nazwę swojego pliku
X7, Y7 = read_columns_from_file(filename)

# Konwersja do tablic numpy
# Konwersja do tablic numpy
X7 = np.array(X7)
Y7 = np.array(Y7)

# Krok 1: Zastosowanie regresji na oryginalnych danych
# Zlogarytmowanie danych Y (zależnej)
Y_log = np.log(Y7)

# Krok 2: Wyznaczenie prostej regresji dla przetransformowanych danych
slope_transformed, intercept_transformed, r_value_transformed, p_value_transformed, std_err_transformed \
    = stats.linregress(X7, Y_log)
print(f'Współczynnik kierunkowy (przetransformowane): {slope_transformed}, wyraz wolny: {intercept_transformed}')

# Krok 3: Obliczenie przewidywanych wartości
Y_pred_transformed = intercept_transformed + slope_transformed * X7

# Krok 4: Przekształcenie przewidywanych wartości z logarytmu
Y_pred_original = np.exp(Y_pred_transformed)

# Krok 5: Wizualizacja regresji dla przetransformowanych danych
plt.figure(figsize=(10, 6))


# Wykres z logarytmem

plt.scatter(X7, Y_log, label='Log(Y)', color='blue')
plt.plot(X7, Y_pred_transformed, color='red', label='Regresja liniowa (log)')
plt.legend()
plt.title('Regresja liniowa dla przetransformowanych danych (log)')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Log(Y) (zmienna objaśniana)')

plt.tight_layout()
plt.show()

