import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Funkcje do obliczania autokowariancji i autokorelacji empirycznych
def sampleACVF(X, h):
    n = len(X)
    x_mean = np.mean(X)
    return 1 / n * sum((X[i] - x_mean) * (X[i + abs(h)] - x_mean) for i in range(n - abs(h)))

def sampleACF(X, h):
    return sampleACVF(X, h) / sampleACVF(X, 0)

# Funkcje teoretyczne
def teoACVF(h):
    return 4 if h == 0 else 0

def teoACF(h):
    return 1 if h == 0 else 0

def calculate_error_single_n(n, max_lag, M):
    errors = []
    for _ in range(M):  # M powtórzeń
        # Generowanie danych
        X = np.random.normal(0, 2, size=n)

        # Obliczanie wartości empirycznych i teoretycznych
        błędy = []
        for h in range(max_lag + 1):
            sample_acvf = sampleACVF(X, h)
            true_acvf = teoACVF(h)
            abs_error = abs(sample_acvf - true_acvf)
            błędy.append(abs_error)

        # Wyznaczenie błędu średniego
        e = np.mean(błędy[:11])  # Uwaga: tylko dla h = 0, ..., 10
        errors.append(e)
    return errors

# Główna funkcja generująca dane i obliczenia
n = 1000  # Rozmiar próby
max_lag = 20  # Maksymalny lag
    # Generowanie danych
X = np.random.normal(0, 2, size=n)


    # Obliczanie wartości empirycznych i teoretycznych
wyniki = []
for h in range(max_lag + 1):
    sample_acvf = sampleACVF(X, h)
    sample_acf = sampleACF(X, h)
    wyniki.append([h, sample_acvf, teoACVF(h), sample_acf, teoACF(h)])

    # Tworzenie DataFrame dla wyników
df = pd.DataFrame(wyniki, columns=['h', 'sampleACVF', 'True ACVF', 'sampleACF', 'True ACF'])

    # Wizualizacja autokowariancji
plt.figure(figsize=(12, 6))
plt.plot(df['h'], df['sampleACVF'], marker='o', label='Empiryczna ACVF')
plt.scatter(df['h'], df['True ACVF'], color='red', label='Teoretyczna ACVF', zorder=3)
plt.xlabel('Lag (h)')
plt.ylabel('Autokowariancja')
plt.title('Porównanie empirycznej i teoretycznej funkcji ACVF')
plt.legend()
plt.grid()
plt.show()

    # Wizualizacja autokorelacji
plt.figure(figsize=(12, 6))
plt.plot(df['h'], df['sampleACF'], marker='o', label='Empiryczna ACF')
plt.scatter(df['h'], df['True ACF'], color='red', label='Teoretyczna ACF', zorder=3)
plt.xlabel('Lag (h)')
plt.ylabel('Autokorelacja')
plt.title('Porównanie empirycznej i teoretycznej funkcji ACF')
plt.legend()
plt.grid()
plt.show()



# 2 czesc


# Funkcje do obliczania autokowariancji i autokorelacji empirycznych


# Funkcja główna do obliczania błędu dla różnych długości trajektorii
def calculate_error(n_values, max_lag):
    errors = []

    for n in n_values:
        # Generowanie danych
        X = np.random.normal(0, 2, size=n)

        # Obliczanie wartości empirycznych i teoretycznych
        błędy = []
        for h in range(max_lag + 1):
            sample_acvf = sampleACVF(X, h)
            true_acvf = teoACVF(h)
            abs_error = abs(sample_acvf - true_acvf)
            błędy.append(abs_error)

        # Wyznaczenie błędu średniego
        e = np.mean(błędy[:11])  # Uwaga: tylko dla h = 0, ..., 10
        errors.append(e)

    return errors

# Funkcja do obliczenia błędu dla jednego n
def calculate_error_single_n(n, max_lag, M):
    errors = []
    for _ in range(M):  # M powtórzeń
        # Generowanie danych
        X = np.random.normal(0, 2, size=n)

        # Obliczanie wartości empirycznych i teoretycznych
        błędy = []
        for h in range(max_lag + 1):
            sample_acvf = sampleACVF(X, h)
            true_acvf = teoACVF(h)
            abs_error = abs(sample_acvf - true_acvf)
            błędy.append(abs_error)

        # Wyznaczenie błędu średniego
        e = np.mean(błędy[:11])  # Uwaga: tylko dla h = 0, ..., 10
        errors.append(e)
    return errors

# Wizualizacja błędów dla różnych n

n_values = range(10, 201, 10)  # Długości trajektorii od 30 do 1000, co 10

    # Obliczanie błędu dla każdego n
errors = calculate_error(n_values, max_lag)

    # Wizualizacja
plt.figure(figsize=(12, 6))
plt.plot(n_values, errors, marker='o', linestyle='-', color='blue', label='Średni błąd e')
plt.xlabel('Długość trajektorii (n)')
plt.ylabel('Średni błąd e')
plt.title('Średni błąd autokowariancji w zależności od długości trajektorii')
plt.grid(linestyle='--', alpha=0.7)
plt.legend()
plt.show()

M = 100  # Liczba powtórzeń dla każdego n

    # Obliczanie błędów dla każdego n
all_errors = []
for n in n_values:
    errors = calculate_error_single_n(n, max_lag, M)
    all_errors.append(errors)

    # Wizualizacja błędów jako boxploty
plt.figure(figsize=(12, 6))
plt.boxplot(all_errors, labels=n_values, patch_artist=True, boxprops=dict(facecolor="skyblue", color="black"))
plt.xlabel('Długość trajektorii (n)')
plt.ylabel('Średni błąd e')
plt.title('Boxplot średniego błędu autokowariancji w zależności od długości trajektorii')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Zad 4
def autokowariancja(h, x):
    n = len(x)
    return 1/n * sum((x[i] - np.mean(x)) * (x[i + abs(h)] - np.mean(x)) for i in range(n-abs(h)))

def autokorelacja(h, x):
    return autokowariancja(h, x) / autokowariancja(0,x)

def autokowariancja_teo(h, x):
    if h == 0:
        return 4
    else:
        return 0

def autokorelacja_teo(h, x):
    if h == 0:
        return 1
    else:
        return 0
hs = np.arange(0, 30, 1)

sigma = 1
theta = 0.5

def ma1_teo_acvf(h, sigma, theta):
    if h == 0:
        return sigma**2 * (1 + theta**2)
    elif h == -1 or h == 1:
        return theta * sigma**2
    else:
        return 0

def ma1_teo_acf(h, theta):
    if h == 0:
        return 1
    elif h == 1 or h == -1:
        return theta / (1 + theta**2)
    else:
        return 0

def ma1_sample(n, sigma, theta):
    z = np.random.normal(0, sigma, n+1)
    x = np.zeros(n)
    for i in range(1,n+1):
        x[i-1] = z[i] + theta * z[i-1]
    return x

x = ma1_sample(1000, sigma, theta)


teo_acf, emp_acf, teo_acvf, emp_acvf = [], [], [], []
for h in hs:
    teo_acf.append(ma1_teo_acf(h, theta))
    emp_acf.append(autokorelacja(h, x))
    teo_acvf.append(ma1_teo_acvf(h, sigma, theta))
    emp_acvf.append(autokowariancja(h, x))


plt.plot(hs, teo_acf, "ro", label="teoretyczna")
plt.plot(hs, emp_acf, label="empiryczna")
plt.legend(loc="best")
plt.title("Funkcja autokorelacji")
plt.show()

plt.plot(hs, teo_acvf, "ro", label="teoretyczna")
plt.plot(hs, emp_acvf, label="empiryczna")
plt.legend(loc="best")
plt.title("Funkcja autokowariancji")
plt.show()