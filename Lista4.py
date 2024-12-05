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

# Zad 5

n = 10000
sigma = 0.5
theta = 2
p = 0.01
a = 20
Z = np.random.normal(0, sigma, n)

X = np.zeros(n)
for t in range(1, n):
    X[t] = Z[t] + theta * Z[t - 1]

uniform_random = np.random.uniform(0, 1, n)

E = np.zeros(n)
E[uniform_random < p / 2] = a
E[(uniform_random >= p / 2) & (uniform_random < p)] = -a

Y = np.zeros(n)
for t in range(1, n):
    Y[t] = X[t] + E[t]


def autokowariancja(x, h1):
    h = abs(h1)
    n = len(x)
    if h >= n:
        return 0
    return (1 / (n - h)) * np.sum(np.sign((x[0:(n - h)] - np.median(x)) * (x[h:(n)] - np.median(x))))


def qautokow(autokowariancja):
    return np.sin(autokowariancja * np.pi / 2)


def theoretical_autocorrelation(theta, sigma, h):
    var_X = sigma ** 2 * (1 + theta ** 2)
    if h == 0:
        return 1
    elif h == 1:
        return theta * sigma ** 2 / var_X
    else:
        return 0


def autocovariance(X, h):
    n = len(X)
    X_mean = np.mean(X)
    return np.sum((X[:n - h] - X_mean) * (X[h:] - X_mean)) / n


def autocorrelation(X, h):
    return autocovariance(X, h) / autocovariance(X, 0)


# Tworzenie wykresów
fig, axs = plt.subplots(2, 2, figsize=(14, 7))

# Wykres trajektorii X
axs[0, 0].plot(X, label="Trajektoria X")
axs[0, 0].set_title("Trajektoria procesu X")
axs[0, 0].set_xlabel("Czas")
axs[0, 0].set_ylabel("Wartość")
axs[0, 0].legend()

# Wykres trajektorii Y
axs[0, 1].plot(Y, label="Trajektoria Y")
axs[0, 1].set_title("Trajektoria procesu Y")
axs[0, 1].set_xlabel("Czas")
axs[0, 1].set_ylabel("Wartość")
axs[0, 1].legend()

# Wykresy autokorelacji dla X
lags = np.arange(11)
autocov_X = [autokowariancja(X, h) for h in lags]
qcov_X = [qautokow(X) for X in autocov_X]
theo_X = [theoretical_autocorrelation(theta, sigma, h) for h in lags]
zad3_X = [autocorrelation(X, h) for h in lags]

axs[1, 0].scatter(lags, theo_X, color='black', marker='o', label="Teoretyczna autokorelacja")
axs[1, 0].plot(lags, zad3_X, label="Autokorelacja obliczona (X)")
axs[1, 0].plot(lags, qcov_X, color='red', label="q-autokorelacja (X)")
axs[1, 0].set_title("Autokorelacja dla procesu X")
axs[1, 0].set_xlabel("Lag")
axs[1, 0].set_ylabel("Autokorelacja")
axs[1, 0].legend()

# Wykresy autokorelacji dla Y
autocov_Y = [autokowariancja(Y, h) for h in lags]
qcov_Y = [qautokow(Y) for Y in autocov_Y]
theo_Y = [theoretical_autocorrelation(theta, sigma, h) for h in lags]
zad3_Y = [autocorrelation(Y, h) for h in lags]

axs[1, 1].scatter(lags, theo_Y, color='black', marker='o', label="Teoretyczna autokorelacja")
axs[1, 1].plot(lags, zad3_Y, label="Autokorelacja obliczona (Y)")
axs[1, 1].plot(lags, qcov_Y, color='red', label="q-autokorelacja (Y)")
axs[1, 1].set_title("Autokorelacja dla procesu Y")
axs[1, 1].set_xlabel("Lag")
axs[1, 1].set_ylabel("Autokorelacja")
axs[1, 1].legend()

# Dostosowanie układu
plt.tight_layout()
plt.show()

# zad 5 czesc dalsza


# Funkcje zoptymalizowane
def autokowariancja(h, x, x_mean):
    n = len(x)
    if h >= n:
        return 0
    return (1 / n) * np.sum((x[:n - h] - x_mean) * (x[h:] - x_mean))

def autokorelacja(h, x, x_mean, var_x):
    return autokowariancja(h, x, x_mean) / var_x

def ma1_teo_acf(h, theta):
    if h == 0:
        return 1
    elif h == 1 or h == -1:
        return theta / (1 + theta**2)
    else:
        return 0

def ma1_sample(n, sigma, theta):
    z = np.random.normal(0, sigma, n + 1)
    return z[1:] + theta * z[:-1]  # Wektoryzacja

# Parametry
n = 10000
sigma = 1
theta = 0.5
p_values = np.arange(0.01, 0.16, 0.01)  # p = 0.01, 0.02, ..., 0.15
a_values = np.arange(1, 11)  # a = 1, 2, ..., 10
M = 1000  # Liczba symulacji Monte Carlo

# Prealokacja wyników
e1_results = np.zeros((len(a_values), len(p_values)))
e2_results = np.zeros((len(a_values), len(p_values)))

for i, a in enumerate(a_values):
    for j, p in enumerate(p_values):
        e1_sum = 0
        e2_sum = 0
        for _ in range(M):
            # Generowanie próbki
            x = ma1_sample(n, sigma, theta)
            x_mean = np.mean(x)
            var_x = np.var(x)

            # Obliczenia teoretyczne i empiryczne
            ro_1_empirical = autokorelacja(1, x, x_mean, var_x)
            ro_1_teo = ma1_teo_acf(1, theta)
            ro_1_q_empirical = np.sin(ro_1_empirical * np.pi / 2)

            # Sumowanie błędów
            e1_sum += abs(ro_1_empirical - ro_1_teo)
            e2_sum += abs(ro_1_q_empirical - ro_1_teo)

        # Średnie błędy dla (a, p)
        e1_results[i, j] = e1_sum / M
        e2_results[i, j] = e2_sum / M

# Wizualizacja wyników w formie heatmap
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Heatmapa dla e1
c1 = axs[0].imshow(e1_results, aspect='auto', cmap='viridis', extent=[p_values[0], p_values[-1], a_values[0], a_values[-1]])
axs[0].set_title("Średni błąd e1")
axs[0].set_xlabel("p")
axs[0].set_ylabel("a")
axs[0].set_ylim(a_values[0], a_values[-1])  # Ustawienie osi Y na rosnąco
fig.colorbar(c1, ax=axs[0], label="e1")

# Heatmapa dla e2
c2 = axs[1].imshow(e2_results, aspect='auto', cmap='viridis', extent=[p_values[0], p_values[-1], a_values[0], a_values[-1]])
axs[1].set_title("Średni błąd e2")
axs[1].set_xlabel("p")
axs[1].set_ylabel("a")
axs[1].set_ylim(a_values[0], a_values[-1])  # Ustawienie osi Y na rosnąco
fig.colorbar(c2, ax=axs[1], label="e2")

plt.tight_layout()
plt.show()
