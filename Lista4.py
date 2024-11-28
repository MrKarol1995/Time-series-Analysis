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

# Główna funkcja generująca dane i obliczenia
def main(n, max_lag):
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

# Ustawienia i uruchomienie
if __name__ == '__main__':
    n = 1000  # Rozmiar próby
    max_lag = 20  # Maksymalny lag
    main(n, max_lag)

# Zad 4

def ma1teoACVF(h, sigma, theta):
    if h == 0:
        return sigma**2*(1+theta**2)
    if abs(h) == 1:
        return theta*sigma**2
    else:
        return 0

def ma1teoACF(h, theta):
    if h == 0:
        return 1
    if abs(h) == 1:
        return theta/(1+theta**2)
    else:
        return 0

def zad4(n, h, sigma, theta):
    Z = np.random.normal(0, sigma**2, size = n+1)
    X = Z[1:] + theta*Z[:-1]
    empACF = sampleACF(X, h)
    trueACF = ma1teoACF(h, theta)
    return n,h,sampleACVF(X,h),ma1teoACVF(h, sigma, theta), empACF,trueACF


sigma = 1
theta = 2
wyniki2 = []
for h in list(range(-50, 51)):
    for n in [1000, 10000]:
        wyniki2.append(zad4(n, h, sigma, theta))

df2 = pd.DataFrame(wyniki2, columns=['n', 'h', 'sampleACVF', 'True ACVF', 'sampleACF', 'True ACF'])

plt.figure(figsize=(10, 6))
for n in sorted(df2['n'].unique()):
    subset = df2[df['n'] == n]
    plt.scatter(subset['h'], subset['sampleACVF'], marker='o', label=f'n = {n}')
plt.scatter(subset['h'], subset['True ACVF'], color = 'k', alpha= 0.7, marker='*', label = 'True ACVF')
# acfs = df2['True ACVF'].unique()
# for value in acfs:
#     plt.axhline(y=value, color='r', linestyle='--', alpha=0.7, label=f'True ACVF={value}')

plt.ylabel('Sample ACVF')
plt.title('Empiryczna ACVF w zależności od h')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(10, 6))
for n in sorted(df2['n'].unique()):
    subset = df2[df['n'] == n]
    plt.scatter(subset['h'], subset['sampleACF'], marker='o', label=f'n = {n}')
plt.scatter(subset['h'], subset['True ACF'], color = 'k', alpha= 0.5, marker='*', label = 'True ACF')
# acfs = df2['True ACF'].unique()
# for value in acfs:
#     plt.axhline(y=value, color='r', linestyle='--', alpha=0.7, label=f'True ACF={value}')

plt.ylabel('Sample ACF')
plt.title('Empiryczna ACF w zależności od h')
plt.legend()
plt.grid()
#plt.show()