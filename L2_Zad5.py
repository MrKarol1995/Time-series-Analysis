import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

sigma1 = 1
sigma2 = 5

x1 = np.random.normal(0,np.sqrt(sigma1), 400)
x2 = np.random.normal(0,np.sqrt(sigma2), 600)

x = np.concatenate((x1,x2))
xs = np.linspace(1, 1000, 1000)

plt.plot(xs, x)
plt.show()

c = np.cumsum(x**2)

plt.plot(xs, c)
plt.show()

k = 300
c1, c2 = c[:k], c[k:]

b0_1, b1_1 = prosta_regresji(xs[:k], c1)
b0_2, b1_2 = prosta_regresji(xs[k:], c2)

y1 = lambda k: b1_1 * xs[:k] + b0_1
y2 = lambda k: b1_2 * xs[k:] + b0_1

func = lambda k: np.sum((c[:k] - y1(k)) ** 2) + np.sum((c[k:] - y2(k)) ** 2)
ks = [i for i in range(1, 1001)]
f_values = [func(k_) for k_ in ks]

print(np.argmin(f_values))


def estimate_change_point(c):
    n = len(c)
    best_k = None
    min_sum_squares = np.inf

    for k in range(1, n - 1):

        y1 = b1_1 * xs[:k] + b0_1
        y2 = b1_2 * xs[k:] + b0_1

        sum_squares = np.sum((c[:k] - y1) ** 2) + np.sum((c[k:] - y2) ** 2)

        if sum_squares < min_sum_squares:
            min_sum_squares = sum_squares
            best_k = k

    return best_k


# Funkcja minimalizacji
def func(k):
    y1 = b1_1 * xs[:k] + b0_1
    y2 = b1_2 * xs[k:] + b0_2
    return np.sum((c[:k] - y1) ** 2) + np.sum((c[k:] - y2) ** 2)

# Oblicz wartości funkcji dla zakresu k
ks = np.arange(1, 1000)
f_values = [func(k_) for k_ in ks]

# Znajdź minimalne k i odpowiadające minimum
min_k = ks[np.argmin(f_values)]
min_value = np.min(f_values)

# Wykres funkcji błędu
plt.figure(figsize=(10, 6))
plt.plot(ks, f_values, label="Sum of Squares Error", color="blue")
plt.scatter(min_k, min_value, color="red", label=f"Min at k={min_k}")
plt.title("Error Function vs Change Point")
plt.xlabel("Change Point k")
plt.ylabel("Sum of Squares Error")
plt.legend()
plt.grid()
plt.show()


# Parametry
n = 1000  # całkowita liczba punktów
L = 400  # długość pierwszego segmentu # Zmieniać w zależności od potrzeb
sigma1 = 1
sigma2_values = [1.5, 3, 4.5, 6]
M = 200  # liczba symulacji dla każdej sigma2


# Funkcja do generowania danych i estymacji punktu zmiany
def generate_data_and_estimate(L, sigma1, sigma2, n):
    x1 = np.random.normal(0, sigma1, L)
    x2 = np.random.normal(0, sigma2, n - L)
    x = np.concatenate((x1, x2))
    c = np.cumsum(x ** 2)

    def func(k):
        y1 = b1_1 * xs[:k] + b0_1
        y2 = b1_2 * xs[k:] + b0_2
        return np.sum((c[:k] - y1) ** 2) + np.sum((c[k:] - y2) ** 2)

    # Oblicz funkcję regresji dla dwóch segmentów
    xs = np.arange(1, n + 1)
    b0_1, b1_1 = prosta_regresji(xs[:L], c[:L])
    b0_2, b1_2 = prosta_regresji(xs[L:], c[L:])

    # Znajdź minimalne k
    ks = np.arange(1, n)
    f_values = [func(k) for k in ks]
    min_k = ks[np.argmin(f_values)]
    return min_k


# Przeprowadź symulacje
results = {sigma2: [] for sigma2 in sigma2_values}

for sigma2 in sigma2_values:
    for _ in range(M):
        k_estimate = generate_data_and_estimate(L, sigma1, sigma2, n)
        results[sigma2].append(k_estimate)

# Przygotowanie danych do wykresu


data = []
for sigma2, ks in results.items():
    data.extend([(sigma2, k) for k in ks])

df = pd.DataFrame(data, columns=["Sigma2", "Change Point"])

# Tworzenie box plotu
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Sigma2", y="Change Point", palette="coolwarm")
plt.title("Change Point Estimates for Different Sigma2 Values")
plt.xlabel("Sigma2")
plt.ylabel("Change Point Estimate")
plt.grid(axis="y")
plt.show()
