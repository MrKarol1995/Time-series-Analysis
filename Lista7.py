import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Zad 1

def estimate_ar2_yule_walker():
    # Ustawienia symulacji
    n_simulations = 1000  # Liczba symulacji Monte Carlo
    n_samples = 500       # Liczba próbek w pojedynczej symulacji
    ar_params = [1.5, -0.75]  # Prawdziwe parametry modelu AR(2)

    # Przechowywanie estymacji parametrów AR(2)
    ar1_estimates = []
    ar2_estimates = []

    for _ in range(n_simulations):
        # Generowanie danych z modelu AR(2)
        ar = np.array([1, -ar_params[0], -ar_params[1]])  # AR(2) z odwróconymi znakami
        ma = np.array([1])  # MA(0)
        model = ArmaProcess(ar, ma)
        data = model.generate_sample(n_samples)

        # Estymacja parametrów metodą Yule-Walkera
        rho, sigma = sm.regression.yule_walker(data, order=2)
        ar1_estimates.append(rho[0])  # Estymowany AR(1)
        ar2_estimates.append(rho[1])  # Estymowany AR(2)

    # Konwersja wyników do numpy array
    ar1_estimates = np.array(ar1_estimates)
    ar2_estimates = np.array(ar2_estimates)

    # Wizualizacja wyników
    plt.figure(figsize=(12, 6))
    plt.boxplot([ar1_estimates, ar2_estimates], tick_labels=["AR(1)", "AR(2)"])
    plt.axhline(y=ar_params[0], color="r", linestyle="--", label=f"AR(1) True: {ar_params[0]}")
    plt.axhline(y=ar_params[1], color="g", linestyle="--", label=f"AR(2) True: {ar_params[1]}")
    plt.title("Estymacja parametrów AR(2) metodą Yule-Walkera")
    plt.ylabel("Estymowane wartości")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    # Wyświetlanie średnich wyników
    print("Średnia estymacja AR(1):", np.mean(ar1_estimates))
    print("Średnia estymacja AR(2):", np.mean(ar2_estimates))
    print("Rzeczywiste wartości AR(2):", ar_params)

# Uruchomienie funkcji

print("Zadanie 1: \n")
estimate_ar2_yule_walker()

#Zad 2

def compare_ar1_methods():
    # Ustawienia symulacji
    n_simulations = 1000
    n_samples = 500
    ar_param = 0.8  # Parametr AR(1)

    yule_walker_estimates = []
    mle_estimates = []

    for _ in range(n_simulations):
        # Generowanie danych AR(1)
        ar = np.array([1, -ar_param])
        ma = np.array([1])
        model = ArmaProcess(ar, ma)
        data = model.generate_sample(n_samples)

        # Estymacja Yule-Walkera
        rho, sigma = sm.regression.yule_walker(data, order=1)
        yule_walker_estimates.append(rho[0])

        # Estymacja MLE
        arma_model = sm.tsa.ARIMA(data, order=(1, 0, 0)).fit()
        mle_estimates.append(arma_model.arparams[0])

    print("Średnia estymacja Yule-Walker:", np.mean(yule_walker_estimates))
    print("Średnia estymacja MLE:", np.mean(mle_estimates))
    print("Rzeczywisty parametr AR(1):", ar_param)

    # Wizualizacja
    plt.figure(figsize=(12, 6))
    plt.boxplot([yule_walker_estimates, mle_estimates], tick_labels=["Yule-Walker", "MLE"])
    plt.axhline(y=ar_param, color="r", linestyle="--", label=f"True AR(1): {ar_param}")
    plt.title("Porównanie estymacji AR(1): Yule-Walker vs MLE")
    plt.ylabel("Estymowane wartości")
    plt.legend()
    plt.show()
print("Zadanie 2: \n")
compare_ar1_methods()

#Zad 3
def ma1_pacf_confidence_intervals():
    # Ustawienia symulacji
    n_simulations = 1000
    n_samples = 500
    theta = 0.5  # Parametr MA(1)
    sigma = 1  # Odchylenie standardowe białego szumu
    h = 20  # Maksymalne opóźnienie

    pacf_values = []

    for _ in range(n_simulations):
        # Generowanie danych MA(1)
        ar = np.array([1])
        ma = np.array([1, theta])
        model = ArmaProcess(ar, ma)
        data = model.generate_sample(n_samples, scale=sigma)

        # Obliczanie PACF
        pacf = sm.tsa.stattools.pacf(data, nlags=h)
        pacf_values.append(pacf)

    pacf_values = np.array(pacf_values)
    pacf_means = np.mean(pacf_values, axis=0)
    pacf_conf_int = np.percentile(pacf_values, [2.5, 97.5], axis=0)

    # Wizualizacja
    plt.figure(figsize=(12, 6))
    lags = np.arange(h + 1)
    plt.plot(lags, pacf_means, marker="o", label="Średnie wartości PACF")
    plt.fill_between(lags, pacf_conf_int[0], pacf_conf_int[1], color="gray", alpha=0.3, label="95% przedziały ufności")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Średnie wartości PACF dla modelu MA(1) i przedziały ufności")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.legend()
    plt.show()

    print("Średnie wartości PACF:", pacf_means)
    print("Przedziały ufności dla PACF:")
    for i in range(h + 1):
        print(f"Lag {i}: {pacf_conf_int[:, i]}")


print("Zadanie 3: \n")
ma1_pacf_confidence_intervals()

#zad inne/4
print("Zadanie 4: \n")
# Parametry modelu AR(2)
phi1 = 0.2
phi2 = 0.4
sigma2 = 1

# Współczynniki AR i MA
ar = np.array([1, -phi1, -phi2])
ma = np.array([1])

# Tworzenie procesu AR(2)
arma_process = ArmaProcess(ar, ma)

# Generowanie próbek
n_samples = 500
np.random.seed(42)
X = arma_process.generate_sample(nsample=n_samples, scale=np.sqrt(sigma2))

# Funkcja obliczająca empiryczną autokowariancję
def empirical_autocovariance(series, lag):
    n = len(series)
    mean = np.mean(series)
    return np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / n

# Obliczenie empirycznej ACVF
h_max = 20
empirical_gamma = [empirical_autocovariance(X, lag) for lag in range(h_max + 1)]

# Teoretyczna autokowariancja
gamma_0 = sigma2 / (1 - phi1**2 - phi2**2 - 2 * phi1 * phi2)
gamma_1 = phi1 * gamma_0 + phi1 * phi2 * gamma_0
gamma_2 = phi2 * gamma_0

theoretical_gamma = [gamma_0, gamma_1, gamma_2]

# Wyświetlenie wyników
print("Empiryczna ACVF:")
for lag, gamma in enumerate(empirical_gamma[:3]):  # Porównanie do lag 2
    print(f"Lag {lag}: {gamma:.4f}")

print("\nTeoretyczna ACVF:")
for lag, gamma in enumerate(theoretical_gamma):
    print(f"Lag {lag}: {gamma:.4f}")

# Wykres porównawczy
plt.figure(figsize=(10, 6))

# Dodanie empirycznych punktów
plt.stem(range(h_max + 1), empirical_gamma, linefmt="b-", markerfmt="bo", basefmt=" ", label="Empiryczna ACVF")

# Dodanie teoretycznych punktów z przesunięciem -0.1
offset_lags = np.arange(3) - 0.1
plt.stem(offset_lags, theoretical_gamma, linefmt="r-", markerfmt="rs", basefmt=" ", label="Teoretyczna ACVF")

# Tytuł, legenda i osie
plt.title("Porównanie empirycznej i teoretycznej ACVF")
plt.xlabel("Opóźnienie (lag)")
plt.ylabel("Autokowariancja")
plt.xticks(range(h_max + 1))
plt.legend()
plt.grid()
plt.show()