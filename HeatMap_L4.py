from joblib import Parallel, delayed
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def autokowariancja(h, x):
    n = len(x)
    mean_x = np.mean(x)
    return 1/n * np.sum((x[:n-abs(h)] - mean_x) * (x[abs(h):] - mean_x))

def ma1_w_noise(n, sigma, theta, a, p):
    y = ma1_sample(n, sigma, theta)
    noise = np.random.random(n)
    x = np.where(noise < p/2, y + a, np.where(noise < p, y - a, y))
    return x, y
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



def ro_star(h, x):
    n = len(x)
    median_x = np.median(x)
    return 1/(n-h) * np.sum(np.sign((x[:n-abs(h)] - median_x) * (x[abs(h):] - median_x)))
def ro(h, x):
    return np.sin(np.pi/2 * ro_star(h,x))

def compute_sums(a, p, num_samples=100, n=1000, sigma=1, theta=2):
    teo_acf1 = ma1_teo_acf(1, theta)
    sum1, sum2 = 0, 0
    for _ in range(num_samples):
        sample_noisy, _ = ma1_w_noise(n, sigma, theta, a, p)
        emp_acf1_noisy = autokowariancja(1, sample_noisy) / autokowariancja(0, sample_noisy)
        emp_acf2_noisy = ro(1, sample_noisy)
        sum1 += abs(teo_acf1 - emp_acf1_noisy)
        sum2 += abs(teo_acf1 - emp_acf2_noisy)
    return sum1 / num_samples, sum2 / num_samples

a_s = np.arange(1, 11, 1)
p_s = np.arange(0.01, 0.16, 0.01)

results = Parallel(n_jobs=-1)(
    delayed(compute_sums)(a, p) for a in a_s for p in p_s
)
e1_a = np.array([res[0] for res in results]).reshape(len(a_s), len(p_s))
e2_a = np.array([res[1] for res in results]).reshape(len(a_s), len(p_s))


# Obliczenie wspólnego zakresu dla obu heatmap
vmin = min(e1_a.min(), e2_a.min())
vmax = max(e1_a.max(), e2_a.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 wiersz, 2 kolumny

# Heatmapa dla e1_a
sns.heatmap(e1_a, annot=False, cmap="viridis",
            xticklabels=p_s.round(2), yticklabels=a_s.round(5),
            vmin=vmin, vmax=vmax, ax=axes[0])
axes[0].set_title("Heatmapa dla e1_a")
axes[0].set_xlabel("p")
axes[0].set_ylabel("a")
axes[0].invert_yaxis()

# Heatmapa dla e2_a
sns.heatmap(e2_a, annot=False, cmap="viridis",
            xticklabels=p_s.round(2), yticklabels=a_s.round(5),
            vmin=vmin, vmax=vmax, ax=axes[1])
axes[1].set_title("Heatmapa dla e2_a")
axes[1].set_xlabel("p")
axes[1].set_ylabel("a")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
