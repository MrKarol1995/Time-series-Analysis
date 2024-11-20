import numpy as np
import matplotlib.pyplot as plt

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

sigma1 = 1
sigma2 = 5

x1 = np.random.normal(0,np.sqrt(sigma1),400)
x2 = np.random.normal(0,np.sqrt(sigma2),600)

x = np.concatenate((x1,x2))
xs = np.linspace(1,1000,1000)

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

l = estimate_change_point(c)

print(l)
