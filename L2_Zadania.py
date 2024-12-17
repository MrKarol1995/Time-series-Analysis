
import seaborn, scipy.stats as stats
import numpy as np, matplotlib.pyplot as plt

#Zad 3
sigma = 1.0
n = 1000
M1 = 1000
M2 = 100
b0s = {}
b1s = {}
b0_true = 2.0
b1_true = 4.0
x = np.linspace(0, 10, n)
errorsEb1 = {}
errorsvarb1 = {}

def b1_emp(x,y):
    return np.sum((x - np.mean(x)) * (y))/np.sum((x-np.mean(x))**2)

def b0_emp(x,y):
    return np.mean(y) - b1_emp(x,y) * np.mean(x)
def sb0(x, sigma = sigma):
    n = len(x)
    x_ = np.mean(x)
    return sigma*1/np.sqrt(1/n + x_**2/sum((xi - x_)**2 for xi in x))
def sb1(x, sigma = sigma):
    x_ = np.mean(x)
    return sigma * 1/np.sqrt(sum((xi - x_)**2 for xi in x))


b1emp = []
b0emp = []
for _ in range(M1):
    eps = np.random.normal(0, sigma**2, size = n)
    y = b0_true + b1_true * x + eps
    b1emp.append(b1_emp(x, y))
    b0emp.append(b0_emp(x,y))

seaborn.ecdfplot(data = b1emp, label = 'empiryczna' )
xs = np.linspace(3.9, 4.1, 1000)
plt.plot(xs, stats.norm.cdf(xs, np.mean(b1emp), sb1(x, sigma)), label = 'teoretyczna')
plt.title('b1')
plt.legend()
plt.show()

plt.hist(b1emp, density = True)
seaborn.kdeplot(b1emp, label= 'gestość empiryczna')
plt.title('b1')
plt.show()

seaborn.ecdfplot(data = b0emp, label = 'empiryczna' )
xs = np.linspace(1.8, 2.2, 1000)
plt.plot(xs, stats.norm.cdf(xs, np.mean(b0emp), sb0(x, sigma)), label = 'teoretyczna')
plt.legend()
plt.show()
print(sb0(x, sigma))

plt.hist(b0emp, density = True)
seaborn.kdeplot(b0emp, label= 'gestość empiryczna')
plt.title('b0')
plt.show()
