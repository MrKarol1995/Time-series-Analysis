import numpy as np, matplotlib.pyplot as plt, pandas as pd

# Zad 3

def sampleACVF(X, h):
    n = len(X)
    x_mean = np.mean(X)
    return 1/n*sum((X[i]-x_mean)*(X[i+abs(h)]-x_mean) for i in range(1,n-abs(h)))

def sampleACF(X, h):
    return sampleACVF(X, h)/sampleACVF(X, 0)

def teoACVF(h):
    return 4 if h==0 else 0

def teoACF(h):
    return 1 if h==0 else 0
def zad3(n,h):
    X = np.random.normal(0, 2, size = n)
    return n,h,sampleACVF(X, h), sampleACF(X, h)


wyniki = []
for h in list(range(-50, 51)):
    for n in [1000, 10000]:
        n, h, acvf, acf = zad3(n, h)
        wyniki.append([n, h, acvf, teoACVF(h), acf, teoACF(h)])

df = pd.DataFrame(wyniki, columns=['n', 'h', 'sampleACVF', 'True ACVF', 'sampleACF', 'True ACF'])


plt.figure(figsize=(10, 6))
for n in sorted(df['n'].unique()):
    subset = df[df['n'] == n]
    plt.scatter(subset['h'], subset['sampleACVF'], marker='o', label=f'n = {n}')
plt.scatter(subset['h'], subset['True ACVF'], color = 'k', alpha=0.5, marker='*', label = 'True ACVF')

plt.axhline(0, color='red', linestyle='dashed', linewidth=0.8, label = 'teoretyczna')  # Linia dla ACVF = 0
plt.axhline(4, color='red', linestyle='dashed', linewidth=0.8, label = 'teoretyczna')  # Linia dla ACVF = 4
plt.ylabel('Sample ACVF')
plt.title('Empiryczna ACVF w zależności od h')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
for n in sorted(df['n'].unique()):
    subset = df[df['n'] == n]
    plt.scatter(subset['h'], subset['sampleACF'], marker='o', label=f'n = {n}')
plt.scatter(subset['h'], subset['True ACF'], color = 'k', alpha=0.5, marker='*', label = 'True ACF')

plt.axhline(0, color='red', linestyle='dashed', linewidth=0.8)
plt.axhline(1, color='red', linestyle='dashed', linewidth=0.8)
plt.ylabel('Sample ACF')
plt.title('Empiryczna ACF w zależności od h')
plt.legend()
plt.grid()
plt.show()

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
plt.show()

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
plt.show()