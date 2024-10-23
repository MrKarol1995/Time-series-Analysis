import numpy as np
from scipy import linalg

# Zad 4
def gauss_elimination(A, b):
    n = len(b)

    # Tworzenie rozszerzonej macierzy układu [A | b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Eliminacja Gaussa
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        if Ab[i, i] == 0:
            raise ValueError("Macierz jest osobliwa (nieodwracalna)!")

        Ab[i] = Ab[i] / Ab[i, i]

        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]

    # Back substitution (wyznaczanie rozwiązania)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])

    return x


# Zad 4:
A4 = np.array([[0, 0, 2, 1, 2],
              [0, 1, 0, 2, -1],
              [1, 2, 0, -2, 0],
              [0, 0, 0, -1, 1],
              [0, 1, -1, 1, -1]], dtype=float)

b4 = np.array([1, 1, -4, -2, -1], dtype=float)

x_gauss = gauss_elimination(A4, b4)

print("Zad 4: Rozwiązanie eliminacją Gaussa:")
print(x_gauss)
# Zad 5: Interpolacja wielomianowa
points = np.array([[0, -1],
                   [1, 1],
                   [3, 3],
                   [5, 2],
                   [6, -2]])

x5 = points[:, 0]
y5 = points[:, 1]

A5 = np.vander(x5, 5)
coefficients_scipy = linalg.solve(A5, y5)

print("Zad 5: Współczynniki wielomianu (scipy.linalg.solve):")
print(coefficients_scipy)

# Zad 6: Układ równań
A6 = [
    [3.50, 2.77, -0.76, 1.80],
    [-1.80, 2.68, 3.44, -0.09],
    [0.27, 5.07, 6.90, 1.61],
    [1.71, 5.45, 2.68, 1.71]
]

b6 = np.array([7.31, 4.23, 13.85, 11.55])

x6_scipy = linalg.solve(A6, b6)
det_A6 = np.linalg.det(A6)
Ax6 = np.dot(A6, x6_scipy)
def get_minor(matrix, i, j):
    """Zwraca macierz pomniejszoną o i-ty wiersz i j-tą kolumnę."""
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def is_square(matrix):
    """Sprawdza, czy macierz jest kwadratowa."""
    return all(len(row) == len(matrix) for row in matrix)

def determinant(matrix):
    """Rekurencyjnie oblicza wyznacznik macierzy."""
    # Sprawdzamy, czy macierz jest kwadratowa
    if not is_square(matrix):
        raise ValueError("Macierz musi być kwadratowa.")

    # Przypadek bazowy dla macierzy 2x2
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Rekurencyjny przypadek dla macierzy większych
    det = 0
    for j in range(len(matrix)):
        det += ((-1) ** j) * matrix[0][j] * determinant(get_minor(matrix, 0, j))
    return det


print("Zad 6: Rozwiązanie układu eliminacja gaussa:")
print(gauss_elimination(A6,b6))
# Obliczanie wyznacznika
try:
    result6 = determinant(A6)
    print(f"\nWyznacznik macierzy per se:\n{result6}")
except ValueError as e:
    print(e)

print("\nWyznacznik macierzy A (numpy.linalg.det):")
print(det_A6)
print("\nIloczyn A * x (taki jak b_hat):")
print(Ax6)

# Zad 7: Macierz 8x8
A7 = [[10, -2, -1, 2, 3, 1, -4, 7],
              [5, 11, 3, 10, -3, 3, 3, -4],
              [7, 12, 1, 5, 3, -12, 2, 3],
              [8, 7, -2, 1, 3, 2, 2, 4],
              [2, -15, -1, 1, 4, -1, 8, 3],
              [4, 2, 9, 1, 12, -1, 4, 1],
              [-1, 4, -7, -1, 1, 1, -1, -3],
              [-1, 3, 4, 1, 3, -4, 7, 6]]

b7 = np.array([0, 12, -5, 3, -25, -26, 9, -7])

x7_scipy = linalg.solve(A7, b7)

print("Zad 7: Rozwiązanie układu (scipy.linalg.solve):")
print(x7_scipy)

# Zad 8: Macierz trójdiagonalna
A8 = [[2, -1, 0, 0, 0, 0],
              [-1, 2, -1, 0, 0, 0],
              [0, -1, 2, -1, 0, 0],
              [0, 0, -1, 2, -1, 0],
              [0, 0, 0, -1, 2, -1],
              [0, 0, 0, 0, -1, 5]]

A8_inv = np.linalg.inv(A8)

def is_tridiagonal(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

print("\nZad 8: Macierz odwrotna A^-1:")
print(A8_inv)
print("\nCzy macierz A^-1 jest trójdiagonalna?")
print(is_tridiagonal(A8_inv))

# Zad 9

print("Zad 9")

def print_matrix(matrix):
    """Pomocnicza funkcja do ładnego wyświetlania macierzy."""
    for row in matrix:
        print(row)


def gauss_jordan_inverse(matrix):
    n = len(matrix)

    # Tworzymy macierz rozszerzoną: oryginalna macierz po lewej + macierz jednostkowa po prawej
    augmented_matrix = [matrix[i] + [float(i == j) for j in range(n)] for i in range(n)]

    # Przekształcanie macierzy do formy macierzy jednostkowej po lewej stronie
    for i in range(n):
        # Normalizujemy wiersz tak, aby element diagonalny wynosił 1
        diagonal_element = augmented_matrix[i][i]
        if diagonal_element == 0:
            raise ValueError("Macierz nie jest odwracalna.")

        for j in range(2 * n):
            augmented_matrix[i][j] /= diagonal_element

        # Zerowanie pozostałych elementów w kolumnie i
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Wyodrębniamy macierz odwrotną z rozszerzonej macierzy
    inverse_matrix = [row[n:] for row in augmented_matrix]

    return inverse_matrix


# Macierz A z obrazka
A9 = [
    [1, 3, -9, 6, 4],
    [2, -1, 6, 7, 1],
    [3, 2, -3, 15, 5],
    [8, -1, 1, 4, 2],
    [11, 1, -2, 18, 7]
]

# Znajdowanie macierzy odwrotnej
try:
    inverse_A = gauss_jordan_inverse(A9)
    print("Macierz odwrotna do A:")
    print_matrix(inverse_A)
except ValueError as e:
    print(e)



