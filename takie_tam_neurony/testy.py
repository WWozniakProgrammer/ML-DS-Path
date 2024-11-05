from perceptron01 import Perceptron
import numpy as np

# Dane treningowe (bramka AND)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])  # Wyniki dla bramki AND (0 reprezentuje False, 1 reprezentuje True)

# Inicjalizacja i trening perceptronu
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

# Testowanie perceptronu
for xi in X:
    print(f"Wejście: {xi}, Wyjście: {perceptron.predict(xi)}")

# Ocena modelu
accuracy = perceptron.score(X, y)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")