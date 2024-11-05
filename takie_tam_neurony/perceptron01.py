import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000, activation='step'):
        self.w = np.zeros(input_size + 1)  # +1 dla biasu
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation

    def net_input(self, X):
        """Oblicza sumę ważoną (net input)."""
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation_function(self, z):
        """Zastosowanie funkcji aktywacji."""
        if self.activation == 'step':
            return np.where(z >= 0.0, 1, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Nieznana funkcja aktywacji.")

    def predict(self, X):
        """Przewidywanie klasy dla danych wejściowych X."""
        z = self.net_input(X)
        return self.activation_function(z)

    def fit(self, X, y):
        """Trenuje perceptron na danych treningowych."""
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                z = self.net_input(xi)
                output = self.activation_function(z)
                error = target - output
                self.w[1:] += self.learning_rate * error * xi
                self.w[0] += self.learning_rate * error

    def score(self, X, y):
        """Ocena dokładności modelu na danych testowych."""
        predictions = self.predict(X)
        predicted_classes = np.where(predictions >= 0.5, 1, 0)
        return np.mean(predicted_classes == y)

# Przykład użycia:

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
