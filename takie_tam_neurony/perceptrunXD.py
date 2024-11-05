import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.w = np.zeros(input_size + 1)  # +1 dla biasu
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, x):
        z = np.dot(x, self.w[1:]) + self.w[0]  # w[0] to bias
        return self.activation_function(z)

    def fit(self, X, y): # X to dane treningowe, y to wyniki
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                if target != output:
                    update = self.learning_rate * target
                    self.w[1:] += update * xi
                    self.w[0] += update  # Aktualizacja biasu

# Przykład użycia:

# Dane treningowe (bramka AND)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([-1, -1, -1, 1])  # Wyniki dla bramki AND (-1 reprezentuje False, 1 reprezentuje True)

# Inicjalizacja i trening perceptronu
perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

# Testowanie perceptronu
for xi in X:
    print(f"Wejście: {xi}, Wyjście: {perceptron.predict(xi)}")

print(perceptron.predict([0,1]))