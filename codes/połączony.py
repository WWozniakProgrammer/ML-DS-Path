from shiny import App, ui, render
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Wczytanie danych Iris
iris = load_iris()

# Konwersja do DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

def ui_func(request):
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_select("var","Wybierz funkcjonalności: ", choices=["Drzewo decyzyjne", "Granice decyzyjne"]),
                ui.input_slider("max_depth", "Maksymalna głębokość drzewa", 1, 10, 3),
                ui.input_slider("min_samples_split", "Minimalna liczba próbek do podziału", 2, 50, 2),
                ui.input_numeric("test_size", "Procent danych testowych", value=0.2, step=0.05),
                ui.input_text("new_sample", "Wprowadź nową próbkę (np. 5.1,3.5)"),
                ui.input_action_button("classify_button", "Klasyfikuj próbkę")
            ),
            ui.panel_main(
                ui.output_plot("decision_regions"),
                ui.output_text("classification_report"),
                ui.output_text("sample_classification"),
                ui.output_text("classification_results")
            )
        )
    )

def server_func(input, output, session):
    @output
    @render.plot
    def decision_regions():
        max_depth = input.max_depth()
        min_samples_split = input.min_samples_split()
        test_size = input.test_size()
        df = iris_df
        feature_names = iris.feature_names
        target_names = iris.target_names

        X = df.iloc[:, :2].values  # Używamy tylko pierwszych dwóch cech dla uproszczenia wizualizacji
        y = df['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)

        plt.figure(figsize=(10, 8))
        plot_decision_regions(X, y, clf=model, legend=2)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'Drzewo decyzyjne (Iris) - Dokładność: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%')
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = [target_names[int(label)] for label in labels]
        plt.legend(handles, labels, title="Kategorie")
        return plt.gcf()

    @output
    @render.text
    def sample_classification():
        if input.classify_button() > 0:
            new_sample = input.new_sample()
            try:
                sample_values = np.array([float(num) for num in new_sample.split(',')]).reshape(1, -2)
                if sample_values.shape[1] != 2:
                    return "Wprowadź prawidłową próbkę z dwiema wartościami oddzielonymi przecinkiem."
            except ValueError:
                return "Wprowadź prawidłowe wartości numeryczne."

            model = DecisionTreeClassifier(max_depth=input.max_depth(), min_samples_split=input.min_samples_split())
            model.fit(iris_df.iloc[:, :2].values, iris_df['target'].values)
            predicted_class_index = model.predict(sample_values)[0]  # Pobierz pierwszy element z wyników
            predicted_class = iris.target_names[int(predicted_class_index)]

            return f"Przewidziana klasa dla próbki [{new_sample}]: {predicted_class}"
        return "Kliknij 'Klasyfikuj próbkę' aby uzyskać wyniki."

    @output
    @render.text
    def classification_results():
        max_depth = input.max_depth()
        min_samples_split = input.min_samples_split()
        test_size = input.test_size()
        df = iris_df

        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :2].values, df['target'].values, test_size=test_size, random_state=42)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Generowanie macierzy pomyłek
        cm = confusion_matrix(y_test, y_pred)

        # Formatowanie macierzy pomyłek w czytelny sposób
        cm_lines = ["|" + "  ".join(f"{num:2d}" for num in row) + " |" for row in cm]
        cm_formatted = "\n".join(cm_lines)

        return f"\nMacierz pomyłek:\n{cm_formatted}"


app = App(ui=ui_func, server=server_func)

if __name__ == "__main__":
    app.run()
