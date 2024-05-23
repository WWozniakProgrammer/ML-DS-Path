from shiny import App, ui, render, reactive
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Wczytanie danych Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

def ui_func(request):
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_slider("max_depth", "Maksymalna głębokość drzewa", 1, 10, 3),
                ui.input_slider("min_samples_split", "Minimalna liczba próbek do podziału", 2, 50, 2),
                ui.input_text("sample_data", "Wprowadź dane próbki (np. 5.1,3.5,1.4,0.2)"),
                ui.input_action_button("classify", "Klasyfikuj próbkę"),
                ui.input_action_button("evaluate", "Oceń jakość drzewa")
            ),
            ui.panel_main(
                ui.output_plot("decision_tree"),
                ui.output_text("sample_classification"),
                ui.output_text("model_evaluation")
            )
        )
    )

def server_func(input, output, session):
    model = reactive.Value(DecisionTreeClassifier())
    X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size=0.3, random_state=42)

    @reactive.Effect
    def _():
        input.max_depth()
        input.min_samples_split()
        model.set(DecisionTreeClassifier(max_depth=input.max_depth(), min_samples_split=input.min_samples_split()))
        model().fit(X_train, y_train)

    @output
    @render.plot
    def decision_tree():
        # Zwiększanie rozmiaru figury i rozdzielczości dla lepszej czytelności
        plt.figure(figsize=(25, 15), dpi=100)  # Zwiększ rozmiar figury i DPI
        tree_plot = plot_tree(model(), 
                            filled=True, 
                            feature_names=iris.feature_names, 
                            class_names=iris.target_names, 
                            rounded=True, 
                            proportion=False, 
                            node_ids=True,
                            fontsize=10)  # Możesz dostosować rozmiar czcionki dla lepszej czytelności
        plt.title("Drzewo Decyzyjne dla Zestawu Danych Iris")
        plt.tight_layout()  # Poprawia rozmieszczenie aby ograniczyć nachodzenie na siebie
        return plt.gcf()

    @output
    @render.text
    def sample_classification():
        if input.classify():
            try:
                sample = np.array(eval(f"[{input.sample_data()}]"))
                prediction = iris.target_names[model().predict([sample])[0]]
                path = export_text(model(), feature_names=iris.feature_names, decision_path=True)
                return f"Klasyfikacja próbki: {prediction}\nŚcieżka decyzyjna:\n{path}"
            except Exception as e:
                return f"Błąd: {str(e)}"

    @output
    @render.text
    def model_evaluation():
        if input.evaluate():
            y_pred = model().predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=iris.target_names)
            return f"Dokładność modelu: {acc*100:.2f}%\nMacierz pomyłek:\n{conf_matrix}\nRaport klasyfikacji:\n{report}"

app = App(ui=ui_func, server=server_func)

if __name__ == "__main__":
    app.run()
