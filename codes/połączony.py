from shiny import App, ui, render, reactive
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
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
                ui.input_numeric("test_size", "Procent danych testowych", value=0.2, step=0.05),
                ui.input_text("new_sample", "Wprowadź nową próbkę (np. 5.1,3.5)"),
                ui.input_text("sample_data", "Wprowadź dane próbki (np. 5.1,3.5,1.4,0.2)"),
                ui.input_action_button("classify_button", "Klasyfikuj próbkę"),
                ui.input_action_button("classify", "Klasyfikuj próbkę"),
                ui.input_action_button("evaluate", "Oceń jakość drzewa")
            ),
            ui.panel_main(
                ui.tabs(
                    ui.tab_panel("Granice Drzewa", 
                                 ui.output_plot("decision_regions"),
                                 ui.output_text("classification_report"),
                                 ui.output_text("sample_classification"),
                                 ui.output_text("classification_results")),
                    ui.tab_panel("Schemat Drzewa",
                                 ui.output_plot("decision_tree"),
                                 ui.output_text("sample_classification"),
                                 ui.output_text("model_evaluation"))
                )
            )
        )
    )

def server_func(input, output, session):
    model = reactive.Value(DecisionTreeClassifier())
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :2].values, df['target'].values, test_size=input.test_size(), random_state=42)
    
    @reactive.Effect
    def update_model():
        model.set(DecisionTreeClassifier(max_depth=input.max_depth(), min_samples_split=input.min_samples_split()))
        model().fit(X_train, y_train)
    
    @output
    @render.plot
    def decision_regions():
        update_model()
        plt.figure(figsize=(10, 8))
        plot_decision_regions(X_train, y_train, clf=model(), legend=2)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title(f'Drzewo decyzyjne (Iris) - Dokładność: {accuracy_score(y_test, model().predict(X_test)) * 100:.2f}%')
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = [iris.target_names[int(label)] for label in labels]
        plt.legend(handles, labels, title="Kategorie")
        return plt.gcf()

    @output
    @render.plot
    def decision_tree():
        update_model()
        plt.figure(figsize=(25, 15), dpi=100)
        tree_plot = plot_tree(model(), 
                              filled=True, 
                              feature_names=iris.feature_names, 
                              class_names=iris.target_names, 
                              rounded=True, 
                              proportion=False, 
                              node_ids=True,
                              fontsize=10)
        plt.title("Drzewo Decyzyjne dla Zestawu Danych Iris")
        plt.tight_layout()
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
        
        
    # Pozostałe funkcje `sample_classification` i `model_evaluation` są analogiczne do kodu z powyższego przykładu.

app = App(ui=ui_func, server=server_func)

if __name__ == "__main__":
    app.run()
