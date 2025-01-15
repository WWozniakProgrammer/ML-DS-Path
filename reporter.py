import os
import datetime

def main():
    # 1. Ustal ścieżkę do katalogu "out"
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "out")

    # 2. Pobierz wszystkie pliki *.txt z katalogu "out", posortowane
    out_files = sorted(
        f for f in os.listdir(out_dir) 
        if f.endswith(".txt")
    )
    count = len(out_files)

    # 3. Sprawdź, czy liczba plików jest podzielna przez 5
    if count != 0 and (count % 5 == 0):
        print(f"[INFO] Znalazłem {count} plików w folderze out. Generuję raport...")

        # 4. Budujemy zawartość raportu HTML
        html = []
        html.append("<html>")
        html.append("<head>")
        html.append("    <meta charset='utf-8' />")
        html.append("    <title>Raport plików out</title>")
        html.append("</head>")
        html.append("<body>")
        html.append(f"    <h2>Raport wygenerowany: {datetime.datetime.now()}</h2>")
        html.append(f"    <p>Liczba plików out: {count}</p>")
        html.append("    <ul>")

        # 5. Dla każdego pliku z out/ odczytujemy zawartość i dodajemy do listy
        for fname in out_files:
            file_path = os.path.join(out_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Wstawiamy wpis: "nazwa_pliku - zawartość_pliku"
            # Możesz użyć np. <pre> dla ładniejszego formatowania zawartości,
            # ale tu wstawiamy prosto w <li>.
            html.append(f"        <li><strong>{fname}</strong> - {content}</li>")

        html.append("    </ul>")
        html.append("</body>")
        html.append("</html>")

        # 6. Zapisujemy raport do pliku raport.html (nadpisując go)
        report_path = os.path.join(base_dir, "raport.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        print(f"[INFO] Raport zapisany w: {report_path}")
    else:
        print(f"[INFO] Plików w 'out': {count}. Nie tworzę raportu (nie jest to wielokrotność 5).")

if __name__ == "__main__":
    main()
