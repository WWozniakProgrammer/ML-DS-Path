@echo off
setlocal enabledelayedexpansion
chcp 65001
REM -----------------------------------------
REM KONFIGURACJA - dostosuj ścieżki do swoich potrzeb:
set "INPUT_DIR=D:\Scripts_And_Projects\Google_sheets\ML-DS-Path\in"
set "OUTPUT_DIR=D:\Scripts_And_Projects\Google_sheets\ML-DS-Path\out"
set "SOLVER_EXE=D:\Scripts_And_Projects\Google_sheets\ML-DS-Path\js.exe"

REM -----------------------------------------
REM Jednorazowe przetworzenie wszystkich plików in*.txt w katalogu in\

for %%F in ("%INPUT_DIR%\in*.txt") do (
    REM %%~nF to nazwa pliku bez rozszerzenia, np. "in1"
    set "filename=%%~nF"
    REM obetnij 2 pierwsze znaki (czyli "in") i zapisz do zmiennej suffix
    set "suffix=!filename:~2!"
    
    echo [INFO] Przetwarzam: %%~nxF 
    echo [INFO] Będę zapisywał do: out!suffix!.txt

    REM wywołaj skrypt Python, przekierowując stdin i stdout
    "%SOLVER_EXE%" < "%%F" > "%OUTPUT_DIR%\out!suffix!.txt"
    
    REM (opcjonalnie) usuwamy albo przenosimy przetworzony plik
    REM move /Y "%%F" "%INPUT_DIR%\archiwum\%%~nxF"
    REM del "%%F"
)
REM -----------------------------------------
REM Po przetworzeniu plików in*.txt do out*.txt:

REM 1. Sprawdź, ile mamy teraz plików w out (i ewentualnie wygeneruj raport)
python D:\Scripts_And_Projects\Google_sheets\ML-DS-Path\reporter.py

REM 2. Koniec
echo [INFO] Koniec przetwarzania.
endlocal
pause
exit /b