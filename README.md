## Opis projektu:
Ten projekt ma na celu analizę danych dotyczących wypadków drogowych na terenach wiejskich, aby zbadać, czy miejsce zamieszkania kierowcy (wiejskie czy miejskie) ma wpływ na prawdopodobieństwo uczestniczenia w takim wypadku. Dodatkowo, projekt wykorzystuje modele uczenia maszynowego do identyfikacji kluczowych czynników, które mogą pomóc w przewidywaniu lokalizacji wypadków drogowych na obszarach wiejskich.

Projekt został w pełni udostępniony online na platformie Streamlit, gdzie użytkownik może przeprowadzić interaktywną analizę danych oraz zapoznać się z wynikami modeli uczenia maszynowego. Dane używane w aplikacji to statycznie wygenerowane zestawy danych przygotowane przez projekt stworzony w Pythonie.

**Kod źródłowy projektu** można przejrzeć:
- na niniejszym GitHubie (`github.com/granast/wsb_zaliczenie_wypadki_ak`),
- bezpośrednio w aplikacji Streamlit, w sekcji **`IX. Podgląd kodu Python`** (opcja w nawigacji aplikacji).

## Zawartość repozytorium:
Repozytorium zawiera następujące pliki i katalogi:

-   `1_Analiza_wypadki_M_W.ipynb`: Główny skrypt pracy dyplomowej, Notebook Jupyter zawierający pełny proces przygotowania danych i budowy modeli.
-   `Streamlit_app_Wypadki_M_W.py`: Skrypt aplikacji Streamlit do wizualizacji danych i wyników.
-   `requirements.txt`: Plik z listą wszystkich użytych bibliotek Pythona, wymaganych do uruchomienia aplikacji Streamlit.
-   `README.md`: Ten plik, zawierający opis projektu.

## Jak uruchomić aplikację Streamlit:
1.  **Uruchom aplikację Streamlit w przeglądrace internetowej:**
    ```bash
    https://wsb-praca-dyplom-ak.streamlit.app
    ```
    
2.  **Uruchom aplikację Streamlit lokalnie za pomocą terminala** (wymagane środowisko Python oraz dostęp do plików projektu).:
    ```bash
    streamlit run Streamlit_app_Wypadki_M_W.py
    ```
## Plan pracy (spis treści)
W aplikacji można przechodzić pomiędzy sekcjami:
- **I. Wstęp** - Opis celu i zakresu projektu.
- **II. Dane i metodyka** - Szczegóły dotyczące źródeł danych i metodologii analizy.
- **III. Analiza związku miejsca zamieszkania kierowców** - Badanie zależności między miejscem zamieszkania a uczestnictwem w wypadkach.
- **IV. Machine Learning — opis i wyniki modeli** - Budowa i ocena modeli uczenia maszynowego.
- **V. Ważność cech (XGBoost)** - Identyfikacja najistotniejszych czynników wpływających na miejsce wypadku.
- **VI. Analiza Chi-Kwadrat ważności cech** - Statystyczna ocena istotności cech.
- **VII. Szczegółowa ocena modeli XGBoost i RandomForest** - Dokładne wyniki i porównanie skuteczności wybranych modeli.
- **VIII. Podsumowanie i wnioski końcowe** - Główne konkluzje wynikające z analizy.
- **IX. Podgląd kodu Python** - Możliwość przeglądania pełnego kodu źródłowego projektu.

## Technologie użyte:
-   Python
-   Streamlit
-   Biblioteki do analizy danych: Pandas (2.2.3), NumPy (1.26.4)
-   Biblioteki do wizualizacji danych: Matplotlib (3.9.2), Plotly (5.24.1)
-   Biblioteki do obliczeń naukowych: SciPy (1.13.1)
-   Biblioteki do uczenia maszynowego: Scikit-learn (1.5.1), Imbalanced-learn (0.12.3), XGBoost (2.1.4)
-   Biblioteka do formatowania tabel: Tabulate (0.9.0)

---
