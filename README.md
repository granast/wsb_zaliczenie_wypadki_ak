# Analiza wpływu miejsca zamieszkania kierowcy na wypadki drogowe na terenach wiejskich

## Opis projektu

Projekt bada zależność między miejscem zamieszkania kierowcy (wiejskie vs. miejskie) a prawdopodobieństwem uczestnictwa w wypadku drogowym na obszarach wiejskich. Dodatkowo wykorzystano modele uczenia maszynowego do identyfikacji kluczowych czynników pozwalających przewidywać lokalizację takich wypadków.

Projekt został w pełni udostępniony online na platformie **Streamlit**, gdzie użytkownik może przeprowadzić interaktywną analizę danych oraz zapoznać się z wynikami modeli uczenia maszynowego. Dane używane w aplikacji to statycznie wygenerowane zestawy danych przygotowane przez projekt stworzony w Pythonie.

**Kod źródłowy projektu** można przejrzeć:

- na GitHubie (repozytorium projektu),
- bezpośrednio w aplikacji Streamlit, w sekcji **IX. Podgląd kodu Python** (opcja w nawigacji aplikacji).

## Zawartość repozytorium

- `1_Analiza_wypadki_M_W.ipynb` — Notebook Jupyter zawierający pełny proces przygotowania danych i budowy modeli.
- `Streamlit_app_Wypadki_M_W.py` — Skrypt aplikacji Streamlit do wizualizacji danych i wyników.
- `requirements.txt` — Plik z listą wszystkich wymaganych bibliotek Pythona.
- `README.md` — Ten plik, zawierający opis projektu.

## Plan pracy (Nawigacja w aplikacji)

W aplikacji można przechodzić pomiędzy sekcjami:

**I. Wstęp**\
Opis celu i zakresu projektu.

**II. Dane i metodyka**\
Szczegóły dotyczące źródeł danych i metodologii analizy.

**III. Analiza związku miejsca zamieszkania kierowców**\
Badanie zależności między miejscem zamieszkania a uczestnictwem w wypadkach.

**IV. Machine Learning — opis i wyniki modeli**\
Budowa i ocena modeli uczenia maszynowego.

**V. Ważność cech (XGBoost)**\
Identyfikacja najistotniejszych czynników wpływających na miejsce wypadku.

**VI. Analiza Chi-Kwadrat ważności cech**\
Statystyczna ocena istotności cech.

**VII. Szczegółowa ocena modeli XGBoost i RandomForest**\
Dokładne wyniki i porównanie skuteczności wybranych modeli.

**VIII. Podsumowanie i wnioski końcowe**\
Główne konkluzje wynikające z analizy.

**IX. Podgląd kodu Python**\
Możliwość przeglądania pełnego kodu źródłowego projektu.

## Jak uruchomić aplikację Streamlit

**1. Online (zalecane):**

Otwórz aplikację w przeglądarce internetowej:\
👉 [https://wsb-praca-dyplom-ak.streamlit.app](https://wsb-praca-dyplom-ak.streamlit.app)

**2. Lokalnie w środowisku Python:**

Upewnij się, że masz zainstalowane wszystkie zależności z pliku `requirements.txt`, a następnie uruchom:

```bash
streamlit run Streamlit_app_Wypadki_M_W.py
```

## Technologie użyte

- Python
- Streamlit
- Pandas (2.2.3)
- NumPy (1.26.4)
- Matplotlib (3.9.2)
- Plotly (5.24.1)
- SciPy (1.13.1)
- Scikit-learn (1.5.1)
- Imbalanced-learn (0.12.3)
- XGBoost (2.1.4)
- Tabulate (0.9.0)
