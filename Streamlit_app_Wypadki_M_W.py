# app_static.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Analiza Wypadków Drogowych UK (Statyczna)")

# --- Pasek boczny nawigacji ---
st.sidebar.title("Nawigacja")
section = st.sidebar.radio(
    "Wybierz sekcję analizy:",
    (
        "I. Wstęp",
        "II. Dane i metodyka",
        "III. Analiza związku miejsca zamieszkania kierowców",
        "IV. Machine Learning - opis i wyniki modeli",
        "V. Ważność cech (XGBoost)",
        "VI. Analiza Chi-Kwadrat ważności cech",
        "VII. Szczegółowa ocena modeli XGBoost i RandomForest",
        "VIII. Podsumowanie i wnioski końcowe"
    )
)

# --- Wyświetlanie wybranej sekcji ---

if section == "I. Wstęp":
    st.title("I. Wstęp")

    st.header("1. Temat")
    st.markdown("""
    **"Analiza związku między miejscem zamieszkania kierowcy, a prawdopodobieństwem udziału w wypadku drogowym."**
    """)

    st.subheader("1.1 Cel pracy:")
    st.markdown("""
    - Zbadanie, czy istnieje związek między miejscem zamieszkania kierowcy (wiejskim lub miejskim), a prawdopodobieństwem jego udziału w wypadku drogowym oraz identyfikacja kluczowych czynników wpływających na przewidywanie lokalizacji wypadku na terenie wiejskim, z wykorzystaniem modeli uczenia maszynowego.
    """)

    st.subheader("1.2 Pytania badawcze:")
    st.markdown("""
    - Czy miejsce zamieszkania kierowcy (miejskie vs. niemiejskie) wpływa na prawdopodobieństwo udziału w wypadku drogowym?
    - Jakie z wybranych cech kontekstowych (np. typ drogi, warunki oświetleniowe, kontrola skrzyżowań) mają największy wpływ na prawdopodobieństwo wystąpienia wypadku na terenie wiejskim?
    - Czy modele uczenia maszynowego (XGBoost, RandomForest) mogą skutecznie przewidzieć lokalizację wypadku na podstawie miejsca zamieszkania kierowcy i cech kontekstowych?
    """)

    st.subheader("1.3 Hipoteza badawcza:")
    st.markdown("""
    - Kierowcy z obszarów miejskich są bardziej narażeni na udział w wypadkach drogowych na terenach wiejskich niż kierowcy z obszarów wiejskich.
    - Specyficzne cechy, takie jak drogi jednopasmowe, brak oświetlenia ulicznego oraz niekontrolowane skrzyżowania, znacząco zwiększają ryzyko wypadku na terenie wiejskim.
    - Modele uczenia maszynowego (XGBoost, RandomForest) nie osiągają wysokiej skuteczności w przewidywaniu lokalizacji wypadku (wiejskiej vs. miejskiej) na podstawie miejsca zamieszkania kierowcy i cech kontekstowych.
    """)

elif section == "II. Dane i metodyka":
    st.title("II. Dane i metodyka")

    st.header("1. Źródła danych")
    st.markdown("""
    - Dane pochodzą z oficjalnych brytyjskich baz danych (Department for Transport - data.gov.uk) dotyczących wypadków drogowych z lat 2021-2023 na terenie UK.
    - Tabele (`casualties`, `vehicles`, `accidents`) zawierające dane m.in. o ofiarach (wiek, miejsce zamieszkania), informacje o pojazdach i kierowcach (np. obszar zamieszkania, odległość od miejsca wypadku) oraz kontekst wypadków (warunki pogodowe, typ drogi) zostały połączone w tabelę `data` po kluczu `accident_index`.
    - Statystyki dotyczą wyłącznie wypadków z obrażeniami ciała na drogach publicznych, które są zgłaszane policji, a następnie rejestrowane przy użyciu formularza zgłaszania kolizji `STATS19`.
    - **Przewodnik** po statystykach dotyczących wypadków drogowych: [link](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
    - **Zestawy danych** do pobrania: [link](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
    - Poniższa analiza została udostępnione na platformie `streamlit.io`, a jej dane zostały wygenerowane za pomocą wartości statycznych wykonanych przez program Python oraz jego wersje bibliotek:
        - **NumPy**: 1.26.4  
        - **Pandas**: 2.2.3  
        - **Matplotlib**: 3.9.2  
        - **Plotly**: 5.24.1  
        - **Scikit-learn**: 1.5.1  
        - **Tabulate**: 0.9.0  
        - **Imbalanced-learn (SMOTE)**: 0.12.3  
        - **XGBoost**: 2.1.4  
        - **SciPy**: 1.13.1
    """)

    st.header("2. Opis użytych zmiennych")
    st.markdown("""
    W analizie wykorzystano następujące zmienne, które opisują okoliczności wypadków drogowych, charakterystyki kierowców, pojazdów oraz poszkodowanych:
    - `road_type` – Rodzaj drogi, na której doszło do wypadku. Kategorie obejmują: rondo (1), ulica jednokierunkowa (2), droga dwujezdniowa (3), droga jednojezdniowa (6), droga dojazdowa (7), nieznana (9), ulica jednokierunkowa/droga dojazdowa (12) lub brak danych (-1).
    - `light_conditions` – Warunki oświetlenia w czasie wypadku. Kategorie: światło dzienne (1), ciemność z działającym oświetleniem (4), ciemność z niedziałającym oświetleniem (5), ciemność bez oświetlenia (6), ciemność z nieznanym stanem oświetlenia (7) lub brak danych (-1).
    - `junction_detail` – Szczegóły dotyczące skrzyżowania w miejscu wypadku. Obejmuje: brak skrzyżowania w promieniu 20 metrów (0), rondo (1), mini-rondo (2), skrzyżowanie typu T lub rozwidlenie (3), droga dojazdowa (5), skrzyżowanie czteroramienne (6), skrzyżowanie z więcej niż 4 ramionami (7), prywatny wjazd (8), inne skrzyżowanie (9), nieznane (99) lub brak danych (-1).
    - `junction_control` – Rodzaj kontroli ruchu na skrzyżowaniu. Kategorie: brak skrzyżowania w promieniu 20 metrów (0), osoba upoważniona (1), sygnalizacja świetlna (2), znak stopu (3), ustąp pierwszeństwa lub brak kontroli (4), nieznane (9) lub brak danych (-1).
    - `driver_home_area_type` – Typ obszaru zamieszkania kierowcy. Obejmuje: obszar miejski (1), małe miasto (2), obszar wiejski (3) lub brak danych (-1).
    - `accident_year` – Rok, w którym doszło do wypadku.
    - `age_of_casualty` – Wiek osoby poszkodowanej w wypadku. Wartość -1 oznacza brak danych.
    - `driver_distance_banding` – Odległość miejsca wypadku od miejsca zamieszkania kierowcy. Kategorie: do 5 km (1), 5,001–10 km (2), 10,001–20 km (3), 20,001–100 km (4), powyżej 100 km (5) lub brak danych (-1).
    - `weather_conditions` – Warunki pogodowe w czasie wypadku. Kategorie: dobra pogoda bez silnego wiatru (1), deszcz bez silnego wiatru (2), śnieg bez silnego wiatru (3), dobra pogoda z silnym wiatrem (4), deszcz z silnym wiatrem (5), śnieg z silnym wiatrem (6), mgła (7), inne (8), nieznane (9) lub brak danych (-1).
    - `urban_or_rural_area` – Typ obszaru, w którym doszło do wypadku: miejski (1), wiejski (2), nieprzypisany (3) lub brak danych (-1).
    - `casualty_type` – Typ poszkodowanego w wypadku, np.: pieszy (0), rowerzysta (1), motocyklista (2–5, 23, 97, 103–106), pasażer
    taksówki (8), pasażer samochodu (9), pasażer busa (10–11), jeździec konny (16), inne typy pojazdów (17–21, 90, 98–99, 108–110, 113) lub brak danych (-1).
    - `speed_limit` – Ograniczenie prędkości na drodze w miejscu wypadku. Wartości w milach na godzinę, np. 30, 60; 99 oznacza nieznane (zgłoszone przez uczestnika), a -1 brak danych.
    - `driver_imd_decile` – Poziom deprywacji społeczno-ekonomicznej kierowcy według indeksu IMD (ang. Index of Multiple Deprivation). Skala od 1 (najbardziej deprywowany 10%) do 10 (najmniej deprywowany 10%) lub brak danych (-1).
    - `age_of_vehicle` – Wiek pojazdu w latach w momencie wypadku. Wartość -1 oznacza brak danych.
    - `age_of_driver` – Wiek kierowcy w momencie wypadku. Wartość -1 oznacza brak danych.
    - `number_of_casualties` – Liczba osób poszkodowanych w wyniku wypadku.
    - `skidding_and_overturning` – Informacja o poślizgu lub przewróceniu pojazdu. Kategorie: brak (0), poślizg (1), poślizg i przewrócenie (2), wyłamanie (3), wyłamanie i przewrócenie (4), przewrócenie (5), nieznane (9) lub brak danych (-1).
    """)

    st.header("3. Opis przygotowania danych")
    st.markdown("""
    W oryginalnej analizie przeprowadzono następujące kroki przygotowania danych (nie są one wykonywane w tej statycznej wersji):
    -  **Oczyszczono dane:** Zastąpiono `-1` i `99` na `NaN`, a następnie usunięto wiersze z brakami w tych kolumnach.
    -  **Przekształcono czas:** Z kolumny `time` utworzono `hour_of_day`.
    -  **Przygotowanie zmiennej docelowej:** dla `driver_home_area_type` zsumowano wartości 2 i 3 (small town oraz unrual) w jedną etykietę nr 2 dla przejrzystości danych
    -  **Tworzenie zmiennych binarnych:**
         - is_urban_driver: Kierowca pochodzi z obszaru miejskiego (`driver_home_area_type` = 1).
         - is_rural_accident: Wypadek miał miejsce na terenie wiejskim (`urban_or_rural_area` = 2).
    -  **Znormalizowano prędkość:** `speed_limit` przekształcono w `speed_limit_normalized`.
    -  **Zbindowano wiek:** `age_of_casualty` i `age_of_driver` przekształcono w `age_of_casualty_binned` i `age_of_driver_binned`.
         - Zarówno age_of_casualty jak i age_of_driver podzielono na 5 przedziałów:
         - ≤17 lat, 18-25 lat, 26-40 lat, 41-60 lat, >60 lat.
    -  **Stworzono nowe cechy:**
        * `urban_driver_speed` jako iloczyn `is_urban_driver` i `speed_limit_normalized`.
        * `is_rush_hour` na podstawie `hour_of_day`.
        * `distance_speed_interaction` jako iloczyn `driver_distance_banding` i `urban_driver_speed`.
    -  **Wybrano cechy:** Ustalono listę `selected_features`, która teraz zawiera również `casualty_type`.
    -  **Zakodowano kategorie:** Zmienne kategorialne z `selected_features` (w tym nowa kolumna `casualty_type` oraz `road_type`, `light_conditions`, `junction_detail`, `junction_control`, `age_of_casualty_binned`, `driver_distance_banding`, `weather_conditions`, `age_of_driver_binned`, `skidding_and_overturning`) zakodowano zero-jedynkowo.
    -  **Stworzono dodatkowe cechy po kodowaniu:**
        * `important_driver_distance` na podstawie `driver_distance_banding_4.0` i `driver_distance_banding_3.0`.
        * `urban_driver_long_distance` jako iloczyn `is_urban_driver` i `important_driver_distance`.
        * `urban_driver_no_junction_control` jako iloczyn `is_urban_driver` i `junction_control_4.0` (jeśli istnieje).
    -  **Podział danych: Dane podzielono na zbiory:**
        * Treningowy + walidacyjny (80%) i testowy (20%) z zachowaniem stratyfikacji..
        * Następnie zbiór treningowy + walidacyjny podzielono na treningowy (60% całości) i walidacyjny (20% całości), również ze stratyfikacją.
    -  **Balansowanie danych:**
        * Zastosowano SMOTE na zbiorze treningowym, aby zrównoważyć klasy zmiennej docelowej `is_rural_accident`.
    -  **Rozmiary zbiorów danych po przetworzeniu:**
        * Zbiór treningowy (po SMOTE): 228388 rekordów / Zbiór walidacyjny: 54611 rekordów / Zbiór testowy: 54611 rekordów.
                
    Celem było przygotowanie danych (X) i zmiennej docelowej (y, czyli `is_rural_accident`) do modelowania poprzez oczyszczenie, transformację i stworzenie nowych cech, uwzględniając teraz również typ uczestnika wypadku (`casualty_type`).
""")

elif section == "III. Analiza związku miejsca zamieszkania kierowców":
    st.title("III. CZ.1 Analiza i wyniki związku między miejscem zamieszkania kierowcy (wiejskim lub miejskim), a prawdopodobieństwem jego udziału w wypadku drogowym.")

    # --- Dane statyczne ---
    total_accidents_static = 273053

    # Tabela 1: Proporcje kierowców
    driver_origin_data = {
        'Pochodzenie': ['kier. Miejski', 'kier. Wiejski', 'Suma'],
        'Liczba': [222719, 50334, 273053],
        'Procent': [81.6, 18.4, 100.0]
    }
    driver_origin_display = pd.DataFrame(driver_origin_data)

    # Tabela 2: Rozkład wg lat
    driver_stats_data = {
        'Rok': [2021, 2022, 2023],
        'kier. Wiejski': [15908, 17419, 17007],
        'kier. Wiejskich (%) ': [17.8, 18.7, 18.9],
        'kier. Miejski': [73686, 75877, 73156],
        'kier. Miejskich (%)': [82.2, 81.3, 81.1],
        'Suma': [89594, 93296, 90163]
    }
    driver_stats_display = pd.DataFrame(driver_stats_data)

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela 1: Proporcje kierowców według miejsca zamieszkania")
    st.dataframe(driver_origin_display.style.format({'Liczba': '{:,.0f}', 'Procent': '{:.1f}%'}))
    st.markdown("""
    **Komentarz:** Kierowcy z obszarów miejskich dominują w ogólnej liczbie wypadków (81.6%), co może odzwierciedlać większą populację miejską lub częstsze korzystanie z dróg.
    """)

    st.subheader("Tabela 2: Rozkład kierowców według miejsca zamieszkania w latach 2021-2023")
    st.dataframe(driver_stats_display.style.format({
        'kier. Wiejski': '{:,.0f}', 'kier. Wiejskich (%) ': '{:.1f}%',
        'kier. Miejski': '{:,.0f}', 'kier. Miejskich (%)': '{:.1f}%',
        'Suma': '{:,.0f}'
    }))
    st.markdown("""
    **Komentarz:** Proporcje pozostają stosunkowo stałe w latach 2021-2023, z lekkim wzrostem udziału kierowców wiejskich w 2023 roku (18.9%), co sugeruje stabilność trendów w czasie.
    """)

    st.subheader("Wykresy tabel 1 i 2")

    # --- Odtworzenie wykresów Matplotlib na podstawie danych statycznych ---
    fig_mpl = plt.figure(figsize=(12, 10))
    gs = fig_mpl.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Wykres 1: Całkowita liczba wypadków
    ax1 = fig_mpl.add_subplot(gs[0, 0])
    bars1 = ax1.bar(['Wszystkie wypadki'], [total_accidents_static], color='#93c47d')
    ax1.set_title('Całkowita liczba analizowanych wypadków')
    ax1.set_ylabel('Liczba')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{int(height):,} (100%)', ha='center', va='center', fontsize=10, color='black')

    # Wykres 2: Proporcje kierowców
    ax2 = fig_mpl.add_subplot(gs[0, 1])
    driver_origin_plot = driver_origin_display[driver_origin_display['Pochodzenie'] != 'Suma'].set_index('Pochodzenie')
    bottom_val = 0
    colors = {'kier. Wiejski': '#ff7f0e', 'kier. Miejski': '#1f77b4'}
    order = ['kier. Wiejski', 'kier. Miejski']
    for origin_type in order:
        if origin_type in driver_origin_plot.index:
            value = driver_origin_plot.loc[origin_type, 'Liczba']
            percentage = driver_origin_plot.loc[origin_type, 'Procent']
            bar = ax2.bar(['Kierowcy'], [value], bottom=[bottom_val], color=colors[origin_type], label=origin_type)
            text_y = bottom_val + value / 2
            ax2.text(0, text_y, f"{int(value):,}\n({percentage:.1f}%)", ha='center', va='center', fontsize=10, color='white')
            bottom_val += value

    ax2.set_title('Proporcje kierowców wg miejsca zamieszkania')
    ax2.set_ylabel('Liczba kierowców')
    ax2.set_xticks([])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, total_accidents_static * 1.1)

    # Wykres 3: Rozkład kierowców według lat
    ax3 = fig_mpl.add_subplot(gs[1, :])
    bar_width = 0.35
    x = np.arange(len(driver_stats_display['Rok']))
    rects1 = ax3.bar(x - bar_width/2, driver_stats_display['kier. Wiejski'], bar_width, label='kier. Wiejski', color='#ff7f0e')
    rects2 = ax3.bar(x + bar_width/2, driver_stats_display['kier. Miejski'], bar_width, label='kier. Miejski', color='#1f77b4')

    ax3.set_title('Rozkład kierowców wg miejsca zamieszkania w latach')
    ax3.set_xlabel('Rok')
    ax3.set_ylabel('Liczba kierowców')
    ax3.set_xticks(x)
    ax3.set_xticklabels(driver_stats_display['Rok'])
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height):,}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    autolabel(rects1, ax3)
    autolabel(rects2, ax3)

    fig_mpl.suptitle('Analiza kierowców w wypadkach drogowych (2021-2023)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    st.pyplot(fig_mpl)

    # --- Dane statyczne ---
    contingency_data = {
        'Wypadek obszar Miejski': [15893, 174431],
        'Wypadek obszar Wiejski': [34441, 48288]
    }
    contingency_table = pd.DataFrame(contingency_data, index=['kier. Wiejski', 'kier. Miejski'])

    location_stats_data = {
        'Wypadki obszar Miejski (%)': [31.6, 78.3],
        'Wypadki obszar Wiejski (%)': [68.4, 21.7]
    }
    location_stats = pd.DataFrame(location_stats_data, index=['kier. Wiejski', 'kier. Miejski'])

    chi2_stat = 42475.60
    p_value_chi2 = 0.0
    dof_chi2 = 1
    phi_stat = 0.394
    strength = "Umiarkowany (φ = 0.3–0.5)"
    conclusion = f"Odrzucamy hipotezę zerową (H₀). Istnieje statystycznie istotny związek (p < 0.0001)."
    alpha = 0.05

    expected_data = {
        'Wypadek obszar Miejski': [35083.9, 155240.1],
        'Wypadek obszar Wiejski': [15250.1, 67478.9]
    }
    expected_df = pd.DataFrame(expected_data, index=['kier. Wiejski', 'kier. Miejski'])

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela 3: Procent uczestników wypadków wg miejsca zamieszkania i lokalizacji")
    st.dataframe(location_stats.style.format("{:.1f}%"))

    # --- Odtworzenie Wykresu Plotly ---
    st.subheader("Wykres tabeli nr 3")
    location_stats_plot = location_stats.reset_index().rename(columns={'index': 'Pochodzenie Kierowcy'})
    location_stats_melted = location_stats_plot.melt(
        id_vars='Pochodzenie Kierowcy',
        var_name='Typ Obszaru Wypadku',
        value_name='Procent Wypadków'
    )
    location_stats_melted['Typ Obszaru Wypadku'] = location_stats_melted['Typ Obszaru Wypadku'].str.replace(' (%)', '')

    fig_plotly = px.bar(location_stats_melted,
                            x='Pochodzenie Kierowcy',
                            y='Procent Wypadków',
                            color='Typ Obszaru Wypadku',
                            title='Procentowy udział wypadków Miejskich i Wiejskich wg Miejsca Zamieszkania Kierowcy',
                            labels={'Procent Wypadków': 'Procent Wypadków (%)', 'Typ Obszaru Wypadku': 'Lokalizacja Wypadku'},
                            color_discrete_map={'Wypadki obszar Miejski': '#1f77b4', 'Wypadki obszar Wiejski': '#ff7f0e'},
                            barmode='group',
                            text='Procent Wypadków'
                            )
    fig_plotly.update_layout(yaxis_ticksuffix='%', yaxis_title='Procent Wypadków (%)', xaxis_title='Pochodzenie Kierowcy',legend_title_text='Lokalizacja Wypadku')
    fig_plotly.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_plotly, use_container_width=True)

    st.subheader("Test chi-kwadrat dla niezależności")
    st.markdown(f"""
    Aby ocenić, czy istnieje statystycznie istotny związek między miejscem zamieszkania kierowcy a lokalizacją wypadku, przeprowadzono **test chi-kwadrat**.  
    - **Hipoteza zerowa (H₀):** Nie ma związku między pochodzeniem kierowcy a lokalizacją wypadku.  
    - **Hipoteza alternatywna (H₁):** Istnieje związek między tymi zmiennymi.  
    """)
    st.markdown(f"""
    **Tabela 4 Kontyngencji (obserwowane częstości):** """)
    st.dataframe(contingency_table.style.format("{:,.0f}"))

    st.markdown(f"""
    **Tabela 5 oczekiwana (dla H₀):** """)
    st.dataframe(expected_df.style.format("{:,.1f}"))

    st.markdown(f"""
    **Wyniki testu statystycznego:** """)
    st.markdown(f"""
    - **Statystyka chi-kwadrat (χ²):** {chi2_stat:.2f}
    - **Wartość p (p-value):** {p_value_chi2:.4e} (bardzo bliska 0)
    - **Stopnie swobody (dof):** {dof_chi2}
    - **Współczynnik Phi (φ) = {phi_stat:.3f} → związek **umiarkowany** 

    **Wniosek (poziom istotności α = {alpha}):** 
    {conclusion}
    """)

    # --- Kluczowe obserwacje z tej sekcji ---
    st.subheader("Kluczowe obserwacje")
    st.markdown("""
    **Kluczowe obserwacje:**
    - **Kierowcy z obszarów wiejskich**: Znacznie częściej uczestniczą w wypadkach na terenach wiejskich (68.4%) niż miejskich (31.6%).
    - **Kierowcy z obszarów miejskich**: Dominują w wypadkach na terenach miejskich (78.3%), a rzadziej uczestniczą w wypadkach na terenach wiejskich (21.7%).

    **Wyniki testu chi-kwadrat:**
    - Test chi-kwadrat (χ² = 42,475.6, p < 0.001) wykazał wartość **p < 0.001**, która wskazuje na **odrzucenie hipotezy zerowej (H₀)**, co potwierdza statystycznie istotny związek między miejscem zamieszkania kierowcy a lokalizacją wypadku. Oznacza to, że korelacja (zależność) między miejscem zamieszkania, a lokalizacją wypadku jest nieprzypadkowa i może być generalizowana na szerszą populację.
    - Siła tej korelacji (związku/zależności), mierzona współczynnikiem Phi, osiągneła wartość (φ ≈ 0.394), co sugeruje **umiarkowaną siłę związku** między zmiennymi. Oznacza to, że zmienna miejsca zamieszkania jest ważna, jednak inne czynniki (zmienne) (np. warunki drogowe, doświadczenie kierowcy) mogą również wpływać na wyniki.

    **Wiarygodność wyników:**  
    - Duża próba (N = 273,053) zwiększa wiarygodność wyników, choć umiarkowana siła związku (φ = 0.394) sugeruje potrzebę uwzględnienia dodatkowych czynników w dalszych analizach.
    """)

    # --- Wnioski końcowe ---
    st.subheader("Wnioski końcowe")
    st.markdown("""
    1.1. **Cel pracy:** Zbadanie, czy istnieje związek między miejscem zamieszkania kierowcy (miejskim lub wiejskim), a prawdopodobieństwem jego udziału w wypadku drogowym.    
    1.2. **Pytanie badawcze:** Czy miejsce zamieszkania kierowcy (miejskie vs. wiejskie) wpływa na prawdopodobieństwo udziału w wypadku drogowym?     
    1.3. **Hipoteza badawcza:** Kierowcy miejscy są bardziej narażeni na wypadki na terenach wiejskich niż kierowcy wiejscy.
                
    **Na podstawie danych z brytyjskich baz wypadków drogowych z lat 2021–2023 stwierdzono:**
                
    **1.1. i 1.2. Odpowiedź:**  
    - Analiza potwierdziła istotny związek. Kierowcy mają tendencję do uczestniczenia w wypadkach w środowisku zgodnym z miejscem zamieszkania — kierowcy miejscy częściej ulegają wypadkom na obszarach miejskich, a kierowcy z obszarów wiejskich na terenach wiejskich. Szczególnie wyraźnie widać to w przypadku kierowców wiejskich, którzy ponad trzykrotnie częściej uczestniczą w wypadkach na obszarach wiejskich (68.4%) niż kierowcy miejscy (21.7%).

    **1.3. Odpowiedź:**  
    - Wyniki nie potwierdzają hipotezy, że kierowcy z obszarów miejskich są bardziej narażeni na wypadki na terenach wiejskich. Przeciwnie, kierowcy z obszarów wiejskich dominują w tej kategorii w swoich grupach.
    """)

elif section == "IV. Machine Learning - opis i wyniki modeli":
    st.title("IV. Modelowanie Uczenia Maszynowego")
    st.header("II. CZ. 2 - Identyfikacja kluczowych czynników wpływających na przewidywanie lokalizacji wypadku (na terenie wiejskim), z wykorzystaniem modeli uczenia maszynowego.")
    st.markdown("""
    - Drugi cel pracy koncentruje się na identyfikacji kluczowych czynników wpływających na przewidywanie, czy wypadek drogowy miał miejsce na terenie wiejskim (`is_rural_accident` = 1), z wykorzystaniem modeli uczenia maszynowego.
    """)
    st.subheader("Wybrane modele w celu zbadania binarnej klasyfikacji lokalizacji wypadku drogowego – czy miał on miejsce na terenie wiejskim, czy nie:")
    st.markdown("- **XGBoost Classifier**")
    st.markdown("- **Random Forest Classifier**")

    st.subheader("Opis Procesu modelowania:")
    st.markdown("""
    1. **Przygotowanie danych** do trenowania modeli:
        - Dane zostały podzielone na trzy zbiory: treningowy (60%), walidacyjny (20%) oraz testowy (20%).
        - Na zbiorze treningowym zastosowano technikę **SMOTE** w celu zrównoważenia klas przed uczeniem modeli.
    2. **Trenowanie modeli XGBoost i Random Forest** na zbalansowanym zbiorze treningowym, z zastosowaniem odpowiednio dobranych hiperparametrów.
    3. **Ocena modeli** na zbiorze walidacyjnym i testowym (pozostawionych w oryginalnej, niezbalansowanej postaci):
        - Wykorzystano metryki klasyfikacji: `classification_report` (dokładność, precyzja, recall, F1-score) oraz **AUC-ROC**.
    4. Na podstawie wyników metryk walidacyjnych i testowych **wybrano model XGBoost** jako skuteczniejszy (wyższe AUC-ROC).
    5. Dla modelu XGBoost przeanalizowano **ważność cech wejściowych (feature importance)**:
        - Utworzono tabelę ważności cech na podstawie `.feature_importances_`.
        - Wybrano 12 najważniejszych cech i zaprezentowano je w formie tekstowej oraz graficznej (poziomy wykres słupkowy).
    6. **Wykonano testy niezależności (chi-kwadrat)**:
        - Przeprowadzono je dla najważniejszych zmiennych kategorialnych względem `is_rural_accident`.
        - Siłę powiązań oceniono na podstawie wartości V Craméra i dokonano interpretacji (słaby, umiarkowany, silny efekt).
    7. **Sprawdzono skuteczność modelu XGBoost**:
        - Wygenerowano krzywą ROC oraz obliczono AUC w celu oceny rozdzielczości modelu.
        - Wygenerowano wykres uczenia (`learning_curve()`) z użyciem scoringu F1 — graficzna analiza overfittingu i underfittingu.
    8. **Przedstawiono wnioski końcowe** z procesu analizy i modelowania.     
                
    *Ta statyczna wersja aplikacji nie trenuje modeli, jedynie prezentuje wcześniej uzyskane wyniki.*
    """)
    st.subheader("Hiperparametry Użyte w Analizie:")
    st.code("""
# XGBoost
params_xgb = {
    'random_state': 42, 'scale_pos_weight': 1, 'max_depth': 9,
    'n_estimators': 269, 'learning_rate': 0.06, 'reg_alpha': 0.1,
    'reg_lambda': 1.9, 'subsample': 0.8, 'colsample_bytree': 0.6,
}

# RandomForest
params_rf = {
    'random_state': 42, 'n_estimators': 229, 'max_depth': 14,
    'min_samples_split': 54, 'min_samples_leaf': 26, 'n_jobs': -1,
    'max_features': 'sqrt', 'criterion': 'entropy', 'bootstrap': False,
}
    """, language='python')

    # --- Przeniesione Wyniki ---
    st.subheader("Wyniki na Zbiorze Walidacyjnym")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {0.9400:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code("""
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.83      0.79      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
        """)
    with col2:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {0.9338:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code("""
                  precision    recall  f1-score   support

           0       0.92      0.88      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
        """)

    st.subheader("Wyniki na Zbiorze Testowym (Ostateczna Ocena)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {0.9400:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code("""
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.84      0.78      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
        """)
    with col4:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {0.9327:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code("""
                  precision    recall  f1-score   support

           0       0.92      0.89      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
        """)

    st.markdown("""
    **Podsumowanie:**
    - **XGBoost**: Lepszy od RandomForest pod względem AUC-ROC (0.9400 vs 0.9327) i F1-score dla klasy wiejskiej (0.81 vs 0.79). Wyższy balans precision-recall.
    - **Wniosek**: XGBoost wybrano do dalszej analizy ze względu na wyższą skuteczność i stabilność.
    """)

elif section == "V. Ważność cech (XGBoost)":
    st.title("V. Najważniejsze czynniki wypływające na przewidywanie lokalizacji wypadku (na obszarze wiejskim) wg Modelu XGBoost")
    st.markdown("Pokazuje, które cechy miały największy wpływ na predykcje modelu XGBoost w niniejszej analizie.")

    # --- Statyczne Dane Ważności Cech (Top 12) ---
    feature_importance_data = {
        'Cecha': [
            'speed_limit_normalized',
            'urban_driver_speed',
            'is_urban_driver',
            'distance_speed_interaction',
            'junction_detail_1.0',
            'road_type_6',
            'junction_control_4.0',
            'light_conditions_6.0',
            'casualty_type_9.0',
            'important_driver_distance',
            'urban_driver_long_distance',
            'skidding_and_overturning_9.0'
        ],
        'Ważność': [
            0.1827,
            0.1711,
            0.0566,
            0.0543,
            0.0291,
            0.0268,
            0.0242,
            0.0231,
            0.0170,
            0.0166,
            0.0148,
            0.0132
        ]
    }
    feature_importance_df = pd.DataFrame(feature_importance_data)

    st.subheader("Tabela 6: Top 12 najważniejszych cech")
    st.dataframe(feature_importance_df.style.format({'Ważność': '{:.4f}'}))

    # --- Wykres Ważności Cech (Top 12) ---
    fig_feature_importance = px.bar(
        feature_importance_df,
        x='Ważność',
        y='Cecha',
        orientation='h',
        title='Wizualizacja tabeli 6: Ważność Cech - Model XGBoost (Top 12)',
        labels={'Ważność': 'Ważność Cechy (udział %)', 'Cecha': 'Nazwa Cechy'},
        color='Ważność',
        color_continuous_scale='viridis'
    )
    fig_feature_importance.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 0.20])
    )
    st.plotly_chart(fig_feature_importance, use_container_width=True)

    # --- Interpretacja 12 Najważności Cech ---
    st.subheader("Identyfikacja kluczowych cech wpływających na przewidywanie lokalizacji wypadku (na obszarze wiejskim)")
    st.markdown("""
    1.  **Ograniczenie prędkości (`speed_limit_normalized`)**: Najważniejsza cecha, co sugeruje, że (wyższe) limity prędkości znacząco wpływają na ryzyko wypadków na obszarach wiejskich.
    2.  **Interakcja prędkości i typu kierowcy (`urban_driver_speed`)**: Zależność między prędkością a pochodzeniem kierowcy (miejski lub wiejski) wydaje się istotna dla modelu.
    3.  **Pochodzenie kierowcy (`is_urban_driver`)**: Informacja, czy kierowca jest z obszaru miejskiego, wpływa na ocenę ryzyka lokalizacji zdarzenia.
    4.  **Interakcja odległości i prędkości (`distance_speed_interaction`)**: Sugeruje, że kierowcy pokonujący większe odległości mogą być bardziej narażeni na ryzyko.
    5.  **Szczegóły skrzyżowania (rondo) (`junction_detail_1.0`)**: Obecność ronda jako typu skrzyżowania może być czynnikiem wpływającym na przewidywania modelu.
    6.  **Typ drogi (droga jednojezdniowa) (`road_type_6`)**: Rodzaj drogi (tutaj jednojezdniowej) może mieć znaczenie przy określaniu lokalizacji wypadków.
    7.  **Kontrola skrzyżowania (brak kontroli) (`junction_control_4.0`)**: Brak kontroli na skrzyżowaniu jest zmienną istotną w kontekście przewidywania.
    8.  **Warunki oświetleniowe (ciemność bez oświetlenia) (`light_conditions_6.0`)**: Słabe warunki oświetleniowe (brak oświetlenia) mogą mieć wpływ na lokalizację zdarzenia.
    9.  **Typ poszkodowanego (pasażer samochodu) (`casualty_type_9.0`)**: Typ poszkodowanej osoby może współwystępować z określonymi lokalizacjami wypadków.
    10. **Odległość od miejsca zamieszkania (`important_driver_distance`)**: Podobnie jak wyżej, większa odległość do pokonania zwiększa ryzyko.
    11. **Interakcja odległości i pochodzenia kierowcy (`urban_driver_long_distance`)**: Podobnie jak wyżej, większa odległość do pokonania zwiększa ryzyko.
    12. **Poślizg i wywrócenie (`skidding_and_overturning_9.0`)**: Informacja o tym, czy doszło do poślizgu i/lub wywrócenia pojazdu, jest istotna dla modelu.
""")

elif section == "VI. Analiza Chi-Kwadrat ważności cech":
    st.title("VI. Analiza Związku Kluczowych Cech z Lokalizacją Wypadku (Test Chi-kwadrat)")

    # --- Opis sekcji ---
    st.markdown("""
    W tej sekcji analizujemy statystyczny związek między kluczowymi cechami zidentyfikowanymi w modelowaniu XGBoost a zmienną `is_rural_accident` (lokalizacją wypadku - teren miejski vs. wiejski) przy użyciu testu chi-kwadrat. Testy te potwierdzają, czy cechy wybrane jako istotne przez model XGBoost (sekcja V) mają statystycznie istotny związek z zmienną docelową (`is_rural_accident`). Siła powiązań mierzona jest współczynnikiem V Craméra.
    """)

    # --- Dane statyczne ---
    summary_data = {
        'Zmienna': [
            '`is_urban_driver` (Pochodzenie kierowcy)',
            '`light_conditions` (Warunki oświetleniowe)',
            '`driver_journey_purpose` (Odległość od miejsca zamieszkania)',
            '`junction_control` (Kontrola skrzyżowania)',
            '`casualty_type` (Typ poszkodowanego)',
            '`skidding_and_overturning` (Poślizg i wywrócenie)',
            '`junction_detail` (Szczegóły skrzyżowania)',
            '`road_type` (Typ drogi)'
        ],
        'Statystyka χ²': [42475.6, 13593.7, 13160.5, 9180.7, 8562.5, 7113.2, 2669.3, 349.1],
        'p-value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.543e-76],
        'V Craméra': [0.394, 0.223, 0.220, 0.183, 0.177, 0.161, 0.099, 0.036],
        'Siła związku': ['Umiarkowany', 'Umiarkowany', 'Umiarkowany', 'Umiarkowany', 'Umiarkowany', 'Umiarkowany', 'Słaby', 'Słaby']
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.index = range(1, len(summary_df) + 1)

    # --- Wyświetlanie tabeli podsumowującej ---
    st.subheader("Tabela 1: Podsumowanie wyników testów chi-kwadrat")
    st.dataframe(summary_df.style.format({
        'Statystyka χ²': '{:.1f}',
        'p-value': '{:.3e}',
        'V Craméra': '{:.3f}'
    }))
    st.markdown("""
    **Wniosek:** Wszystkie zmienne wykazują statystycznie istotny związek z `is_rural_accident` (p < 0.05). Najsilniejszy związek obserwujemy dla `is_urban_driver` (Pochodzenie kierowcy) (V = 0.394), co zgadza się z wysoką ważnością tej cechy w modelu XGBoost (3. miejsce, ważność = 0.0566). Słabsze związki dla `junction_detail` (Szczegóły skrzyżowania) i `road_type` (Typ drogi) potwierdzają ich mniejszy wpływ w modelowaniu.
    """)

    # --- Wykres V Craméra (interaktywny) ---
    st.subheader("Wykres 2: Siła związku V Craméra (interaktywny)")
    fig_plotly = px.bar(summary_df,
                        x='Zmienna',
                        y='V Craméra',
                        color='Siła związku',
                        title='Siła związku (V Craméra) dla kluczowych cech z modelu XGBoost',
                        labels={'V Craméra': 'V Craméra', 'Zmienna': 'Cecha'},
                        color_discrete_map={'Umiarkowany': '#1f77b4', 'Słaby': '#ff7f0e'},
                        text='V Craméra')
    fig_plotly.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_plotly.update_layout(
        yaxis_title='V Craméra',
        xaxis_title='Cecha',
        legend_title_text='Siła związku',
        shapes=[
            # Orange line at 0.1
            dict(
                type="line",
                x0=-0.5,  # Start of x-axis range
                x1=len(summary_df)-0.5,  # End of x-axis range
                y0=0.1,  # Orange threshold
                y1=0.1,  # Orange threshold
                line=dict(color="orange", width=2, dash="dash")
            ),
            # Blue line at 0.3
            dict(
                type="line",
                x0=-0.5,  # Start of x-axis range
                x1=len(summary_df)-0.5,  # End of x-axis range
                y0=0.3,  # Blue threshold
                y1=0.3,  # Blue threshold
                line=dict(color="blue", width=2, dash="dash")
            ),
            # Red line at 0.5
            dict(
                type="line",
                x0=-0.5,  # Start of x-axis range
                x1=len(summary_df)-0.5,  # End of x-axis range
                y0=0.5,  # Red threshold
                y1=0.5,  # Red threshold
                line=dict(color="red", width=2, dash="dash")
            )
        ]
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- Szczegółowe tabele kontyngencji ---
    st.subheader("Szczegółowe wyniki testów chi-kwadrat")
    st.markdown("""
    Poniżej przedstawiono tabele kontyngencji, wartości oczekiwane (dla wybranych cech) oraz wyniki testów chi-kwadrat dla każdej zmiennej, w odniesieniu do lokalizacji wypadku `is_rural_accident` (teren miejski vs. wiejski).
    """)

    # Dane dla tabel kontyngencji
    contingency_tables = {
        '`is_urban_driver`': pd.DataFrame({
            'Wypadek miejski': [15893, 174431],
            'Wypadek wiejski': [34441, 48288]
        }, index=['Kier. wiejski', 'Kier. miejski']),
        '`light_conditions`': pd.DataFrame({
            'Wypadek miejski': [962, 189362],
            'Wypadek wiejski': [7299, 75430]
        }, index=['6.0', '99.0']),
        '`driver_journey_purpose`': pd.DataFrame({
            'Wypadek miejski': [154837, 35487],
            'Wypadek wiejski': [50211, 32518]
        }, index=['0', '1']),
        '`junction_control`': pd.DataFrame({
            'Wypadek miejski': [46255, 139688, 4381],
            'Wypadek wiejski': [7068, 73807, 1854]
        }, index=['2.0', '4.0', '9.0']),
        '`casualty_type`': pd.DataFrame({
            'Wypadek miejski': [108675, 81649],
            'Wypadek wiejski': [62652, 20077]
        }, index=['9.0', '99.0']),
        '`skidding_and_overturning`': pd.DataFrame({
            'Wypadek miejski': [165545, 8322, 1618, 14, 2, 3003, 11820],
            'Wypadek wiejski': [71679, 7137, 1376, 24, 9, 2234, 270]
        }, index=['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '9.0']),
        '`junction_detail`': pd.DataFrame({
            'Wypadek miejski': [19981, 170343],
            'Wypadek wiejski': [14606, 68123]
        }, index=['1.0', '99.0']),
        '`road_type`': pd.DataFrame({
            'Wypadek miejski': [27128, 138106, 25090],
            'Wypadek wiejski': [9607, 61655, 11467]
        }, index=['3', '6', '99'])
    }

    # Wartości oczekiwane (przykładowe, dla „Pochodzenie kierowcy”)
    expected_tables = {
        '`is_urban_driver`': pd.DataFrame({
            'Wypadek miejski': [35083.9, 155240.1],
            'Wypadek wiejski': [15250.1, 67478.9]
        }, index=['Kier. wiejski', 'Kier. miejski']),
        # Uwaga: Dla pozostałych zmiennych wartości oczekiwane można wygenerować na podstawie danych obserwowanych
    }

    # Wyświetlanie szczegółowych wyników
    for idx, (variable, contingency_table) in enumerate(contingency_tables.items(), 1):
        st.markdown(f"**{idx}. Test chi-kwadrat: {variable} vs `is_rural_accident`** (obszar wypadku)")
        st.markdown("**Tabela kontyngencji (obserwowane częstości):**")
        st.dataframe(contingency_table.style.format("{:,.0f}"))

        if variable in expected_tables:
            st.markdown("**Tabela oczekiwana (dla H₀):**")
            st.dataframe(expected_tables[variable].style.format("{:,.1f}"))

        stat = summary_data['Statystyka χ²'][idx-1]
        p_val = summary_data['p-value'][idx-1]
        v_cramer = summary_data['V Craméra'][idx-1]
        strength = summary_data['Siła związku'][idx-1]
        st.markdown(f"""
        *Wyniki testu statystycznego:*
        - *Statystyka chi-kwadrat (χ²):* {stat:.1f}
        - *Wartość p (p-value):* {p_val:.3e}
        - *V Craméra:* {v_cramer:.3f} ({strength} związek)
        - *Wniosek (α = 0.05):* Odrzucamy hipotezę zerową (H₀). Istnieje statystycznie istotny związek (p < 0.0001).
        """)

    # --- Podsumowanie ---
    st.subheader("Podsumowanie wyników testów chi-kwadrat")
    st.markdown("""
    - Wszystkie analizowane cechy wykazują **statystycznie istotny związek** z lokalizacją wypadku `is_rural_accident` (p < 0.001), co potwierdza odrzucenie hipotezy zerowej (H₀) i wspiera wyniki modelowania XGBoost.
    - Najsilniejszy wpływ na lokalizację wypadku ma `is_urban_driver` (pochodzenie kierowcy) (V = 0.394), co potwierdza jego wysoką ważność w modelu XGBoost (3. miejsce), natomiast umiarkowany związek wykazują `light_conditions` (warunki oświetleniowe) (V = 0.223) i `driver_journey_purpose` (odległość od miejsca zamieszkania) (V = 0.220), a także `junction_control` (kontrola skrzyżowania) (V = 0.183), `casualty_type` (typ poszkodowanego) (V = 0.177) oraz `skidding_and_overturning` (poślizg i wywrócenie) (V = 0.161), przy czym `junction_detail` (szczegóły skrzyżowania) (V = 0.099) i `road_type` (typ drogi) (V = 0.036) mają słabszy, ale nadal istotny związek, co odpowiada ich niższej, choć zauważ Baghdad University College of Sciencealnej ważności w modelu.
    - Umiarkowana siła związku dla większości cech (V = 0.161–0.394) wskazuje, że analizowane zmienne są istotne, ale inne czynniki (np. `speed_limit_normalized`, interakcje cech) mogą dodatkowo wpływać na wyniki, jak sugeruje analiza ważności cech XGBoost.
    - Słaby związek dla `junction_detail` (szczegóły skrzyżowania) i `road_type` (typ drogi) sugeruje, że te cechy mogą być mniej uniwersalne lub wymagać bardziej szczegółowych kategorii w przyszłych analizach.
    
    **Wiarygodność wyników:**
    - Duża próba (N = 273,053) zapewnia wysoką wiarygodność wyników testów chi-kwadrat, co jest zgodne z wysoką skutecznością modelu XGBoost (AUC-ROC = 0.9400).
    """)

elif section == "VII. Szczegółowa ocena modeli XGBoost i RandomForest":
    st.title("VII. Szczegółowa Ocena Modeli Uczenia Maszynowego (Krzywe ROC i Uczenia się)")
    st.markdown("Ocena przeprowadzona na zbiorach **walidacyjnym** i **testowym** (bez SMOTE). Próg decyzyjny: 0.5.")

    # --- Krzywe ROC (Odtworzone - przykładowe dane) ---
    st.subheader("Krzywe ROC (Zbiór Testowy)")
    fpr_xgb_static = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
    tpr_xgb_static = np.array([0, 0.6, 0.8, 0.88, 0.92, 0.96, 1])
    fpr_rf_static = np.array([0, 0.07, 0.15, 0.25, 0.35, 0.55, 1])
    tpr_rf_static = np.array([0, 0.55, 0.75, 0.85, 0.90, 0.94, 1])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_xgb_static, y=tpr_xgb_static, mode='lines', name=f'XGBoost (AUC ≈ {0.9400:.4f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_rf_static, y=tpr_rf_static, mode='lines', name=f'Random Forest (AUC ≈ {0.9327:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Losowy Klasyfikator', line=dict(dash='dash')))

    fig_roc.update_layout(
        title='Krzywa ROC - Zbiór Testowy',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend_title='Model',
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("Uwaga: Krzywa ROC jest ilustracją opartą na przykładowych danych dla tej wersji statycznej.")

    # --- Krzywa Uczenia się (Learning Curve) ---
    st.subheader("Krzywa Uczenia się (F1-score) - XGBoost")
    train_sizes = np.array([18271, 36542, 54813, 73084, 91355, 109626, 127897, 146168, 164439, 182710])
    f1_train = np.array([np.nan, 0.85596769, 0.84882457, 0.84334605, 0.83608956, 0.83447398, 0.84805978, 0.87375429, 0.90109748, 0.91796358])
    f1_val = np.array([np.nan, 0.80674855, 0.82824203, 0.8356456, 0.83552517, 0.83828277, 0.84275766, 0.88375834, 0.89837232, 0.90064811])

    valid_indices = ~np.isnan(f1_train) & ~np.isnan(f1_val)
    train_sizes = train_sizes[valid_indices]
    f1_train = f1_train[valid_indices]
    f1_val = f1_val[valid_indices]

    fig_learning = plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, f1_train, label='F1-score XGBoost (trening)', color='blue', marker='o')
    plt.plot(train_sizes, f1_val, label='F1-score XGBoost (walidacja)', color='cyan', marker='o')
    plt.fill_between(train_sizes, f1_train - 0.01, f1_train + 0.01, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, f1_val - 0.01, f1_val + 0.01, alpha=0.1, color='cyan')

    plt.title('Krzywa uczenia (F1-score) - XGBoost', fontsize=14)
    plt.xlabel('Rozmiar zbioru treningowego', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_learning)

    # --- Podsumowanie ---
    st.subheader("Podsumowanie wyników oceny modeli")
    st.markdown("""
    **Kluczowe obserwacje:**
    - **Krzywe ROC**: Model XGBoost osiąga AUC-ROC = 0.9400 na zbiorze testowym, przewyższając Random Forest (AUC-ROC = 0.9327). Oba modele znacząco przewyższają losowy klasyfikator (AUC = 0.5), co potwierdza ich wysoką zdolność do rozróżniania wypadków wiejskich od miejskich.
    - **Krzywa uczenia (XGBoost)**: F1-score na zbiorze treningowym rośnie od 0.856 do 0.918, a na zbiorze walidacyjnym od 0.807 do 0.901 wraz ze wzrostem rozmiaru danych. Stabilizacja wyników na poziomie ~0.90 (walidacja) i ~0.92 (trening) wskazuje na dobrą generalizację modelu z minimalnym ryzykiem nadmiernego dopasowania.
    - **Porównanie modeli**: Wyższa wartość AUC-ROC dla XGBoost oraz stabilne wyniki F1-score potwierdzają jego przewagę nad Random Forest, szczególnie w predykcji wypadków wiejskich.
    - **Kontekst wcześniejszych analiz**: Wysoka skuteczność modeli jest wspierana przez statystycznie istotne cechy zidentyfikowane w testach chi-kwadrat (sekcja VI, np. V Craméra = 0.394 dla „is_urban_driver”), które odpowiadają kluczowym predyktorom w modelowaniu XGBoost (sekcja V).

    **Wiarygodność wyników:**
    - Duża próba danych (N = 273,053) oraz brak zastosowania SMOTE na zbiorach walidacyjnym i testowym zapewniają realistyczne odzwierciedlenie rozkładów danych, zwiększając wiarygodność wyników.
    - Wysoka wartość AUC-ROC (0.9400) i stabilność F1-score na krzywej uczenia wskazują na solidność modelu XGBoost, szczególnie w zadaniu klasyfikacji binarnej.
    - Ilustracyjny charakter krzywych ROC (oparty na przykładowych danych) nie wpływa na ogólne wnioski, które są zgodne z wysoką skutecznością modeli.
    - Spójność wyników z analizą ważności cech (sekcja V) i testami statystycznymi (sekcja VI) wzmacnia zaufanie do uzyskanych rezultatów.
    """)

elif section == "VIII. Podsumowanie i wnioski końcowe":
    st.title("VIII. Podsumowanie i wnioski końcowe")

    st.header("1. Cel pracy i pytania badawcze")
    st.markdown("""
    Celem pracy było zbadanie związku między miejscem zamieszkania kierowcy (miejskim lub wiejskim) a prawdopodobieństwem udziału w wypadku drogowym, z naciskiem na identyfikację kluczowych czynników wpływających na przewidywanie lokalizacji wypadków na terenach wiejskich przy użyciu modeli uczenia maszynowego. Analiza opierała się na danych z brytyjskich baz wypadków drogowych z lat 2021–2023 (N = 273,053). Pytania badawcze koncentrowały się na:
    - Związku między miejscem zamieszkania kierowcy a lokalizacją wypadku (`is_rural_accident`).
    - Kluczowych cechach kontekstowych (np. `road_type`, `light_conditions`) wpływających na wypadki na obszarach wiejskich.
    - Skuteczności modeli uczenia maszynowego (XGBoost, RandomForest) w przewidywaniu lokalizacji wypadków.
    """)

    st.header("2. Kluczowe wyniki")
    
    st.subheader("2.1 Związek między miejscem zamieszkania a lokalizacją wypadku")
    st.markdown("""
    - **Statystyczna istotność**: Test chi-kwadrat (χ² = 42,475.6, p < 0.001, V Craméra = 0.394) potwierdził umiarkowany, statystycznie istotny związek między miejscem zamieszkania kierowcy (`is_urban_driver`) a lokalizacją wypadku (`is_rural_accident`). Kierowcy wiejscy częściej uczestniczą w wypadkach na terenach wiejskich (68.4%) niż miejscy (21.7%), a kierowcy miejscy dominują w wypadkach miejskich (78.3%).
    - **Obalenie hipotezy badawczej**: Hipoteza, że kierowcy miejscy są bardziej narażeni na wypadki na terenach wiejskich, została obalona. Kierowcy wiejscy wykazują wyższe prawdopodobieństwo wypadków w środowisku wiejskim, co może wynikać z większej znajomości dróg miejskich przez kierowców miejskich lub różnic w infrastrukturze drogowej.
    """)

    st.subheader("2.2 Kluczowe czynniki wpływające na wypadki wiejskie")
    st.markdown("""
    - **Najważniejsze cechy (XGBoost)**: Analiza ważności cech w modelu XGBoost wskazała, że ograniczenie prędkości (`speed_limit_normalized`, 18.27%), interakcja prędkości i typu kierowcy (`urban_driver_speed`, 17.11%), pochodzenie kierowcy (`is_urban_driver`, 5.66%) oraz interakcja odległości i prędkości (`distance_speed_interaction`, 5.43%) mają największy wpływ na przewidywanie wypadków wiejskich (`is_rural_accident`). Inne istotne cechy to brak kontroli skrzyżowań (`junction_control`), ciemność bez oświetlenia (`light_conditions`) i typ drogi jednojezdniowej (`road_type`).
    - **Testy chi-kwadrat**: Wszystkie kluczowe cechy wykazały statystycznie istotny związek z lokalizacją wypadku (`is_rural_accident`) (p < 0.001). Najsilniejszy związek miał `is_urban_driver` (V = 0.394), a umiarkowane powiązania dotyczyły `light_conditions` (warunki oświetleniowe) (V = 0.223), `driver_journey_purpose` (odległość od miejsca zamieszkania) (V = 0.220) oraz `junction_control` (kontrola skrzyżowań) (V = 0.183).
    """)

    st.subheader("2.3 Skuteczność modeli uczenia maszynowego")
    st.markdown("""
    - **XGBoost vs. Random Forest**: Model XGBoost osiągnął wyższą skuteczność (AUC-ROC = 0.9400, F1-score dla klasy wiejskiej = 0.81 na zbiorze testowym) w porównaniu do Random Forest (AUC-ROC = 0.9327, F1-score = 0.79). XGBoost wykazał lepszy balans między precyzją a czułością oraz stabilność wyników w przewidywaniu `is_rural_accident`.
    - **Krzywa uczenia się**: Krzywa uczenia dla XGBoost pokazała stabilizację F1-score na poziomie ~0.90 (walidacja) i ~0.92 (trening), wskazując na dobrą generalizację modelu z minimalnym ryzykiem nadmiernego dopasowania.
    - **Wiarygodność modeli**: Duża próba danych i realistyczny rozkład klas w zbiorach walidacyjnym i testowym (bez SMOTE) zapewniają wysoką wiarygodność wyników. Wysoka wartość AUC-ROC potwierdza zdolność modelu do rozróżniania wypadków wiejskich (`is_rural_accident = 1`) od miejskich (`is_rural_accident = 0`).
    """)

    st.header("3. Odpowiedzi na pytania badawcze")
    st.markdown("""
    1. **Czy miejsce zamieszkania kierowcy wpływa na prawdopodobieństwo udziału w wypadku drogowym?**  
       Tak, istnieje statystycznie istotny, umiarkowany związek między miejscem zamieszkania kierowcy (`is_urban_driver`) a lokalizacją wypadku (`is_rural_accident`). Kierowcy wiejscy są bardziej narażeni na wypadki na terenach wiejskich, a kierowcy miejscy na terenach miejskich.
    2. **Jakie cechy kontekstowe mają największy wpływ na wypadki wiejskie?**  
       Kluczowe cechy to wyższe limity prędkości (`speed_limit_normalized`), brak kontroli skrzyżowań (`junction_control`), ciemność bez oświetlenia (`light_conditions`), drogi jednojezdniowe (`road_type`) oraz interakcje między prędkością (`urban_driver_speed`), odległością od miejsca zamieszkania (`driver_journey_purpose`) i pochodzeniem kierowcy (`is_urban_driver`).
    3. **Czy modele uczenia maszynowego mogą skutecznie przewidzieć lokalizację wypadku?**  
       Tak, model XGBoost osiągnął wysoką skuteczność (AUC-ROC = 0.9400), obalając hipotezę, że modele uczenia maszynowego nie są skuteczne w tym zadaniu. Model dobrze radzi sobie z przewidywaniem wypadków wiejskich (`is_rural_accident`), choć precyzja dla tej klasy (0.84) wskazuje na potencjalne obszary do poprawy.
    """)

    st.header("4. Wnioski końcowe")
    st.markdown("""
    - **Związek miejsca zamieszkania z wypadkami**: Miejsce zamieszkania kierowcy (`is_urban_driver`) istotnie wpływa na lokalizację wypadków (`is_rural_accident`), przy czym kierowcy wiejscy są bardziej narażeni na wypadki w środowisku wiejskim, co może wynikać z różnic w infrastrukturze, nawykach jazdy lub warunkach drogowych.
    - **Kluczowe czynniki ryzyka**: Wysokie limity prędkości (`speed_limit_normalized`), brak oświetlenia (`light_conditions`), niekontrolowane skrzyżowania (`junction_control`) i drogi jednojezdniowe (`road_type`) znacząco zwiększają ryzyko wypadków na terenach wiejskich, co powinno być uwzględnione w strategiach prewencji.
    - **Skuteczność modelowania**: Model XGBoost okazał się skutecznym narzędziem do przewidywania lokalizacji wypadków (`is_rural_accident`), oferując wysoką dokładność i możliwość identyfikacji kluczowych czynników ryzyka. Jego przewaga nad Random Forest oraz stabilność wyników potwierdzają przydatność uczenia maszynowego w analizie bezpieczeństwa drogowego.
    - **Praktyczne implikacje**: Wyniki sugerują potrzebę dostosowania działań prewencyjnych do specyfiki obszarów wiejskich, np. poprawy oświetlenia (`light_conditions`), kontroli skrzyżowań (`junction_control`) i edukacji kierowców miejskich podróżujących na tereny wiejskie.
    """)

    st.header("5. Podsumowanie")
    st.markdown("""
    Analiza potwierdziła istotny związek między miejscem zamieszkania kierowcy (`is_urban_driver`) a lokalizacją wypadków drogowych (`is_rural_accident`), identyfikując kluczowe czynniki ryzyka, takie jak prędkość (`speed_limit_normalized`), oświetlenie (`light_conditions`) i kontrola skrzyżowań (`junction_control`). Model XGBoost okazał się skutecznym narzędziem predykcyjnym, oferując wysoką dokładność i cenne wskazówki dla dalszych badań oraz działań prewencyjnych. Wyniki podkreślają znaczenie dostosowania infrastruktury i edukacji kierowców do specyfiki obszarów wiejskich, aby zmniejszyć ryzyko wypadków drogowych.
    """)