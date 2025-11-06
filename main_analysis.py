"""
Główny skrypt do kompleksowej analizy konkursu Chopinowskiego 2025
Integruje wszystkie moduły i przeprowadza pełną analizę
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Import modułów
from chopin_data_processor import ChopinCompetitionProcessor, process_competition_data
from chopin_advanced_analyzer import ChopinAdvancedAnalyzer, run_advanced_analysis
from chopin_visualization import ChopinVisualization

def create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis'):
    """
    Tworzy kompletny raport analizy konkursu
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Otwórz plik raportu
    report_path = f'{output_dir}/analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Raport Analizy Konkursu Chopinowskiego 2025\n\n")
        f.write(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # 1. Podstawowe statystyki
        f.write("## 1. Podstawowe statystyki\n\n")
        
        # Liczba uczestników per etap
        f.write("### Liczba uczestników w poszczególnych etapach:\n")
        for stage_name, df in processor.stages_data.items():
            f.write(f"- **{stage_name}**: {len(df)} uczestników\n")
        
        # 2. Analiza sędziów
        f.write("\n## 2. Analiza sędziów\n\n")
        
        judge_stats = analyzer.get_judge_statistics()
        if not judge_stats.empty:
            f.write("### Wykorzystanie skali przez sędziów:\n\n")
            scale_usage = analyzer.analyze_scale_usage()
            
            # Top 3 najbardziej liberalni w ocenach
            top_range = scale_usage.nlargest(3, 'overall_range')
            f.write("**Sędziowie używający najszerszej skali:**\n")
            for _, row in top_range.iterrows():
                f.write(f"- {row['judge']}: rozpiętość {row['overall_range']:.1f} punktów\n")
            
            # Top 3 najbardziej konserwatywni
            bottom_range = scale_usage.nsmallest(3, 'overall_range')
            f.write("\n**Sędziowie używający najwęższej skali:**\n")
            for _, row in bottom_range.iterrows():
                f.write(f"- {row['judge']}: rozpiętość {row['overall_range']:.1f} punktów\n")
        
        # 3. Tendencje sędziowskie
        f.write("\n### Tendencje sędziowskie:\n\n")
        tendencies = analyzer.analyze_judge_tendencies()
        
        if not tendencies.empty:
            # Najbardziej surowi
            harsh = tendencies.nsmallest(3, 'overall_harshness')
            f.write("**Najbardziej surowi sędziowie:**\n")
            for _, row in harsh.iterrows():
                f.write(f"- {row['judge']}: średnio {abs(row['overall_harshness']):.2f} punkta poniżej konsensusu\n")
            
            # Najbardziej łagodni
            lenient = tendencies.nlargest(3, 'overall_harshness')
            f.write("\n**Najbardziej łagodni sędziowie:**\n")
            for _, row in lenient.iterrows():
                f.write(f"- {row['judge']}: średnio {row['overall_harshness']:.2f} punkta powyżej konsensusu\n")
        
        # 4. Sojusze sędziowskie
        f.write("\n## 3. Sojusze i korelacje\n\n")
        correlation_matrix, alliances = analyzer.analyze_judge_alliances(threshold=0.7)
        
        if not alliances.empty:
            f.write("### Najsilniejsze sojusze (korelacja > 0.7):\n")
            for _, row in alliances.head(5).iterrows():
                f.write(f"- **{row['judge1']}** i **{row['judge2']}**: korelacja {row['correlation']:.3f}\n")
        
        # 5. Wpływ usunięcia sędziego
        f.write("\n## 4. Wpływ pojedynczych sędziów na wyniki\n\n")
        removal_impact = analyzer.simulate_judge_removal()
        
        if not removal_impact.empty:
            most_influential = removal_impact.nlargest(3, 'avg_rank_change')
            f.write("### Sędziowie o największym wpływie na ranking:\n")
            for _, row in most_influential.iterrows():
                f.write(f"- **{row['judge_removed']}**: usunięcie zmienia ranking średnio o {row['avg_rank_change']:.2f} pozycji\n")
        
        # 6. Faworyci
        f.write("\n## 5. Analiza faworyzowania\n\n")
        favorites = analyzer.find_judge_favorites(min_stages=3)
        
        if not favorites.empty:
            # Najsilniejsze przypadki faworyzowania
            top_fav = favorites[favorites['type'] == 'favorite'].nlargest(3, 'avg_difference')
            if not top_fav.empty:
                f.write("### Najsilniejsze przypadki faworyzowania:\n")
                for _, row in top_fav.iterrows():
                    f.write(f"- **{row['judge']}** → {row['participant_name']}: +{row['avg_difference']:.2f} punkta średnio\n")
            
            # Najsilniejsze przypadki niedoceniania
            top_unfav = favorites[favorites['type'] == 'unfavorite'].nsmallest(3, 'avg_difference')
            if not top_unfav.empty:
                f.write("\n### Najsilniejsze przypadki niedoceniania:\n")
                for _, row in top_unfav.iterrows():
                    f.write(f"- **{row['judge']}** → {row['participant_name']}: {row['avg_difference']:.2f} punkta średnio\n")
        
        # 7. Wyniki finalne
        f.write("\n## 6. Wyniki końcowe\n\n")
        if 'final_cumulative' in processor.cumulative_scores:
            final_results = processor.cumulative_scores['final_cumulative']
            f.write("### TOP 10 finalistów:\n")
            for _, row in final_results.head(10).iterrows():
                f.write(f"{int(row['rank'])}. **{row['imię']} {row['nazwisko']}** - {row['cumulative_score']:.2f} punktów\n")
        
        # 8. Wnioski
        f.write("\n## 7. Kluczowe wnioski\n\n")
        
        # Sprawdź czy system był używany równomiernie
        if not scale_usage.empty:
            avg_coverage = scale_usage['scale_coverage'].mean()
            f.write(f"- **Wykorzystanie skali**: Średnio sędziowie używali {avg_coverage:.1f}% dostępnej skali (1-25)\n")
            
            if avg_coverage < 50:
                f.write("  - ⚠️ Niska dywersyfikacja ocen - sędziowie używają ograniczonego zakresu punktacji\n")
        
        # Sprawdź zgodność sędziów
        if not tendencies.empty:
            avg_consensus = tendencies['consensus_correlation'].mean()
            f.write(f"- **Zgodność oceniania**: Średnia korelacja z konsensusem wynosi {avg_consensus:.3f}\n")
            
            if avg_consensus < 0.7:
                f.write("  - ⚠️ Znaczące różnice w kryteriach oceniania między sędziami\n")
        
        # Sprawdź wpływ pojedynczych sędziów
        if not removal_impact.empty:
            max_impact = removal_impact['avg_rank_change'].max()
            if max_impact > 2:
                f.write(f"- **Wpływ pojedynczych sędziów**: Maksymalny wpływ na ranking to {max_impact:.2f} pozycji\n")
                f.write("  - ⚠️ Niektórzy sędziowie mają nieproporcjonalnie duży wpływ na wyniki\n")
        
        f.write("\n---\n")
        f.write("*Raport wygenerowany automatycznie*\n")
    
    print(f"Raport tekstowy zapisany w: {report_path}")

def visualize_normalization_comparison(normalization_results: dict, save_path: str = None):
    """
    Wizualizuje porównanie końcowych wyników dla różnych metod normalizacji.
    Tworzy 3 tabelki obok siebie (jedna dla każdej normalizacji).

    Uczestnicy z rank=999 są wyświetleni na końcu i przekreśleni.

    Args:
        normalization_results: dict z kluczami 'zscore', 'minmax', 'rank' zawierający DataFrames
        save_path: ścieżka do zapisu pliku
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Kolejność metod normalizacji
    norm_methods = ['zscore', 'minmax', 'rank']
    method_labels = {
        'zscore': 'Z-Score',
        'minmax': 'Min-Max',
        'rank': 'Rank'
    }

    # Figura z 3 subplotami obok siebie
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))

    # Kolory tła na przemian dla wierszy
    bg_colors = ['#F0FFFF', '#F0FFFF']

    for idx, norm_method in enumerate(norm_methods):
        ax = axes[idx]
        ax.axis('off')

        # Pobierz dane dla tej metody
        df = normalization_results[norm_method].copy()

        # 1. ZMIANA SORTOWANIA:
        # Sortuj najpierw po rank (malejąco), a następnie po cumulative_score (malejąco).
        # Używamy False dla rank, aby 1 < 2 < 3, a True dla rank, aby 999 było na końcu.
        # Konwencjonalnie: sortujemy najpierw po rank (rosnąco), a 999 traktujemy jako dużą liczbę.
        df_sorted = df.sort_values(
            ['rank', 'cumulative_score'],
            ascending=[True, False]
        ).reset_index(drop=True)

        # 2. UWZGLĘDNIAMY WSZYSTKICH UCZESTNIKÓW (w tym rank=999)
        n_participants = len(df_sorted)

        # Tło
        bg_color = bg_colors[idx % 2]
        rect_bg = mpatches.Rectangle((0, 0), 1, 1,
                                     linewidth=0,
                                     facecolor=bg_color,
                                     transform=ax.transAxes, zorder=0)
        ax.add_patch(rect_bg)

        # Tytuł metody normalizacji
        ax.text(0.5, 0.97, f'Normalization: {method_labels[norm_method]}',
                ha='center', va='top', fontsize=13, fontweight='bold',
                transform=ax.transAxes)

        # Parametry tabelki
        y_start = 0.91
        line_height = 0.082

        # Nagłówki kolumn
        header_y = y_start + 0.015
        ax.text(0.10, header_y, 'Rank', ha='center', va='center',
                fontsize=10, fontweight='bold', transform=ax.transAxes)
        ax.text(0.42, header_y, 'Contestant', ha='left', va='center',
                fontsize=10, fontweight='bold', transform=ax.transAxes)
        # ax.text(0.85, header_y, 'Wynik', ha='center', va='center',
        #         fontsize=10, fontweight='bold', transform=ax.transAxes)

        # Rysuj linie poziome
        for i in range(n_participants + 1):
            y_line = y_start - i * line_height
            ax.plot([0.05, 0.95], [y_line, y_line], 'k-', linewidth=0.5,
                    alpha=0.3, transform=ax.transAxes, zorder=1)

        # Rysuj linie pionowe
        # col_positions = [0.05, 0.18, 0.75, 0.95]
        col_positions = [0.05, 0.18, 0.95]
        for x_pos in col_positions:
            ax.plot([x_pos, x_pos], [y_start, y_start - n_participants * line_height],
                    'k-', linewidth=0.5, alpha=0.3, transform=ax.transAxes, zorder=1)

        # Kolory dla podium
        rank_colors = {
            1: '#FFD700',  # Złoty
            2: '#C0C0C0',  # Srebrny
            3: '#CD7F32'  # Brązowy
        }

        # Rysuj wiersze z danymi
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            y_pos = y_start - (i + 0.5) * line_height

            rank = int(row['rank'])
            participant = f"{int(row['Nr'])}. {row['imię']} {row['nazwisko']}"
            # score = f"{row['cumulative_score']:.2f}"

            # Ustawienia przekreślenia i koloru tekstu
            text_color = 'black'
            strikethrough_settings = {}
            rank_label = f"{rank}."

            # 3. DODANIE PRZEKREŚLENIA DLA RANK=999
            if rank == 999:
                text_color = 'gray'  # Przyciemnij tekst
                rank_label = 'n/a'  # Zmień etykietę pozycji
                # Przekreślenie przez rysowanie linii poziomej dla każdego tekstu
                # (Najprostsza i najczystsza metoda w Matplotlib)

                # Rysowanie przekreślenia dla uczestnika
                ax.plot([0.20, 0.43], [y_pos, y_pos],
                        'r-', linewidth=1.0, alpha=0.6, transform=ax.transAxes, zorder=3)

                # Kolor tła dla podium (pomijamy dla 999)
            else:
                # Kolor tła dla podium
                if rank in rank_colors:
                    rect_highlight = mpatches.Rectangle(
                        (0.05, y_start - (i + 1) * line_height),
                        0.90, line_height,
                        linewidth=0,
                        facecolor=rank_colors[rank],
                        alpha=0.15,
                        transform=ax.transAxes, zorder=1
                    )
                    ax.add_patch(rect_highlight)

            # Pozycja w rankingu
            ax.text(0.115, y_pos, rank_label, ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold' if rank != 999 else 'normal',
                    transform=ax.transAxes, zorder=2, **strikethrough_settings)

            # Uczestnik
            ax.text(0.20, y_pos, participant, ha='left', va='center',
                    fontsize=10, color=text_color,
                    transform=ax.transAxes, zorder=2, **strikethrough_settings)

            # # Wynik skumulowany
            # ax.text(0.85, y_pos, score, ha='center', va='center',
            #         fontsize=10, color=text_color, fontweight='bold',
            #         transform=ax.transAxes, zorder=2, **strikethrough_settings)

        # Ramka zewnętrzna
        rect = mpatches.Rectangle((0.05, y_start - n_participants * line_height),
                                  0.90, n_participants * line_height,
                                  linewidth=1.5, edgecolor='black',
                                  facecolor='none', transform=ax.transAxes, zorder=4)
        ax.add_patch(rect)

    # Tytuł główny
    fig.suptitle('Comparison of final rankings across normalization methods',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is None:
        save_path = 'normalization_comparison.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano wizualizację porównania normalizacji do: {save_path}")
    plt.close()

def analyze_normalization_impact(save_path):
    """
    Analizuje wpływ normalizacji na wyniki konkursu
    """
    print("\n=== ANALIZA WPŁYWU NORMALIZACJI ===\n")

    QUALIFICATION_THRESHOLDS = {
        'stage1': 40,  # top 40 przechodzi do stage2
        'stage2': 20,  # top 20 przechodzi do stage3
        'stage3': 10,  # top 10 przechodzi do finału
        'real_stage3': 11  # top 11 przeszło do finału
    }
    # Wczytaj dane
    processor = ChopinCompetitionProcessor()
    processor.load_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    )
    
    # Oblicz wyniki oryginalne
    processor.calculate_all_stages()
    processor.calculate_cumulative_scores()
    original_final = processor.cumulative_scores.get('final_cumulative')

    if original_final is None:
        print("Brak wyników finalnych")
        return
    
    # Przygotuj analizator
    analyzer = ChopinAdvancedAnalyzer(processor)
    
    # Pobierz znormalizowane dane
    normalized_data = analyzer.normalize_scores()
    
    # Dla każdej metody normalizacji, oblicz nowe wyniki
    normalization_results = {}
    
    for norm_method in ['zscore', 'minmax', 'rank']:
        print(f"Przetwarzanie normalizacji: {norm_method}")
        
        # Stwórz nowy procesor z znormalizowanymi danymi
        norm_processor = ChopinCompetitionProcessor()
        norm_processor.judge_columns = processor.judge_columns
        norm_processor.weights = processor.weights
        norm_processor.deviation_limits = processor.deviation_limits
        
        # Użyj znormalizowanych danych
        norm_processor.stages_data = {}
        for stage in ['stage1', 'stage2', 'stage3', 'final']:
            key = f'{stage}_{norm_method}'
            if key in normalized_data:
                norm_processor.stages_data[stage] = normalized_data[key]
        
        # Oblicz wyniki
        norm_processor.calculate_all_stages()
        norm_processor.calculate_cumulative_scores()

        # -----------
        exclude_from_finals = set()
        # Sprawdź kto przeszedłby do stage2
        new_stage1 = norm_processor.cumulative_scores.get('stage1_cumulative')
        stage2_qualifiers = set(
            new_stage1.nsmallest(QUALIFICATION_THRESHOLDS['stage1'], 'rank')['Nr'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['Nr'] in stage2_qualifiers:
                exclude_from_finals.add(int(row['Nr']))

        # Sprawdź kto przeszedłby do stage3
        new_stage2 = norm_processor.cumulative_scores.get('stage2_cumulative')
        stage3_qualifiers = set(
            new_stage2.nsmallest(QUALIFICATION_THRESHOLDS['stage2'], 'rank')['Nr'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['Nr'] in stage3_qualifiers:
                exclude_from_finals.add(int(row['Nr']))

        # Sprawdź kto przeszedłby do finału
        new_stage3 = norm_processor.cumulative_scores.get('stage3_cumulative')
        final_qualifiers = set(
            new_stage3.nsmallest(QUALIFICATION_THRESHOLDS['stage3'], 'rank')['Nr'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['Nr'] in final_qualifiers:
                exclude_from_finals.add(int(row['Nr']))
        # -----------
        norm_processor.calculate_cumulative_scores(exclude_from_finals)

        if 'final_cumulative' in norm_processor.cumulative_scores:
            normalization_results[norm_method] = norm_processor.cumulative_scores['final_cumulative']
            # print(f'WYNIKI PO NORMALIZACJI TYPU {norm_method}')
            # print(normalization_results[norm_method])
            # print()
    visualize_normalization_comparison(normalization_results, save_path)
    # Porównaj wyniki
    print("\n=== PORÓWNANIE WYNIKÓW PO NORMALIZACJI ===\n")
    
    comparison_data = []
    
    for idx, row in original_final.iterrows():
        participant = f"{row['imię']} {row['nazwisko']}"
        nr = row['Nr']
        
        comparison = {
            'Uczestnik': participant,
            'Nr': nr,
            'Oryginał_rank': row['rank'],
            'Oryginał_score': row['cumulative_score']
        }
        
        for norm_method, norm_df in normalization_results.items():
            if nr in norm_df['Nr'].values:
                norm_row = norm_df[norm_df['Nr'] == nr].iloc[0]
                comparison[f'{norm_method}_rank'] = norm_row['rank']
                comparison[f'{norm_method}_score'] = norm_row['cumulative_score']
                comparison[f'{norm_method}_rank_change'] = row['rank'] - norm_row['rank']
        
        comparison_data.append(comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Wypisz największe zmiany
    print("Najistotniejsze zmiany po normalizacji Z-score:")
    if 'zscore_rank_change' in comparison_df.columns:
        top_changes = comparison_df.nlargest(10, 'zscore_rank_change', keep='all')
        for _, row in top_changes.iterrows():
            print(f"  {row['Uczestnik']}: {row['Oryginał_rank']:.0f} → {row['zscore_rank']:.0f} "
                  f"(zmiana: {row['zscore_rank_change']:+.0f})")
    
    # Zapisz pełne porównanie
    comparison_df.to_csv('normalization_comparison.csv', index=False)
    print("\nPełne porównanie zapisane w: normalization_comparison.csv")
    
    # Wizualizacja zmian
    if 'zscore_rank_change' in comparison_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, norm_method in enumerate(['zscore', 'minmax', 'rank']):
            ax = axes[i]
            col = f'{norm_method}_rank_change'
            
            if col in comparison_df.columns:
                changes = comparison_df[col].abs()
                
                ax.hist(changes, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(changes.mean(), color='red', linestyle='--', 
                          label=f'Średnia: {changes.mean():.2f}')
                ax.set_xlabel('Zmiana pozycji w rankingu', fontsize=12)
                ax.set_ylabel('Liczba uczestników', fontsize=12)
                ax.set_title(f'Wpływ normalizacji {norm_method.upper()}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Wpływ normalizacji na ranking końcowy', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('normalization_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nWykres zapisany w: normalization_impact.png")
    
    return comparison_df

def main():
    """
    Główna funkcja uruchamiająca kompletną analizę
    """
    print("=" * 60)
    print("ANALIZA KONKURSU CHOPINOWSKIEGO 2025")
    print("=" * 60)
    
    # Sprawdź czy pliki istnieją
    required_files = [
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n⚠️ Brakujące pliki:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nUpewnij się, że wszystkie pliki CSV znajdują się w bieżącym katalogu.")
        return
    
    # 1. Przetworzenie podstawowych danych
    print("\n[1/6] Wczytywanie i przetwarzanie danych...")
    processor = process_competition_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv',
        output_dir='results'
    )
    
    # 2. Zaawansowane analizy
    print("\n[2/6] Przeprowadzanie zaawansowanych analiz...")
    analyzer = run_advanced_analysis(processor, output_dir='advanced_results')
    
    # 3. Wizualizacje
    print("\n[3/6] Generowanie wizualizacji...")
    visualizer = ChopinVisualization(processor, analyzer)
    visualizer.create_comprehensive_report(output_dir='visualizations')
    
    # 4. Analiza normalizacji
    print("\n[4/6] Analiza wpływu normalizacji...")
    normalization_comparison = analyze_normalization_impact()
    
    # 5. Raport tekstowy
    print("\n[5/6] Generowanie raportu tekstowego...")
    create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis')
    
    # 6. Podsumowanie
    print("\n[6/6] Zakończono!")
    print("\n" + "=" * 60)
    print("STRUKTURA WYNIKÓW:")
    print("=" * 60)
    print("📁 results/           - Podstawowe wyniki i skorygowane oceny")
    print("📁 advanced_results/  - Zaawansowane analizy statystyczne")
    print("📁 visualizations/    - Wszystkie wykresy i wizualizacje")
    print("📁 full_analysis/     - Kompletny raport tekstowy")
    print("\n📊 Kluczowe pliki:")
    print("  - results/final_cumulative.csv          - Końcowy ranking")
    print("  - advanced_results/judge_tendencies.csv - Tendencje sędziów")
    print("  - advanced_results/judge_alliances.csv  - Sojusze sędziowskie")
    print("  - normalization_comparison.csv          - Wpływ normalizacji")
    print("  - full_analysis/analysis_report.md      - Raport tekstowy")
    
    # Przykład analizy konkretnego uczestnika
    print("\n" + "=" * 60)
    print("PRZYKŁAD: Analiza przebiegu dla uczestnika nr 73")
    print("=" * 60)
    visualizer.visualize_participant_journey(73, save_path='participant_73_analysis.png')
    
    print("\n✅ Analiza zakończona pomyślnie!")


if __name__ == "__main__":
    main()
