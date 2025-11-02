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

# Import modułów podstawowych
from chopin_data_processor import ChopinCompetitionProcessor, process_competition_data
from chopin_advanced_analyzer import ChopinAdvancedAnalyzer, run_advanced_analysis
from chopin_visualization import ChopinVisualization

# Import nowych modułów analitycznych
from chopin_controversy_analyzer import ChopinControversyAnalyzer, run_controversy_analysis
from chopin_statistical_analyzer import ChopinStatisticalAnalyzer, run_statistical_analysis
from chopin_clustering_analyzer import ChopinClusteringAnalyzer, run_clustering_analysis
from chopin_statistical_visualization import ChopinStatisticalVisualization
from chopin_advanced_visualizations import run_advanced_visualizations
from chopin_multistage_clustering import run_multistage_analysis


def create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis'):
    """
    Tworzy kompletny raport analizy konkursu
    POPRAWIONA WERSJA - zawiera wszystkie analizy
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
            f.write("### Sędziowie o największym wpływie na ranking końcowy:\n")
            for _, row in most_influential.iterrows():
                f.write(f"- **{row['judge_removed']}**: usunięcie zmienia ranking średnio o {row['avg_rank_change']:.2f} pozycji\n")
        
        # 5.1 NOWA SEKCJA - Wpływ na kwalifikacje
        f.write("\n### Wpływ usunięcia sędziego na kwalifikacje do kolejnych rund:\n\n")
        qualification_impact = analyzer.analyze_qualification_after_judge_removal()
        
        if not qualification_impact.empty:
            f.write("Analiza pokazuje, jak usunięcie poszczególnych sędziów wpłynęłoby na kwalifikację uczestników do kolejnych etapów:\n\n")
            
            # Znajdź sędziów którzy mają największy wpływ na kwalifikacje
            for _, row in qualification_impact.iterrows():
                judge = row['judge_removed']
                
                # Stage1 -> Stage2
                lost_s1 = row.get('stage1_to_stage2', {}).get('lost_qualification', [])
                gained_s1 = row.get('stage1_to_stage2', {}).get('gained_qualification', [])
                
                # Stage2 -> Stage3
                lost_s2 = row.get('stage2_to_stage3', {}).get('lost_qualification', [])
                gained_s2 = row.get('stage2_to_stage3', {}).get('gained_qualification', [])
                
                # Stage3 -> Final
                lost_s3 = row.get('stage3_to_final', {}).get('lost_qualification', [])
                gained_s3 = row.get('stage3_to_final', {}).get('gained_qualification', [])
                
                total_changes = len(lost_s1) + len(gained_s1) + len(lost_s2) + len(gained_s2) + len(lost_s3) + len(gained_s3)
                
                if total_changes > 0:
                    f.write(f"**{judge}**:\n")
                    if lost_s1 or gained_s1:
                        f.write(f"  - Stage1→Stage2: {len(gained_s1)} nowych, {len(lost_s1)} odpadłoby\n")
                    if lost_s2 or gained_s2:
                        f.write(f"  - Stage2→Stage3: {len(gained_s2)} nowych, {len(lost_s2)} odpadłoby\n")
                    if lost_s3 or gained_s3:
                        f.write(f"  - Stage3→Finał: {len(gained_s3)} nowych, {len(lost_s3)} odpadłoby\n")
        
        # 5.2 NOWA SEKCJA - Finalne wyniki bez sędziów
        f.write("\n### Wpływ usunięcia sędziego na finalne wyniki:\n\n")
        results_after_removal = analyzer.generate_results_after_judge_removal()
        
        if not results_after_removal.empty:
            f.write("Symulacja pełnych zawodów (od Stage1 do Finału) bez poszczególnych sędziów pokazuje:\n\n")
            
            # Znajdź największe zmiany
            rank_change_cols = [col for col in results_after_removal.columns if col.endswith('_change')]
            
            if rank_change_cols:
                # Dla każdego finalisty sprawdź największe wahania
                top_finalists = results_after_removal.head(10)
                
                f.write("**TOP 10 finalistów - stabilność pozycji:**\n\n")
                for _, row in top_finalists.iterrows():
                    name = f"{row['imię']} {row['nazwisko']}"
                    orig_rank = row['original_rank']
                    
                    # Znajdź największą zmianę dla tego uczestnika
                    changes = []
                    for col in rank_change_cols:
                        if row[col] != 'n/a' and row[col] != 'error':
                            changes.append(abs(int(row[col])))
                    
                    if changes:
                        max_change = max(changes)
                        avg_change = np.mean(changes)
                        f.write(f"- **Miejsce {orig_rank}: {name}** - max wahanie: ±{max_change}, średnie: ±{avg_change:.1f}\n")
        
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


def main():
    """
    Główna funkcja uruchamiająca kompletną analizę
    ROZSZERZONA WERSJA - zawiera nowe analizy statystyczne
    """
    print("=" * 60)
    print("ANALIZA KONKURSU CHOPINOWSKIEGO 2025")
    print("Wersja rozszerzona z analizami statystycznymi")
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
    print("\n[1/8] Wczytywanie i przetwarzanie danych...")
    processor = process_competition_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv',
        output_dir='results'
    )
    
    # 2. Zaawansowane analizy
    print("\n[2/8] Przeprowadzanie zaawansowanych analiz...")
    analyzer = run_advanced_analysis(processor, output_dir='advanced_results')
    
    # 3. Wizualizacje podstawowe
    print("\n[3/8] Generowanie wizualizacji podstawowych...")
    visualizer = ChopinVisualization(processor, analyzer)
    visualizer.create_comprehensive_report(output_dir='visualizations')
    
    # 4. NOWE - Analiza zróżnicowania ocen
    print("\n[4/8] Analiza zróżnicowania ocen uczestników...")
    controversy_analyzer = run_controversy_analysis(processor, output_dir='score_diversity_results')
    
    # 5. NOWE - Analizy statystyczne
    print("\n[5/8] Zaawansowane analizy statystyczne...")
    statistical_analyzer = run_statistical_analysis(processor, output_dir='statistical_results')
    
    # 6. NOWE - Clustering i PCA
    print("\n[6/8] Analiza clusteringu i PCA...")
    clustering_analyzer = run_clustering_analysis(processor, output_dir='clustering_results')
    
    # 7. NOWE - Wizualizacje statystyczne
    print("\n[7/8] Generowanie wizualizacji statystycznych...")
    stat_visualizer = ChopinStatisticalVisualization(
        processor, 
        controversy_analyzer=controversy_analyzer,
        statistical_analyzer=statistical_analyzer,
        clustering_analyzer=clustering_analyzer
    )
    stat_visualizer.create_comprehensive_statistical_report(output_dir='visualizations')

    analyzer = run_multistage_analysis(data_files, output_dir='multistage_results')
    run_advanced_visualizations(clustering_analyzer, output_dir='visualizations')
    
    # 8. Raport tekstowy (POPRAWIONY)
    print("\n[8/8] Generowanie raportu tekstowego...")
    create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis')
    
    # 9. Podsumowanie
    print("\n" + "=" * 60)
    print("STRUKTURA WYNIKÓW:")
    print("=" * 60)
    print("📁 results/                      - Podstawowe wyniki i skorygowane oceny")
    print("📁 advanced_results/             - Zaawansowane analizy sędziów")
    print("📁 score_diversity_results/      - Analiza zróżnicowania ocen uczestników")
    print("📁 statistical_results/          - Analizy statystyczne (CI, significance)")
    print("📁 clustering_results/           - Clustering i PCA")
    print("📁 visualizations/               - Wykresy podstawowe")
    print("📁 statistical_visualizations/   - Wykresy statystyczne")
    print("📁 full_analysis/                - Kompletny raport tekstowy")
    
    print("\n📊 Kluczowe pliki:")
    print("  - results/final_cumulative.csv                      - Końcowy ranking")
    print("  - advanced_results/judge_tendencies.csv             - Tendencje sędziów")
    print("  - score_diversity_results/most_diverse_scores.csv   - Najbardziej zróżnicowane oceny")
    print("  - statistical_results/bootstrap_ci_final.csv        - Confidence intervals")
    print("  - statistical_results/significance_final.csv        - Istotność statystyczna")
    print("  - statistical_results/kendall_tau_pairwise.csv      - Zgodność sędziów")
    print("  - clustering_results/kmeans_clusters_final.csv      - Clustering uczestników")
    print("  - clustering_results/judge_pca_scores.csv           - PCA sędziów")
    print("  - full_analysis/analysis_report.md                  - Raport tekstowy")
    
    print("\n✅ Analiza zakończona pomyślnie!")


if __name__ == "__main__":
    main()
