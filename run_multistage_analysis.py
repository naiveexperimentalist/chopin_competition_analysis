"""
GŁÓWNY SKRYPT DO ANALIZY WIELOETAPOWEJ KONKURSU CHOPINOWSKIEGO

Uruchamia pełną analizę klastrową przez wszystkie etapy konkursu.
"""

import os
from chopin_multistage_clustering import run_multistage_analysis
from chopin_advanced_visualizations import run_advanced_visualizations


def main():
    """
    Główna funkcja uruchamiająca całą analizę
    """
    print("=" * 80)
    print("PEŁNA ANALIZA WIELOETAPOWA KONKURSU CHOPINOWSKIEGO")
    print("=" * 80)
    print()
    
    # Ścieżki do plików CSV
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    # Sprawdź czy pliki istnieją
    print("Sprawdzanie plików...")
    missing_files = []
    for stage, filepath in data_files.items():
        if not os.path.exists(filepath):
            missing_files.append(filepath)
            print(f"  ✗ Brak: {filepath}")
        else:
            print(f"  ✓ Znaleziono: {filepath}")
    
    if missing_files:
        print(f"\nBŁĄD: Brak {len(missing_files)} plików!")
        print("Upewnij się, że wszystkie pliki CSV są w tym samym katalogu co skrypt.")
        return
    
    print("\n" + "=" * 80)
    print("CZĘŚĆ 1: PODSTAWOWA ANALIZA KLASTROWA")
    print("=" * 80)
    
    # Uruchom podstawową analizę
    analyzer = run_multistage_analysis(data_files, output_dir='multistage_results')
    
    print("\n" + "=" * 80)
    print("CZĘŚĆ 2: ZAAWANSOWANE WIZUALIZACJE")
    print("=" * 80)
    
    # Uruchom zaawansowane wizualizacje
    run_advanced_visualizations(analyzer, output_dir='multistage_advanced')
    
    print("\n" + "=" * 80)
    print("ANALIZA ZAKOŃCZONA!")
    print("=" * 80)
    print("\nWyniki zapisane w:")
    print("  - multistage_results/      (podstawowe analizy i klastry)")
    print("  - multistage_advanced/     (zaawansowane wizualizacje)")
    print()
    print("Najważniejsze pliki:")
    print("  📊 multistage_results/participant_dendrogram.png")
    print("     -> Hierarchiczne klasterowanie uczestników przez wszystkie etapy")
    print()
    print("  📊 multistage_results/judge_dendrogram.png")
    print("     -> Hierarchiczne klasterowanie sędziów (wzorce oceniania)")
    print()
    print("  📊 multistage_results/participant_clusters_k5.png")
    print("     -> K-means klasterowanie uczestników (5 klastrów)")
    print()
    print("  📊 multistage_results/participant_progression.png")
    print("     -> Progresja uczestników przez etapy")
    print()
    print("  📊 multistage_advanced/participant_pca.png")
    print("     -> PCA - projekcja uczestników w przestrzeni 2D/3D")
    print()
    print("  📊 multistage_advanced/judge_pca.png")
    print("     -> PCA - podobieństwa między sędziami")
    print()
    print("  📊 multistage_advanced/multistage_heatmap.png")
    print("     -> Heatmapa wszystkich ocen ze wszystkich etapów")
    print()
    print("  📊 multistage_advanced/judge_consistency.png")
    print("     -> Analiza konsystencji sędziów między etapami")
    print()
    print("  📊 multistage_advanced/participant_trajectories.png")
    print("     -> Trajektorie uczestników (jak ewoluowały ich oceny)")
    print()
    print("  📊 multistage_advanced/cluster_evolution.png")
    print("     -> Jak zmieniały się przypisania do klastrów przez etapy")
    print()
    print("Pliki CSV z wynikami:")
    print("  📄 multistage_results/participant_clusters_k*.csv")
    print("  📄 multistage_results/judge_clusters_k*.csv")
    print("  📄 multistage_results/participant_progression.csv")
    print("  📄 multistage_advanced/judge_consistency.csv")
    print("  📄 multistage_advanced/pca_variance_explained.csv")
    print()


if __name__ == "__main__":
    main()
