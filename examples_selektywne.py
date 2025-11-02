"""
PRZYKŁADY UŻYCIA - Selektywne analizy

Ten skrypt pokazuje jak wykonywać tylko wybrane analizy zamiast wszystkiego naraz.
"""

from chopin_multistage_clustering import MultiStageClusteringAnalyzer
from chopin_multistage_clustering import (
    visualize_participant_dendrogram,
    visualize_judge_dendrogram,
    visualize_kmeans_clusters,
    visualize_participant_progression
)
from chopin_advanced_visualizations import (
    visualize_pca_participants,
    visualize_pca_judges,
    visualize_judge_consistency,
    visualize_multistage_heatmap,
    visualize_participant_trajectories,
    visualize_cluster_evolution
)


def przykład_1_podstawowe_klastry():
    """Tylko hierarchiczne klasterowanie uczestników i sędziów"""
    print("=" * 80)
    print("PRZYKŁAD 1: Podstawowe dendrogramy")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # Dendrogram uczestników
    print("\nTworzę dendrogram uczestników...")
    linkage_matrix, labels, _, _ = analyzer.cluster_participants_hierarchical(min_stages=2)
    visualize_participant_dendrogram(linkage_matrix, labels, 
                                    save_path='example_participant_dendrogram.png')
    
    # Dendrogram sędziów
    print("Tworzę dendrogram sędziów...")
    linkage_matrix, labels, _ = analyzer.cluster_judges_hierarchical()
    visualize_judge_dendrogram(linkage_matrix, labels,
                               save_path='example_judge_dendrogram.png')
    
    print("\n✓ Gotowe! Zobacz pliki: example_participant_dendrogram.png, example_judge_dendrogram.png")


def przykład_2_kmeans():
    """Tylko K-means dla uczestników"""
    print("=" * 80)
    print("PRZYKŁAD 2: K-means clustering uczestników")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # K-means z 5 klastrami
    print("\nK-means clustering (5 klastrów)...")
    results_df, cluster_stats_df, kmeans, scaler, profile = \
        analyzer.kmeans_cluster_participants(n_clusters=5, min_stages=2)
    
    # Zapisz wyniki
    results_df.to_csv('example_participant_clusters.csv', index=False, encoding='utf-8-sig')
    cluster_stats_df.to_csv('example_cluster_stats.csv', index=False, encoding='utf-8-sig')
    
    # Wizualizacja
    visualize_kmeans_clusters(results_df, cluster_stats_df,
                             save_path='example_kmeans_clusters.png')
    
    print("\n✓ Gotowe!")
    print("  - example_participant_clusters.csv (przypisania do klastrów)")
    print("  - example_cluster_stats.csv (statystyki klastrów)")
    print("  - example_kmeans_clusters.png (wizualizacja)")
    
    # Wyświetl podsumowanie klastrów
    print("\nPodsumowanie klastrów:")
    for idx, row in cluster_stats_df.iterrows():
        print(f"  Klaster {row['cluster']}: {row['n_members']} uczestników, "
              f"średnia ocena: {row['avg_mean_score']:.2f}")


def przykład_3_progresja():
    """Tylko analiza progresji uczestników"""
    print("=" * 80)
    print("PRZYKŁAD 3: Progresja uczestników")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # Analiza progresji
    print("\nAnalizuję progresję uczestników...")
    progression_df = analyzer.analyze_participant_progression()
    
    # Zapisz
    progression_df.to_csv('example_progression.csv', index=False, encoding='utf-8-sig')
    
    # Wizualizacja
    visualize_participant_progression(progression_df,
                                     save_path='example_progression.png')
    
    print("\n✓ Gotowe!")
    print("  - example_progression.csv")
    print("  - example_progression.png")
    
    # Znajdź uczestników którzy najbardziej się poprawili
    print("\nUczestnicy którzy przeszli przez >= 3 etapy:")
    multi_stage = progression_df.groupby('Nr').size()
    multi_stage = multi_stage[multi_stage >= 3]
    
    for nr in multi_stage.index[:5]:
        participant = progression_df[progression_df['Nr'] == nr]
        first_score = participant.iloc[0]['mean_score']
        last_score = participant.iloc[-1]['mean_score']
        improvement = last_score - first_score
        name = f"{participant.iloc[0]['imię']} {participant.iloc[0]['nazwisko']}"
        print(f"  {nr}: {name} - zmiana: {improvement:+.2f} ({first_score:.2f} → {last_score:.2f})")


def przykład_4_pca():
    """Tylko PCA uczestników i sędziów"""
    print("=" * 80)
    print("PRZYKŁAD 4: Analiza PCA")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # PCA uczestników
    print("\nPCA uczestników...")
    pca, components, variance_df = visualize_pca_participants(
        analyzer, n_components=3, save_path='example_pca_participants.png')
    variance_df.to_csv('example_pca_variance.csv', index=False, encoding='utf-8-sig')
    
    # PCA sędziów
    print("PCA sędziów...")
    visualize_pca_judges(analyzer, save_path='example_pca_judges.png')
    
    print("\n✓ Gotowe!")
    print("  - example_pca_participants.png")
    print("  - example_pca_judges.png")
    print("  - example_pca_variance.csv")
    
    # Wyświetl wyjaśnioną wariancję
    print("\nWyjaśniona wariancja:")
    for idx, row in variance_df.iterrows():
        print(f"  {row['component']}: {row['variance_explained']:.2%} "
              f"(kumulatywnie: {row['cumulative_variance']:.2%})")


def przykład_5_konsystencja_sędziów():
    """Tylko analiza konsystencji sędziów"""
    print("=" * 80)
    print("PRZYKŁAD 5: Konsystencja sędziów")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # Analiza konsystencji
    print("\nAnalizuję konsystencję sędziów...")
    consistency_df = visualize_judge_consistency(
        analyzer, save_path='example_judge_consistency.png')
    
    if consistency_df is not None:
        consistency_df.to_csv('example_consistency.csv', index=False, encoding='utf-8-sig')
        
        print("\n✓ Gotowe!")
        print("  - example_judge_consistency.png")
        print("  - example_consistency.csv")
        
        # Top 5 najbardziej konsystentnych
        print("\nTop 5 najbardziej konsystentnych sędziów:")
        top5 = consistency_df.nlargest(5, 'mean_correlation')
        for idx, row in top5.iterrows():
            print(f"  {row['judge']}: korelacja = {row['mean_correlation']:.3f}")
        
        # Top 5 najmniej konsystentnych
        print("\nTop 5 najmniej konsystentnych sędziów:")
        bottom5 = consistency_df.nsmallest(5, 'mean_correlation')
        for idx, row in bottom5.iterrows():
            print(f"  {row['judge']}: korelacja = {row['mean_correlation']:.3f}")


def przykład_6_heatmapa():
    """Tylko gigantyczna heatmapa wszystkich ocen"""
    print("=" * 80)
    print("PRZYKŁAD 6: Heatmapa wszystkich ocen")
    print("=" * 80)
    
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # Heatmapa
    print("\nTworzę heatmapę wszystkich ocen...")
    visualize_multistage_heatmap(analyzer, 
                                 save_path='example_multistage_heatmap.png')
    
    print("\n✓ Gotowe!")
    print("  - example_multistage_heatmap.png")
    print("\nHeatmapa pokazuje WSZYSTKIE oceny ze WSZYSTKICH etapów.")
    print("Niebieskie linie pionowe oddzielają etapy.")


def menu():
    """Interaktywne menu wyboru przykładów"""
    print("\n" + "=" * 80)
    print("PRZYKŁADY UŻYCIA - ANALIZA WIELOETAPOWA")
    print("=" * 80)
    print("\nWybierz przykład do uruchomienia:\n")
    print("1. Podstawowe dendrogramy (uczestników + sędziów)")
    print("2. K-means clustering uczestników")
    print("3. Progresja uczestników przez etapy")
    print("4. PCA (uczestników + sędziów)")
    print("5. Konsystencja sędziów między etapami")
    print("6. Heatmapa wszystkich ocen")
    print("7. Uruchom wszystkie przykłady")
    print("0. Wyjdź")
    print()
    
    choice = input("Twój wybór (0-7): ").strip()
    
    examples = {
        '1': przykład_1_podstawowe_klastry,
        '2': przykład_2_kmeans,
        '3': przykład_3_progresja,
        '4': przykład_4_pca,
        '5': przykład_5_konsystencja_sędziów,
        '6': przykład_6_heatmapa,
    }
    
    if choice == '0':
        print("Do widzenia!")
        return
    elif choice == '7':
        print("\nUruchamiam wszystkie przykłady...\n")
        for func in examples.values():
            func()
            print()
    elif choice in examples:
        examples[choice]()
    else:
        print("Nieprawidłowy wybór!")


if __name__ == "__main__":
    import os
    import sys
    
    # Sprawdź czy pliki istnieją
    data_files = [
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    ]
    
    missing = [f for f in data_files if not os.path.exists(f)]
    if missing:
        print("BŁĄD: Brak następujących plików:")
        for f in missing:
            print(f"  - {f}")
        print("\nUpewnij się, że wszystkie pliki CSV są w tym samym katalogu!")
        sys.exit(1)
    
    # Uruchom menu
    menu()
