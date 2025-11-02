#!/usr/bin/env python3
"""
Przykłady użycia analizy stabilności wyniku końcowego
Dostosuj do swoich potrzeb!
"""

from chopin_final_score_stability import (
    FinalScoreStabilityAnalyzer,
    FinalScoreStabilityVisualizer
)
import time


def quick_analysis():
    """Szybka analiza - 5000 iteracji"""
    print("\n" + "="*70)
    print("SZYBKA ANALIZA (5000 iteracji)")
    print("="*70 + "\n")
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    start = time.time()
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=5000)
    
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results,
        output_dir='stability_quick'
    )
    
    elapsed = time.time() - start
    print(f"\nCzas wykonania: {elapsed:.1f} sekund")


def detailed_analysis():
    """Szczegółowa analiza - 20000 iteracji"""
    print("\n" + "="*70)
    print("SZCZEGÓŁOWA ANALIZA (20000 iteracji)")
    print("="*70 + "\n")
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    start = time.time()
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=20000)
    
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results,
        output_dir='stability_detailed'
    )
    
    elapsed = time.time() - start
    print(f"\nCzas wykonania: {elapsed:.1f} sekund")


def finalists_only():
    """Analiza wszystkich finalistów"""
    print("\n" + "="*70)
    print("ANALIZA FINALISTÓW (10000 iteracji)")
    print("="*70 + "\n")
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)
    
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results,
        output_dir='stability_finalists'
    )


def custom_visualizations():
    """Własne kombinacje wizualizacji"""
    print("\n" + "="*70)
    print("NIESTANDARDOWE WIZUALIZACJE")
    print("="*70 + "\n")
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)
    
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    
    # Tylko rozkłady wyników dla wszystkich finalistów
    print("Generuję rozkłady wyników dla finalistów...")
    visualizer.visualize_score_distributions(
        bootstrap_results,
        save_path='custom_distributions.png'
    )
    
    # Tylko przedziały ufności z innym poziomem (90%)
    print("Generuję przedziały ufności 90%...")
    visualizer.visualize_confidence_intervals(
        bootstrap_results,
        confidence=0.90,
        save_path='custom_ci_90.png'
    )
    
    # Macierz stabilności dla wszystkich finalistów
    print("Generuję macierz stabilności...")
    visualizer.visualize_ranking_stability_matrix(
        bootstrap_results,
        save_path='custom_matrix.png'
    )
    
    print("\nWizualizacje zapisane!")


def print_statistics():
    """Wyświetl tylko statystyki tekstowe bez wizualizacji"""
    print("\n" + "="*70)
    print("STATYSTYKI STABILNOŚCI (bez wizualizacji)")
    print("="*70 + "\n")
    
    import numpy as np
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)
    
    actual_scores = analyzer.get_actual_final_scores()
    
    print(f"\n{'Ranga':<6} {'Uczestnik':<30} {'Wynik':<8} {'SD':<7} {'95% CI':<20} {'Szerokość'}")
    print("-" * 90)
    
    for _, row in actual_scores.iterrows():
        nr = row['Nr']
        scores = bootstrap_results[nr]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_low = np.percentile(scores, 2.5)
        ci_high = np.percentile(scores, 97.5)
        ci_width = ci_high - ci_low
        
        name = f"{row['imię']} {row['nazwisko']}"[:29]
        
        print(f"{row['rank']:<6} {name:<30} {row['final_score']:<8.2f} "
              f"{std_score:<7.3f} [{ci_low:6.2f}, {ci_high:6.2f}]  {ci_width:6.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'quick':
            quick_analysis()
        elif mode == 'detailed':
            detailed_analysis()
        elif mode == 'finalists':
            finalists_only()
        elif mode == 'custom':
            custom_visualizations()
        elif mode == 'stats':
            print_statistics()
        else:
            print("Dostępne tryby:")
            print("  python examples_stability.py quick      - szybka analiza")
            print("  python examples_stability.py detailed   - szczegółowa analiza")
            print("  python examples_stability.py finalists  - analiza finalistów")
            print("  python examples_stability.py custom     - niestandardowe wizualizacje")
            print("  python examples_stability.py stats      - tylko statystyki tekstowe")
    else:
        print("\n" + "="*70)
        print("PRZYKŁADY UŻYCIA ANALIZY STABILNOŚCI")
        print("="*70 + "\n")
        print("Wybierz tryb:")
        print()
        print("1. quick      - Szybka analiza (5000 iteracji)")
        print("2. detailed   - Szczegółowa analiza (20000 iteracji)")
        print("3. finalists  - Analiza finalistów (10000 iteracji)")
        print("4. custom     - Niestandardowe wizualizacje")
        print("5. stats      - Tylko statystyki tekstowe")
        print()
        print("Użycie:")
        print("  python examples_stability.py [tryb]")
        print()
        print("Przykład:")
        print("  python examples_stability.py quick")
