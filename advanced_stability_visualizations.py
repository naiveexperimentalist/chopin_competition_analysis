"""
Zaawansowane wizualizacje stabilności wyniku końcowego
- Potencjalne zmiany w rankingu
- Overlapping confidence intervals
- Stability score per participant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chopin_final_score_stability import FinalScoreStabilityAnalyzer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedStabilityVisualizer:
    """Zaawansowane wizualizacje stabilności"""
    
    def __init__(self, analyzer: FinalScoreStabilityAnalyzer, 
                 bootstrap_results: dict):
        self.analyzer = analyzer
        self.bootstrap_results = bootstrap_results
    
    def calculate_rank_changes_data(self):
        """
        Oblicza dane o potencjalnych zmianach w rankingu
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Dla każdej iteracji bootstrapu oblicz rankingi
        n_iterations = len(list(self.bootstrap_results.values())[0])
        all_ranks = {nr: [] for nr in actual_scores['Nr']}
        
        print("Obliczam możliwe zmiany w rankingu...")
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")
            
            # Pobierz wyniki wszystkich w tej iteracji
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['Nr']
                iter_scores.append({
                    'nr': nr,
                    'score': self.bootstrap_results[nr][iteration]
                })
            
            # Ranking
            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['rank'] = range(1, len(iter_df) + 1)
            
            for _, row in iter_df.iterrows():
                all_ranks[row['nr']].append(row['rank'])
        
        # Przygotuj dane do wizualizacji
        rank_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            ranks = all_ranks[nr]
            
            rank_data.append({
                'participant': f"{row['imię']} {row['nazwisko']}",
                'actual_rank': row['rank'],
                'best_rank': np.percentile(ranks, 5),  # 5th percentile
                'worst_rank': np.percentile(ranks, 95),  # 95th percentile
                'median_rank': np.median(ranks),
                'rank_range': np.percentile(ranks, 95) - np.percentile(ranks, 5)
            })
        
        df = pd.DataFrame(rank_data)
        df = df.sort_values('actual_rank')
        
        return df
    
    def visualize_combined_stability_analysis(self, save_path: str = None):
        """
        Połączona wizualizacja: rank changes + stability scores (4 wykresy w jednym)
        """
        # Oblicz dane
        rank_changes_df = self.calculate_rank_changes_data()
        stability_df = self.calculate_stability_scores()
        
        # Stwórz figurę 2x2
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # [0,0] Zakres pozycji dla każdego uczestnika
        ax1 = axes[0, 0]
        y_pos = range(len(rank_changes_df))
        
        for i, row in rank_changes_df.iterrows():
            ax1.plot([row['best_rank'], row['worst_rank']], 
                    [i, i], 'o-', linewidth=2, markersize=4, alpha=0.6)
            ax1.plot(row['actual_rank'], i, 'ro', markersize=8, zorder=10)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['actual_rank']}. {row['participant'][:25]}" 
                             for _, row in rank_changes_df.iterrows()], fontsize=9)
        ax1.set_xlabel('Pozycja w rankingu', fontsize=12)
        ax1.set_ylabel('Uczestnik (rzeczywista pozycja. nazwisko)', fontsize=12)
        ax1.set_title('Możliwe zmiany pozycji w rankingu\n' +
                     'Linia = 90% przedział możliwych pozycji, Czerwony punkt = rzeczywista pozycja',
                     fontsize=13, fontweight='bold')
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # [0,1] Rozpiętość możliwych pozycji
        ax2 = axes[0, 1]
        df_sorted_by_range = rank_changes_df.sort_values('rank_range', ascending=True)
        colors2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_sorted_by_range)))
        
        bars = ax2.barh(range(len(df_sorted_by_range)), 
                       df_sorted_by_range['rank_range'],
                       color=colors2, alpha=0.7)
        
        ax2.set_yticks(range(len(df_sorted_by_range)))
        ax2.set_yticklabels([f"{row['actual_rank']}. {row['participant'][:25]}" 
                            for _, row in df_sorted_by_range.iterrows()], fontsize=9)
        ax2.set_xlabel('Rozpiętość możliwych pozycji', fontsize=12)
        ax2.set_ylabel('Uczestnik', fontsize=12)
        ax2.set_title('Niestabilność pozycji w rankingu\n' +
                     'Największa możliwa zmiana pozycji',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # [1,0] Stability score vs pozycja w rankingu
        ax3 = axes[1, 0]
        
        scatter = ax3.scatter(stability_df['rank'], stability_df['stability_score'],
                            s=120, alpha=0.6, c=stability_df['stability_score'],
                            cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        # Annotacje dla wszystkich (jest ich tylko 11)
        for _, row in stability_df.iterrows():
            ax3.annotate(f"{row['rank']}", 
                        (row['rank'], row['stability_score']),
                        fontsize=9, alpha=0.8)
        
        ax3.set_xlabel('Pozycja w rankingu', fontsize=12)
        ax3.set_ylabel('Stability Score (0-100)', fontsize=12)
        ax3.set_title('Stability Score vs Pozycja\n' +
                     '100 = najbardziej stabilny, 0 = najmniej stabilny',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Stability Score')
        
        # [1,1] Najbardziej stabilni
        ax4 = axes[1, 1]
        
        top_stable = stability_df.sort_values('stability_score', ascending=True)
        colors4 = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_stable)))
        
        bars4 = ax4.barh(range(len(top_stable)), top_stable['stability_score'],
                        color=colors4, alpha=0.7)
        ax4.set_yticks(range(len(top_stable)))
        ax4.set_yticklabels([f"{row['rank']}. {row['participant'][:25]}" 
                            for _, row in top_stable.iterrows()], fontsize=9)
        ax4.set_xlabel('Stability Score', fontsize=12)
        ax4.set_ylabel('Uczestnik', fontsize=12)
        ax4.set_title('Ranking według stabilności wyniku\n' +
                     'Wyżej = bardziej stabilny wynik',
                     fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Analiza stabilności wyniku końcowego - Finaliści',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return rank_changes_df, stability_df
    
    def visualize_overlapping_intervals(self, confidence: float = 0.95,
                                       save_path: str = None):
    def visualize_overlapping_intervals(self, confidence: float = 0.95,
                                       save_path: str = None):
        """
        Wizualizacja nakładających się przedziałów ufności
        Pokazuje, którzy uczestnicy mają statystycznie nierozróżnialne wyniki
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Oblicz przedziały ufności
        alpha = (1 - confidence) / 2
        ci_data = []
        
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = self.bootstrap_results[nr]
            
            ci_low = np.percentile(scores, alpha * 100)
            ci_high = np.percentile(scores, (1 - alpha) * 100)
            
            ci_data.append({
                'rank': row['rank'],
                'participant': f"{row['rank']}. {row['nazwisko']}",
                'actual_score': row['final_score'],
                'ci_low': ci_low,
                'ci_high': ci_high
            })
        
        df = pd.DataFrame(ci_data)
        
        # Wizualizacja
        fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.35)))
        
        y_pos = range(len(df))
        
        # Dla każdego uczestnika narysuj przedział ufności
        for i, row in df.iterrows():
            # Linia przedziału
            ax.plot([row['ci_low'], row['ci_high']], [i, i], 
                   'o-', linewidth=3, markersize=6, alpha=0.7)
            
            # Punkt rzeczywistego wyniku
            ax.plot(row['actual_score'], i, 'ro', markersize=10, zorder=10)
        
        # Zaznacz overlapping pairs
        overlaps = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Czy przedziały się nakładają?
                if (df.iloc[i]['ci_low'] <= df.iloc[j]['ci_high'] and
                    df.iloc[j]['ci_low'] <= df.iloc[i]['ci_high']):
                    overlaps.append((i, j))
                    
                    # Narysuj connection
                    ax.plot([df.iloc[i]['actual_score'], df.iloc[j]['actual_score']],
                           [i, j], 'gray', linewidth=0.5, alpha=0.3, zorder=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['participant'], fontsize=9)
        ax.set_xlabel('Wynik końcowy', fontsize=12)
        ax.set_ylabel('Uczestnik (pozycja. nazwisko)', fontsize=12)
        ax.set_title(f'Nakładające się przedziały ufności ({confidence*100:.0f}%) - Finaliści\n' +
                    f'Czerwony punkt = rzeczywisty wynik, Szare linie = statystycznie nierozróżnialne pary\n' +
                    f'Znaleziono {len(overlaps)} par z nakładającymi się przedziałami',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Zwróć informacje o parach
        overlap_info = []
        for i, j in overlaps:
            overlap_info.append({
                'participant1': df.iloc[i]['participant'],
                'rank1': df.iloc[i]['rank'],
                'participant2': df.iloc[j]['participant'],
                'rank2': df.iloc[j]['rank'],
                'rank_diff': abs(df.iloc[i]['rank'] - df.iloc[j]['rank'])
            })
        
        return pd.DataFrame(overlap_info)
    
    def calculate_stability_scores(self) -> pd.DataFrame:
        """
        Oblicza stability score dla każdego uczestnika
        
        Stability score = normalized measure combining:
        - SD of scores (lower is better)
        - Width of 95% CI (lower is better)
        - Range of ranks (lower is better)
        
        Returns 0-100 score (100 = most stable)
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        n_iterations = len(list(self.bootstrap_results.values())[0])
        
        stability_data = []
        
        print("Obliczam stability scores...")
        
        # Najpierw oblicz rankingi dla wszystkich iteracji
        all_ranks = {nr: [] for nr in actual_scores['Nr']}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")
            
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['Nr']
                iter_scores.append({
                    'nr': nr,
                    'score': self.bootstrap_results[nr][iteration]
                })
            
            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['rank'] = range(1, len(iter_df) + 1)
            
            for _, row in iter_df.iterrows():
                all_ranks[row['nr']].append(row['rank'])
        
        # Oblicz metryki
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = self.bootstrap_results[nr]
            ranks = all_ranks[nr]
            
            std_score = np.std(scores)
            ci_width = np.percentile(scores, 97.5) - np.percentile(scores, 2.5)
            rank_range = np.percentile(ranks, 95) - np.percentile(ranks, 5)
            
            stability_data.append({
                'Nr': nr,
                'participant': f"{row['imię']} {row['nazwisko']}",
                'rank': row['rank'],
                'actual_score': row['final_score'],
                'std_score': std_score,
                'ci_width': ci_width,
                'rank_range': rank_range
            })
        
        df = pd.DataFrame(stability_data)
        
        # Normalizuj metryki (odwróć, żeby wyższa wartość = bardziej stabilny)
        df['std_normalized'] = 1 - (df['std_score'] - df['std_score'].min()) / \
                                   (df['std_score'].max() - df['std_score'].min())
        df['ci_normalized'] = 1 - (df['ci_width'] - df['ci_width'].min()) / \
                                  (df['ci_width'].max() - df['ci_width'].min())
        df['rank_normalized'] = 1 - (df['rank_range'] - df['rank_range'].min()) / \
                                    (df['rank_range'].max() - df['rank_range'].min())
        
        # Stability score = średnia znormalizowanych metryk * 100
        df['stability_score'] = (df['std_normalized'] + df['ci_normalized'] + 
                                 df['rank_normalized']) / 3 * 100
        
        return df.sort_values('rank')
    
def main():
    """Przykład użycia zaawansowanych wizualizacji"""
    
    from chopin_final_score_stability import FinalScoreStabilityAnalyzer
    
    print("="*70)
    print("ZAAWANSOWANA ANALIZA STABILNOŚCI")
    print("="*70 + "\n")
    
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    # Analiza podstawowa
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)
    
    # Zaawansowane wizualizacje
    adv_viz = AdvancedStabilityVisualizer(analyzer, bootstrap_results)
    
    import os
    output_dir = 'advanced_stability'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1/2 Analiza stabilności (rank changes + stability scores połączone)...")
    rank_changes_df, stability_df = adv_viz.visualize_combined_stability_analysis(
        save_path=f'{output_dir}/combined_stability_analysis.png'
    )
    
    print("\n2/2 Nakładające się przedziały ufności...")
    overlaps_df = adv_viz.visualize_overlapping_intervals(
        confidence=0.95,
        save_path=f'{output_dir}/overlapping_intervals.png'
    )
    
    # Zapisz tabelę z wynikami
    print("\nZapisuję tabele z wynikami...")
    
    rank_changes_df.to_csv(f'{output_dir}/rank_changes.csv', index=False)
    overlaps_df.to_csv(f'{output_dir}/overlapping_pairs.csv', index=False)
    stability_df.to_csv(f'{output_dir}/stability_scores.csv', index=False)
    
    print(f"\n{'='*70}")
    print("ZAKOŃCZONO!")
    print(f"Wszystkie wyniki w: {output_dir}/")
    print(f"{'='*70}\n")
    
    # Wyświetl kilka interesujących statystyk
    print("\nINTERESUJĄCE STATYSTYKI:")
    print("-" * 70)
    
    print(f"\nNajbardziej stabilny uczestnik:")
    most_stable = stability_df.nlargest(1, 'stability_score').iloc[0]
    print(f"  {most_stable['rank']}. {most_stable['participant']}")
    print(f"  Stability Score: {most_stable['stability_score']:.1f}/100")
    
    print(f"\nNajmniej stabilny uczestnik:")
    least_stable = stability_df.nsmallest(1, 'stability_score').iloc[0]
    print(f"  {least_stable['rank']}. {least_stable['participant']}")
    print(f"  Stability Score: {least_stable['stability_score']:.1f}/100")
    
    print(f"\nLiczba par z nakładającymi się przedziałami ufności: {len(overlaps_df)}")
    
    if not overlaps_df.empty:
        max_diff = overlaps_df['rank_diff'].max()
        print(f"Największa różnica w rankingu między nakładającymi się: {max_diff} pozycji")
    
    print("\nSzczegóły wszystkich finalistów:")
    print("-" * 70)
    print(f"{'Ranga':<6} {'Uczestnik':<30} {'Stability':<10} {'Rank Range':<12}")
    print("-" * 70)
    for _, row in stability_df.iterrows():
        rank_info = rank_changes_df[rank_changes_df['actual_rank'] == row['rank']].iloc[0]
        print(f"{row['rank']:<6} {row['participant']:<30} "
              f"{row['stability_score']:<10.1f} {rank_info['rank_range']:<12.1f}")


if __name__ == "__main__":
    main()
