"""
Kompleksowa analiza stabilności wyniku końcowego konkursu Chopinowskiego
Łączy wszystkie analizy w jednym miejscu

Bootstrap resampling wszystkich 4 etapów + wizualizacje podstawowe i zaawansowane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChopinScoreCalculator:
    """Kalkulator wyników z korekcjami outlierów"""
    
    @staticmethod
    def calculate_corrected_average(scores: np.ndarray, threshold: float = 3.0) -> float:
        """Oblicza średnią z korekcją outlierów"""
        if len(scores) == 0:
            return 0.0
        
        avg = np.mean(scores)
        corrected = scores.copy()
        for i, score in enumerate(scores):
            if score > avg + threshold:
                corrected[i] = avg + threshold
            elif score < avg - threshold:
                corrected[i] = avg - threshold
        
        final_avg = np.mean(corrected)
        return round(final_avg, 2)


class StabilityAnalyzer:
    """Główny analizator stabilności"""
    
    def __init__(self, stage_files: Dict[str, str]):
        self.stage_files = stage_files
        self.stages_data = {}
        self.judge_columns = []
        self.thresholds = {
            'stage1': 3.0,
            'stage2': 2.0,
            'stage3': 2.0,
            'final': 2.0
        }
        self.weights = {
            'stage1': 0.10,
            'stage2': 0.20,
            'stage3': 0.35,
            'final': 0.35
        }
        
        self._load_data()
    
    def _load_data(self):
        """Wczytuje dane ze wszystkich etapów"""
        print("Wczytuję dane z etapów...")
        
        for stage, filepath in self.stage_files.items():
            df = pd.read_csv(filepath)
            
            if not self.judge_columns:
                self.judge_columns = [col for col in df.columns 
                                     if col not in ['Nr', 'imię', 'nazwisko']]
            
            for judge in self.judge_columns:
                df[judge] = pd.to_numeric(df[judge], errors='coerce')
            
            self.stages_data[stage] = df
            print(f"  {stage}: {len(df)} uczestników, {len(self.judge_columns)} sędziów")
    
    def calculate_stage_score_for_participant(self, stage: str, participant_nr: int, 
                                              judge_subset: List[str] = None) -> float:
        """Oblicza wynik uczestnika w danym etapie"""
        df = self.stages_data[stage]
        participant = df[df['Nr'] == participant_nr]
        
        if participant.empty:
            return 0.0
        
        judges = judge_subset if judge_subset else self.judge_columns
        
        scores = []
        for judge in judges:
            score = participant[judge].iloc[0]
            if pd.notna(score):
                scores.append(score)
        
        if not scores:
            return 0.0
        
        threshold = self.thresholds[stage]
        calculator = ChopinScoreCalculator()
        return calculator.calculate_corrected_average(np.array(scores), threshold)
    
    def calculate_final_weighted_score(self, participant_nr: int, 
                                       judge_subsets: Dict[str, List[str]] = None) -> float:
        """Oblicza końcowy ważony wynik uczestnika"""
        if judge_subsets is None:
            judge_subsets = {stage: self.judge_columns for stage in self.stage_files.keys()}
        
        available_stages = []
        for stage in self.stage_files.keys():
            if participant_nr in self.stages_data[stage]['Nr'].values:
                available_stages.append(stage)
        
        if not available_stages:
            return 0.0
        
        stage_scores = {}
        for stage in available_stages:
            stage_scores[stage] = self.calculate_stage_score_for_participant(
                stage, participant_nr, judge_subsets.get(stage)
            )
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for stage in available_stages:
            weight = self.weights[stage]
            weighted_sum += stage_scores[stage] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight * sum(self.weights.values())
    
    def bootstrap_final_scores(self, n_iterations: int = 10000) -> Dict[int, List[float]]:
        """Bootstrap resampling - losowanie sędziów w każdym etapie"""
        print(f"\nRozpoczynanie bootstrapu: {n_iterations} iteracji...")
        
        final_participants = self.stages_data['final']['Nr'].values
        bootstrap_results = {nr: [] for nr in final_participants}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteracja {iteration + 1}/{n_iterations}")
            
            judge_subsets = {}
            for stage in self.stage_files.keys():
                n_judges = len(self.judge_columns)
                judge_subsets[stage] = list(np.random.choice(
                    self.judge_columns, size=n_judges, replace=True
                ))
            
            for participant_nr in final_participants:
                final_score = self.calculate_final_weighted_score(
                    participant_nr, judge_subsets
                )
                bootstrap_results[participant_nr].append(final_score)
        
        print("Bootstrap zakończony!")
        return bootstrap_results
    
    def get_actual_final_scores(self) -> pd.DataFrame:
        """Oblicza rzeczywiste wyniki końcowe"""
        final_df = self.stages_data['final'][['Nr', 'imię', 'nazwisko']].copy()
        
        final_scores = []
        for _, row in final_df.iterrows():
            score = self.calculate_final_weighted_score(row['Nr'])
            final_scores.append(score)
        
        final_df['final_score'] = final_scores
        final_df = final_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        final_df['rank'] = range(1, len(final_df) + 1)
        
        return final_df
    
    def calculate_rank_changes_data(self, bootstrap_results: Dict[int, List[float]]) -> pd.DataFrame:
        """Oblicza dane o potencjalnych zmianach w rankingu"""
        actual_scores = self.get_actual_final_scores()
        n_iterations = len(list(bootstrap_results.values())[0])
        all_ranks = {nr: [] for nr in actual_scores['Nr']}
        
        print("Obliczam możliwe zmiany w rankingu...")
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")
            
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['Nr']
                iter_scores.append({
                    'nr': nr,
                    'score': bootstrap_results[nr][iteration]
                })
            
            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['rank'] = range(1, len(iter_df) + 1)
            
            for _, row in iter_df.iterrows():
                all_ranks[row['nr']].append(row['rank'])
        
        rank_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            ranks = all_ranks[nr]
            
            rank_data.append({
                'participant': f"{row['imię']} {row['nazwisko']}",
                'actual_rank': row['rank'],
                'best_rank': np.percentile(ranks, 5),
                'worst_rank': np.percentile(ranks, 95),
                'median_rank': np.median(ranks),
                'rank_range': np.percentile(ranks, 95) - np.percentile(ranks, 5)
            })
        
        df = pd.DataFrame(rank_data)
        df = df.sort_values('actual_rank')
        
        return df
    
    def calculate_stability_scores(self, bootstrap_results: Dict[int, List[float]]) -> pd.DataFrame:
        """Oblicza stability score dla każdego uczestnika"""
        actual_scores = self.get_actual_final_scores()
        n_iterations = len(list(bootstrap_results.values())[0])
        
        stability_data = []
        
        print("Obliczam stability scores...")
        
        all_ranks = {nr: [] for nr in actual_scores['Nr']}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")
            
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['Nr']
                iter_scores.append({
                    'nr': nr,
                    'score': bootstrap_results[nr][iteration]
                })
            
            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['rank'] = range(1, len(iter_df) + 1)
            
            for _, row in iter_df.iterrows():
                all_ranks[row['nr']].append(row['rank'])
        
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = bootstrap_results[nr]
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
        
        # Normalizuj metryki
        df['std_normalized'] = 1 - (df['std_score'] - df['std_score'].min()) / \
                                   (df['std_score'].max() - df['std_score'].min())
        df['ci_normalized'] = 1 - (df['ci_width'] - df['ci_width'].min()) / \
                                  (df['ci_width'].max() - df['ci_width'].min())
        df['rank_normalized'] = 1 - (df['rank_range'] - df['rank_range'].min()) / \
                                    (df['rank_range'].max() - df['rank_range'].min())
        
        df['stability_score'] = (df['std_normalized'] + df['ci_normalized'] + 
                                 df['rank_normalized']) / 3 * 100
        
        return df.sort_values('rank')


class StabilityVisualizer:
    """Wizualizacje stabilności"""
    
    def __init__(self, analyzer: StabilityAnalyzer):
        self.analyzer = analyzer
    
    def plot_violin_distributions(self, bootstrap_results: Dict[int, List[float]],
                                  save_path: str = None):
        """Violin plots rozkładów wyników"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        plot_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            name = f"{row['rank']}. {row['nazwisko']}"
            actual_score = row['final_score']
            
            for score in bootstrap_results[nr]:
                plot_data.append({
                    'participant': name,
                    'rank': row['rank'],
                    'score': score,
                    'actual_score': actual_score
                })
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(16, max(10, len(actual_scores) * 0.4)))
        
        order = [f"{r}. {n}" for r, n in 
                zip(actual_scores['rank'], actual_scores['nazwisko'])]
        
        sns.violinplot(data=df, y='participant', x='score', order=order, 
                      inner='quartile', ax=ax, palette='Set2')
        
        for i, (_, row) in enumerate(actual_scores.iterrows()):
            ax.plot(row['final_score'], i, 'ro', markersize=8, 
                   label='Rzeczywisty wynik' if i == 0 else '')
        
        ax.set_xlabel('Wynik końcowy', fontsize=12)
        ax.set_ylabel('Uczestnik (ranga. nazwisko)', fontsize=12)
        ax.set_title('Stabilność wyniku końcowego - Finaliści\n' +
                    'Rozkład możliwych wyników z bootstrapu',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        plt.close()
    
    def plot_confidence_intervals(self, bootstrap_results: Dict[int, List[float]],
                                  confidence: float = 0.95, save_path: str = None):
        """Przedziały ufności dla wyników"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        ci_data = []
        alpha = (1 - confidence) / 2
        
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = bootstrap_results[nr]
            
            ci_low = np.percentile(scores, alpha * 100)
            ci_high = np.percentile(scores, (1 - alpha) * 100)
            ci_width = ci_high - ci_low
            std = np.std(scores)
            
            ci_data.append({
                'rank': row['rank'],
                'participant': f"{row['imię']} {row['nazwisko']}",
                'actual_score': row['final_score'],
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_width': ci_width,
                'std': std
            })
        
        ci_df = pd.DataFrame(ci_data)
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        
        # Przedziały ufności
        ax1 = axes[0]
        y_pos = range(len(ci_df))
        # ax1.errorbar(ci_df['actual_score'], y_pos,
        #             xerr=[ci_df['actual_score'] - ci_df['ci_low'],
        #                   ci_df['ci_high'] - ci_df['actual_score']],
        #             fmt='o', markersize=6, capsize=5, capthick=2,
        #             alpha=0.7, label=f'{confidence*100:.0f}% przedział ufności')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}" 
                            for _, row in ci_df.iterrows()], fontsize=8)
        ax1.set_xlabel('Wynik końcowy', fontsize=12)
        ax1.set_ylabel('Uczestnik', fontsize=12)
        ax1.set_title(f'Przedziały ufności ({confidence*100:.0f}%) dla wyniku końcowego',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Szerokość przedziałów
        ax2 = axes[1]
        df_sorted = ci_df.sort_values('ci_width', ascending=True)
        colors2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_sorted)))
        
        bars = ax2.barh(range(len(df_sorted)), df_sorted['ci_width'],
                       color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}" 
                            for _, row in df_sorted.iterrows()], fontsize=8)
        ax2.set_xlabel('Szerokość przedziału ufności (punkty)', fontsize=12)
        ax2.set_ylabel('Uczestnik', fontsize=12)
        ax2.set_title('Najbardziej niestabilne wyniki',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        plt.close()
    
    def plot_combined_stability_analysis(self, bootstrap_results: Dict[int, List[float]],
                                         save_path: str = None):
        """Połączona analiza: rank changes + stability scores"""
        rank_changes_df = self.analyzer.calculate_rank_changes_data(bootstrap_results)
        stability_df = self.analyzer.calculate_stability_scores(bootstrap_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # [0,0] Zakres pozycji
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
        ax1.set_ylabel('Uczestnik', fontsize=12)
        ax1.set_title('Możliwe zmiany pozycji w rankingu\n' +
                     '90% przedział możliwych pozycji',
                     fontsize=13, fontweight='bold')
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # [0,1] Rozpiętość pozycji
        ax2 = axes[0, 1]
        df_sorted = rank_changes_df.sort_values('rank_range', ascending=True)
        colors2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_sorted)))
        
        ax2.barh(range(len(df_sorted)), df_sorted['rank_range'],
                color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels([f"{row['actual_rank']}. {row['participant'][:25]}" 
                            for _, row in df_sorted.iterrows()], fontsize=9)
        ax2.set_xlabel('Rozpiętość możliwych pozycji', fontsize=12)
        ax2.set_ylabel('Uczestnik', fontsize=12)
        ax2.set_title('Niestabilność pozycji w rankingu',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # [1,0] Stability score vs pozycja
        ax3 = axes[1, 0]
        scatter = ax3.scatter(stability_df['rank'], stability_df['stability_score'],
                            s=120, alpha=0.6, c=stability_df['stability_score'],
                            cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        for _, row in stability_df.iterrows():
            ax3.annotate(f"{row['rank']}", 
                        (row['rank'], row['stability_score']),
                        fontsize=9, alpha=0.8)
        
        ax3.set_xlabel('Pozycja w rankingu', fontsize=12)
        ax3.set_ylabel('Stability Score (0-100)', fontsize=12)
        ax3.set_title('Stability Score vs Pozycja\n100 = najbardziej stabilny',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Stability Score')
        
        # [1,1] Ranking według stabilności
        ax4 = axes[1, 1]
        top_stable = stability_df.sort_values('stability_score', ascending=True)
        colors4 = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_stable)))
        
        ax4.barh(range(len(top_stable)), top_stable['stability_score'],
                color=colors4, alpha=0.7)
        ax4.set_yticks(range(len(top_stable)))
        ax4.set_yticklabels([f"{row['rank']}. {row['participant'][:25]}" 
                            for _, row in top_stable.iterrows()], fontsize=9)
        ax4.set_xlabel('Stability Score', fontsize=12)
        ax4.set_ylabel('Uczestnik', fontsize=12)
        ax4.set_title('Ranking według stabilności wyniku',
                     fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Analiza stabilności wyniku końcowego - Finaliści',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        plt.close()
        
        return rank_changes_df, stability_df
    
    def plot_overlapping_intervals(self, bootstrap_results: Dict[int, List[float]],
                                   confidence: float = 0.95, save_path: str = None):
        """Nakładające się przedziały ufności"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        alpha = (1 - confidence) / 2
        ci_data = []
        
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = bootstrap_results[nr]
            
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
        
        fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.35)))
        
        y_pos = range(len(df))
        
        for i, row in df.iterrows():
            ax.plot([row['ci_low'], row['ci_high']], [i, i], 
                   'o-', linewidth=3, markersize=6, alpha=0.7)
            ax.plot(row['actual_score'], i, 'ro', markersize=10, zorder=10)
        
        overlaps = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if (df.iloc[i]['ci_low'] <= df.iloc[j]['ci_high'] and
                    df.iloc[j]['ci_low'] <= df.iloc[i]['ci_high']):
                    overlaps.append((i, j))
                    ax.plot([df.iloc[i]['actual_score'], df.iloc[j]['actual_score']],
                           [i, j], 'gray', linewidth=0.5, alpha=0.3, zorder=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['participant'], fontsize=9)
        ax.set_xlabel('Wynik końcowy', fontsize=12)
        ax.set_ylabel('Uczestnik', fontsize=12)
        ax.set_title(f'Nakładające się przedziały ufności ({confidence*100:.0f}%) - Finaliści\n' +
                    f'Znaleziono {len(overlaps)} par z nakładającymi się przedziałami',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        plt.close()
        
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


def main():
    """Główna funkcja - kompleksowa analiza stabilności"""
    
    print("="*70)
    print("KOMPLEKSOWA ANALIZA STABILNOŚCI WYNIKU KOŃCOWEGO")
    print("Konkurs Chopinowski - Bootstrap Resampling")
    print("="*70 + "\n")
    
    # Ścieżki do plików
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    # Inicjalizacja
    analyzer = StabilityAnalyzer(stage_files)
    
    # Bootstrap (10000 iteracji)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)
    
    # Wizualizacje
    visualizer = StabilityVisualizer(analyzer)
    
    output_dir = 'stability_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENEROWANIE WIZUALIZACJI")
    print(f"{'='*70}\n")
    
    print("1/4 Rozkłady wyników (violin plots)...")
    visualizer.plot_violin_distributions(
        bootstrap_results,
        save_path=f'{output_dir}/score_distributions.png'
    )
    
    print("\n2/4 Przedziały ufności...")
    visualizer.plot_confidence_intervals(
        bootstrap_results,
        confidence=0.95,
        save_path=f'{output_dir}/confidence_intervals.png'
    )
    
    print("\n3/4 Połączona analiza stabilności...")
    rank_changes_df, stability_df = visualizer.plot_combined_stability_analysis(
        bootstrap_results,
        save_path=f'{output_dir}/combined_stability_analysis.png'
    )
    
    print("\n4/4 Nakładające się przedziały...")
    overlaps_df = visualizer.plot_overlapping_intervals(
        bootstrap_results,
        confidence=0.95,
        save_path=f'{output_dir}/overlapping_intervals.png'
    )
    
    # Zapisz dane CSV
    print("\nZapisuję dane CSV...")
    actual_scores = analyzer.get_actual_final_scores()
    actual_scores.to_csv(f'{output_dir}/final_scores.csv', index=False)
    rank_changes_df.to_csv(f'{output_dir}/rank_changes.csv', index=False)
    stability_df.to_csv(f'{output_dir}/stability_scores.csv', index=False)
    overlaps_df.to_csv(f'{output_dir}/overlapping_pairs.csv', index=False)
    
    print(f"\n{'='*70}")
    print("ZAKOŃCZONO!")
    print(f"Wszystkie wyniki w: {output_dir}/")
    print(f"{'='*70}\n")
    
    # Statystyki
    print("STATYSTYKI STABILNOŚCI:")
    print("-" * 70)
    
    print(f"\nNajbardziej stabilny uczestnik:")
    most_stable = stability_df.nlargest(1, 'stability_score').iloc[0]
    print(f"  {most_stable['rank']}. {most_stable['participant']}")
    print(f"  Stability Score: {most_stable['stability_score']:.1f}/100")
    
    print(f"\nNajmniej stabilny uczestnik:")
    least_stable = stability_df.nsmallest(1, 'stability_score').iloc[0]
    print(f"  {least_stable['rank']}. {least_stable['participant']}")
    print(f"  Stability Score: {least_stable['stability_score']:.1f}/100")
    
    print(f"\nLiczba par z nakładającymi się przedziałami: {len(overlaps_df)}")
    
    if not overlaps_df.empty:
        max_diff = overlaps_df['rank_diff'].max()
        print(f"Największa różnica w rankingu między nakładającymi się: {max_diff} pozycji")
    
    print("\nSzczegóły wszystkich finalistów:")
    print("-" * 70)
    print(f"{'Ranga':<6} {'Uczestnik':<30} {'Wynik':<8} {'Stability':<10} {'Rank Range':<12}")
    print("-" * 70)
    
    for _, row in stability_df.iterrows():
        rank_info = rank_changes_df[rank_changes_df['actual_rank'] == row['rank']].iloc[0]
        print(f"{row['rank']:<6} {row['participant']:<30} "
              f"{row['actual_score']:<8.2f} {row['stability_score']:<10.1f} "
              f"{rank_info['rank_range']:<12.1f}")


if __name__ == "__main__":
    main()
