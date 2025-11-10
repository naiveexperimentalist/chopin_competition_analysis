"""Final-score stability via jury bootstrapping.

Bootstraps (with replacement) judge-score vectors within each stage,
recomputes stage and cumulative results, and reports distributions,
confidence intervals, and rank-position probabilities for finalists."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChopinScoreCalculator:

    @staticmethod
    def calculate_corrected_average(scores: np.ndarray, threshold: float = 3.0) -> float:
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


class FinalScoreStabilityAnalyzer:

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
        print("Loading stage data…")
        
        for stage, filepath in self.stage_files.items():
            df = pd.read_csv(filepath)
            
            if not self.judge_columns:
                self.judge_columns = [col for col in df.columns 
                                     if col not in ['number', 'firstname', 'lastname']]
            
            for judge in self.judge_columns:
                df[judge] = pd.to_numeric(df[judge], errors='coerce')
            
            self.stages_data[stage] = df
            print(f"  {stage}: {len(df)} contestants, {len(self.judge_columns)} judges")
    
    def calculate_stage_score_for_participant(self, stage: str, participant_nr: int, 
                                              judge_subset: List[str] = None) -> float:
        df = self.stages_data[stage]
        participant = df[df['number'] == participant_nr]
        
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
        if judge_subsets is None:
            judge_subsets = {stage: self.judge_columns for stage in self.stage_files.keys()}
        
        available_stages = []
        for stage in self.stage_files.keys():
            if participant_nr in self.stages_data[stage]['number'].values:
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
        print(f"\nStarting bootstrapping: {n_iterations} iterations…")
        
        final_participants = self.stages_data['final']['number'].values
        
        bootstrap_results = {nr: [] for nr in final_participants}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}")
            
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
        
        print("Bootstrap completed!")
        return bootstrap_results
    
    def get_actual_final_scores(self) -> pd.DataFrame:
        final_df = self.stages_data['final'][['number', 'firstname', 'lastname']].copy()
        
        final_scores = []
        for _, row in final_df.iterrows():
            score = self.calculate_final_weighted_score(row['number'])
            final_scores.append(score)
        
        final_df['final_score'] = final_scores
        final_df = final_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        final_df['rank'] = range(1, len(final_df) + 1)
        
        return final_df


class FinalScoreStabilityVisualizer:

    def __init__(self, analyzer: FinalScoreStabilityAnalyzer):
        self.analyzer = analyzer

    def visualize_score_distributions(self, bootstrap_results: Dict[int, List[float]],
                                      save_path: str = None):
        top_participants = self.analyzer.get_actual_final_scores()

        # Przygotuj dane do violin plot
        plot_data = []
        for _, row in top_participants.iterrows():
            nr = row['number']
            name = f"{row['rank']}. {row['lastname']}"
            actual_score = row['final_score']

            for score in bootstrap_results[nr]:
                plot_data.append({
                    'participant': name,
                    'rank': row['rank'],
                    'score': score,
                    'actual_score': actual_score
                })

        df = pd.DataFrame(plot_data)

        # Rysuj
        fig, ax = plt.subplots(figsize=(16, max(10, len(top_participants) * 0.4)))

        # Sortuj po ranking
        order = [f"{r}. {n}" for r, n in
                 zip(top_participants['rank'], top_participants['lastname'])]

        # Violin plot
        sns.violinplot(data=df, y='participant', x='score', order=order,
                       inner='quartile', ax=ax, palette='Set2')

        for i, (_, row) in enumerate(top_participants.iterrows()):
            ax.plot(row['final_score'], i, 'ro', markersize=8,
                    label='Actual score' if i == 0 else '')

        ax.set_xlabel('Final score', fontsize=12)
        ax.set_ylabel('Contestant (rank, surname)', fontsize=12)
        ax.set_title('Final score stability — finalists\n'
                     'Distributions of possible scores from bootstrap '
                     '(resampling judges across all stages)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_confidence_intervals(self, bootstrap_results: Dict[int, List[float]],
                                       confidence: float = 0.95, save_path: str = None):
        actual_scores = self.analyzer.get_actual_final_scores()

        ci_data = []
        alpha = (1 - confidence) / 2

        for _, row in actual_scores.iterrows():
            nr = row['number']
            scores = bootstrap_results[nr]

            ci_low = np.percentile(scores, alpha * 100)
            ci_high = np.percentile(scores, (1 - alpha) * 100)
            ci_width = ci_high - ci_low
            std = np.std(scores)

            ci_data.append({
                'rank': row['rank'],
                'participant': f"{row['firstname']} {row['lastname']}",
                'actual_score': row['final_score'],
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_width': ci_width,
                'std': std
            })

        ci_df = pd.DataFrame(ci_data)

        # Rysuj
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))

        ax1 = axes[0]

        y_pos = range(len(ci_df))
        ax1.errorbar(ci_df['actual_score'], y_pos,
                     xerr=[ci_df['actual_score'] - ci_df['ci_low'],
                           ci_df['ci_high'] - ci_df['actual_score']],
                     fmt='o', markersize=6, capsize=5, capthick=2,
                     alpha=0.7, label=f'{confidence * 100:.0f}% confidence interval')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}"
                             for _, row in ci_df.iterrows()], fontsize=8)
        ax1.set_xlabel('Final score', fontsize=12)
        ax1.set_ylabel('Contestant', fontsize=12)
        ax1.set_title(f'Confidence intervals ({confidence * 100:.0f}%) for final score\n'
                      'Red dot = actual score; lines = possible range',
                      fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        ax2 = axes[1]

        top_unstable = ci_df.nlargest(30, 'ci_width')
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_unstable)))

        bars = ax2.barh(range(len(top_unstable)), top_unstable['ci_width'],
                        color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_unstable)))
        ax2.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}"
                             for _, row in top_unstable.iterrows()], fontsize=8)
        ax2.set_xlabel('Confidence-interval width (points)', fontsize=12)
        ax2.set_ylabel('Contestant', fontsize=12)
        ax2.set_title('Result uncertainty\n'
                      '(wider interval implies greater uncertainty)',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_ranking_stability_matrix(self, bootstrap_results: Dict[int, List[float]],
                                           save_path: str = None):
        top_participants = self.analyzer.get_actual_final_scores()

        n_iterations = len(list(bootstrap_results.values())[0])
        n_finalists = len(top_participants)
        ranking_matrix = np.zeros((n_finalists, n_finalists))

        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}")

            iter_scores = []
            for _, row in top_participants.iterrows():
                nr = row['number']
                iter_scores.append({
                    'nr': nr,
                    'score': bootstrap_results[nr][iteration]
                })

            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['bootstrap_rank'] = range(1, len(iter_df) + 1)

            for idx, (_, row) in enumerate(top_participants.iterrows()):
                nr = row['number']
                bootstrap_rank = iter_df[iter_df['nr'] == nr]['bootstrap_rank'].iloc[0]
                ranking_matrix[idx, bootstrap_rank - 1] += 1

        ranking_matrix = ranking_matrix / n_iterations * 100

        fig, ax = plt.subplots(figsize=(16, 12))

        participant_labels = [f"{row['rank']}. {row['lastname']}"
                              for _, row in top_participants.iterrows()]

        sns.heatmap(ranking_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=range(1, n_finalists + 1),
                    yticklabels=participant_labels,
                    cbar_kws={'label': "Probability (%)"},
                    ax=ax)

        ax.set_xlabel('Possible rank position', fontsize=12)
        ax.set_ylabel('Contestant (actual position, surname)', fontsize=12)
        ax.set_title('Ranking stability matrix — finalists\n'
                     'Probability of occupying each position (bootstrap)',
                     fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_score_vs_uncertainty(self, bootstrap_results: Dict[int, List[float]],
                                       save_path: str = None):
        actual_scores = self.analyzer.get_actual_final_scores()

        scatter_data = []
        for _, row in actual_scores.iterrows():
            nr = row['number']
            scores = bootstrap_results[nr]
            std = np.std(scores)

            scatter_data.append({
                'participant': f"{row['firstname']} {row['lastname']}",
                'rank': row['rank'],
                'actual_score': row['final_score'],
                'std': std
            })

        df = pd.DataFrame(scatter_data)

        # Rysuj
        fig, ax = plt.subplots(figsize=(14, 10))

        # Scatter plot z colour wg ranking
        scatter = ax.scatter(df['actual_score'], df['std'],
                             s=150, alpha=0.6, c=df['rank'],
                             cmap='viridis_r', edgecolors='black', linewidth=0.5)

        top_13 = df[df['rank'] <= 13]
        most_unstable = df.nlargest(5, 'std')
        to_annotate = pd.concat([top_13, most_unstable]).drop_duplicates()

        for _, row in to_annotate.iterrows():
            ax.annotate(f"{row['rank']}. {row['participant'].split()[-1]}",
                        (row['actual_score'], row['std']),
                        fontsize=8, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Actual final score', fontsize=12)
        ax.set_ylabel('Standard deviation (uncertainty)', fontsize=12)
        ax.set_title('Final score vs uncertainty\n'
                     'Colour = rank position (darker = higher)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Rank position')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_full_stability_report(self, bootstrap_results: Dict[int, List[float]],
                                     output_dir: str = 'final_score_stability'):
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("GENERATING STABILITY REPORT")
        print(f"{'=' * 60}\n")

        print("1/4 score distribution…")
        self.visualize_score_distributions(
            bootstrap_results,
            save_path=f'{output_dir}/34_score_distributions.png'
        )

        print("\n2/4 confidence intervals…")
        self.visualize_confidence_intervals(
            bootstrap_results, confidence=0.95,
            save_path=f'{output_dir}/35_confidence_intervals.png'
        )

        print("\n3/4 ranking stability matrix…")
        self.visualize_ranking_stability_matrix(
            bootstrap_results,
            save_path=f'{output_dir}/36_ranking_stability_matrix.png'
        )

        print("\n4/4 score vs uncertainty…")
        self.visualize_score_vs_uncertainty(
            bootstrap_results,
            save_path=f'{output_dir}/37_score_vs_uncertainty.png'
        )

        print(f"\n{'=' * 60}")
        print(f"DONE!")
        print(f"All saved in: {output_dir}/")
        print(f"{'=' * 60}\n")


def main():

    stage_files = {
        'stage1': "Chopin_2025_stage1_by_judge.csv",
        'stage2': "Chopin_2025_stage2_by_judge.csv",
        'stage3': "Chopin_2025_stage3_by_judge.csv",
        'final': "Chopin_2025_final_by_judge.csv"
    }
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    
    n_iterations = 10000
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=n_iterations)
    
    # Visualisation
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results, 
        output_dir='visualizations'
    )

if __name__ == "__main__":
    main()
