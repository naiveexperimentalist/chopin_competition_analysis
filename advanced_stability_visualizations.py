"""
Advanced visualizations of final score stability
- Potential ranking changes
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
    """Advanced stability visualizations"""

    def __init__(self, analyzer: FinalScoreStabilityAnalyzer,
                 bootstrap_results: dict):
        self.analyzer = analyzer
        self.bootstrap_results = bootstrap_results

    def calculate_rank_changes_data(self):
        """
        Calculates data about potential ranking changes
        """
        actual_scores = self.analyzer.get_actual_final_scores()

        # For each bootstrap iteration calculate rankings
        n_iterations = len(list(self.bootstrap_results.values())[0])
        all_ranks = {nr: [] for nr in actual_scores['number']}

        print("Calculating possible ranking changes...")

        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")

            # Get all participant scores in this iteration
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['number']
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

        # Prepare visualization data
        rank_data = []
        for _, row in actual_scores.iterrows():
            nr = row['number']
            ranks = all_ranks[nr]

            rank_data.append({
                'participant': f"{row['firstname']} {row['lastname']}",
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
        Combined visualization: rank changes + stability scores (4 charts in one)
        """
        # Calculate data
        rank_changes_df = self.calculate_rank_changes_data()
        stability_df = self.calculate_stability_scores()

        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # [0,0] Range of positions for each participant
        ax1 = axes[0, 0]
        y_pos = range(len(rank_changes_df))

        for i, row in rank_changes_df.iterrows():
            ax1.plot([row['best_rank'], row['worst_rank']],
                    [i, i], 'o-', linewidth=2, markersize=4, alpha=0.6)
            ax1.plot(row['actual_rank'], i, 'ro', markersize=8, zorder=10)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['actual_rank']}. {row['participant'][:25]}"
                             for _, row in rank_changes_df.iterrows()], fontsize=9)
        ax1.set_xlabel('Rank position', fontsize=12)
        ax1.set_ylabel('Contestant', fontsize=12)
        ax1.set_title('Possible rank changes\n' +
                     'Line = 90% interval of possible positions; red dot = actual rank.',
                     fontsize=13, fontweight='bold')
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # [0,1] Range of possible positions
        ax2 = axes[0, 1]
        df_sorted_by_range = rank_changes_df.sort_values('rank_range', ascending=True)
        colors2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_sorted_by_range)))

        bars = ax2.barh(range(len(df_sorted_by_range)),
                       df_sorted_by_range['rank_range'],
                       color=colors2, alpha=0.7)

        ax2.set_yticks(range(len(df_sorted_by_range)))
        ax2.set_yticklabels([f"{row['actual_rank']}. {row['participant'][:25]}"
                            for _, row in df_sorted_by_range.iterrows()], fontsize=9)
        ax2.set_xlabel('Range of possible ranks', fontsize=12)
        ax2.set_ylabel('Contestant', fontsize=12)
        ax2.set_title('Instability of rank position\nLargest possible rank change',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # [1,0] Stability score vs ranking position
        ax3 = axes[1, 0]

        scatter = ax3.scatter(stability_df['rank'], stability_df['stability_score'],
                            s=120, alpha=0.6, c=stability_df['stability_score'],
                            cmap='RdYlGn', edgecolors='black', linewidth=0.5)

        # Annotations for all (there are only 11)
        for _, row in stability_df.iterrows():
            ax3.annotate(f"{row['rank']}",
                        (row['rank'], row['stability_score']),
                        fontsize=9, alpha=0.8)

        ax3.set_xlabel('Rank position', fontsize=12)
        ax3.set_ylabel('Stability Score (0-100)', fontsize=12)
        ax3.set_title('Stability Score vs Rank\n' +
                     '100 = most stable, 0 = least stable',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Stability Score')

        # [1,1] Most stable participants
        ax4 = axes[1, 1]

        top_stable = stability_df.sort_values('stability_score', ascending=True)
        colors4 = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_stable)))

        bars4 = ax4.barh(range(len(top_stable)), top_stable['stability_score'],
                        color=colors4, alpha=0.7)
        ax4.set_yticks(range(len(top_stable)))
        ax4.set_yticklabels([f"{row['rank']}. {row['participant'][:25]}"
                            for _, row in top_stable.iterrows()], fontsize=9)
        ax4.set_xlabel('Stability Score', fontsize=12)
        ax4.set_ylabel('Contestant', fontsize=12)
        ax4.set_title('Contestants ranked by stability score\n' +
                     'Higher = more stable',
                     fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Stability analysis of final results — finalists',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        return rank_changes_df, stability_df

    def visualize_overlapping_intervals(self, confidence: float = 0.95,
                                       save_path: str = None):
        """
        Visualization of overlapping confidence intervals
        Shows which participants have statistically indistinguishable results
        """
        actual_scores = self.analyzer.get_actual_final_scores()

        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        ci_data = []

        for _, row in actual_scores.iterrows():
            nr = row['number']
            scores = self.bootstrap_results[nr]

            ci_low = np.percentile(scores, alpha * 100)
            ci_high = np.percentile(scores, (1 - alpha) * 100)

            ci_data.append({
                'participant': f"{row['firstname']} {row['lastname']}",
                'rank': row['rank'],
                'actual_score': row['final_score'],
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_width': ci_high - ci_low
            })

        df = pd.DataFrame(ci_data)
        df = df.sort_values('rank')

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        y_pos = range(len(df))

        # Confidence intervals
        for i, row in df.iterrows():
            ax.plot([row['ci_low'], row['ci_high']], [i, i],
                   'b-', linewidth=2, alpha=0.6)
            ax.plot([row['ci_low'], row['ci_high']], [i, i],
                   'bo', markersize=3)

        # Actual scores
        ax.plot(df['actual_score'], y_pos, 'ro', markersize=8,
               label='Actual score', zorder=10)

        # Find and visualize overlaps
        overlaps = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Check if intervals overlap
                if (df.iloc[i]['ci_low'] <= df.iloc[j]['ci_high'] and
                    df.iloc[j]['ci_low'] <= df.iloc[i]['ci_high']):
                    overlaps.append((i, j))

                    # Draw connection
                    ax.plot([df.iloc[i]['actual_score'], df.iloc[j]['actual_score']],
                           [i, j], 'gray', linewidth=0.5, alpha=0.3, zorder=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['participant'], fontsize=9)
        ax.set_xlabel('Final score', fontsize=12)
        ax.set_ylabel('Contestant', fontsize=12)
        ax.set_title(f'Overlapping confidence intervals ({confidence*100:.0f}%) — finalists\n' +
                    f'Red dot = actual score, Gray lines = statistically indistinguishable pairs\n' +
                    f'Found {len(overlaps)} pairs with overlapping confidence intervals',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        # Return information about pairs
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
        Calculates stability score for each participant

        Stability score = normalized measure combining:
        - SD of scores (lower is better)
        - Width of 95% CI (lower is better)
        - Range of ranks (lower is better)

        Returns 0-100 score (100 = most stable)
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        n_iterations = len(list(self.bootstrap_results.values())[0])

        stability_data = []

        print("Calculating stability scores...")

        # First calculate rankings for all iterations
        all_ranks = {nr: [] for nr in actual_scores['number']}

        for iteration in range(n_iterations):
            if (iteration + 1) % 2000 == 0:
                print(f"  {iteration + 1}/{n_iterations}")

            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['number']
                iter_scores.append({
                    'nr': nr,
                    'score': self.bootstrap_results[nr][iteration]
                })

            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['rank'] = range(1, len(iter_df) + 1)

            for _, row in iter_df.iterrows():
                all_ranks[row['nr']].append(row['rank'])

        # Calculate metrics
        for _, row in actual_scores.iterrows():
            nr = row['number']
            scores = self.bootstrap_results[nr]
            ranks = all_ranks[nr]

            std_score = np.std(scores)
            ci_width = np.percentile(scores, 97.5) - np.percentile(scores, 2.5)
            rank_range = np.percentile(ranks, 95) - np.percentile(ranks, 5)

            stability_data.append({
                'number': nr,
                'participant': f"{row['firstname']} {row['lastname']}",
                'rank': row['rank'],
                'actual_score': row['final_score'],
                'std_score': std_score,
                'ci_width': ci_width,
                'rank_range': rank_range
            })

        df = pd.DataFrame(stability_data)

        # Normalize metrics (invert so higher value = more stable)
        df['std_normalized'] = 1 - (df['std_score'] - df['std_score'].min()) / \
                                   (df['std_score'].max() - df['std_score'].min())
        df['ci_normalized'] = 1 - (df['ci_width'] - df['ci_width'].min()) / \
                                  (df['ci_width'].max() - df['ci_width'].min())
        df['rank_normalized'] = 1 - (df['rank_range'] - df['rank_range'].min()) / \
                                    (df['rank_range'].max() - df['rank_range'].min())

        # Stability score = average of normalized metrics * 100
        df['stability_score'] = (df['std_normalized'] + df['ci_normalized'] +
                                 df['rank_normalized']) / 3 * 100

        return df.sort_values('rank')

def main():
    """Example usage of advanced visualizations"""

    from chopin_final_score_stability import FinalScoreStabilityAnalyzer

    print("="*70)
    print("ADVANCED STABILITY ANALYSIS")
    print("="*70 + "\n")

    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }

    # Basic analysis
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=10000)

    # Advanced visualizations
    adv_viz = AdvancedStabilityVisualizer(analyzer, bootstrap_results)

    import os
    output_dir = 'advanced_stability'
    os.makedirs(output_dir, exist_ok=True)

    print("\n1/2 Stability analysis (rank changes + stability scores combined)...")
    rank_changes_df, stability_df = adv_viz.visualize_combined_stability_analysis(
        save_path=f'{output_dir}/combined_stability_analysis.png'
    )

    print("\n2/2 Overlapping confidence intervals...")
    overlaps_df = adv_viz.visualize_overlapping_intervals(
        confidence=0.95,
        save_path=f'{output_dir}/overlapping_intervals.png'
    )

    # Save result tables
    print("\nSaving result tables...")

    rank_changes_df.to_csv(f'{output_dir}/rank_changes.csv', index=False)
    overlaps_df.to_csv(f'{output_dir}/overlapping_pairs.csv', index=False)
    stability_df.to_csv(f'{output_dir}/stability_scores.csv', index=False)

    print(f"\n{'='*70}")
    print("COMPLETED!")
    print(f"All results in: {output_dir}/")
    print(f"{'='*70}\n")

    # Display some interesting statistics
    print("\nINTERESTING STATISTICS:")
    print("-" * 70)

    print(f"\nMost stable participant:")
    most_stable = stability_df.nlargest(1, 'stability_score').iloc[0]
    print(f"  {most_stable['rank']}. {most_stable['participant']}")
    print(f"  Stability Score: {most_stable['stability_score']:.1f}/100")

    print(f"\nLeast stable participant:")
    least_stable = stability_df.nsmallest(1, 'stability_score').iloc[0]
    print(f"  {least_stable['rank']}. {least_stable['participant']}")
    print(f"  Stability Score: {least_stable['stability_score']:.1f}/100")

    print(f"\nNumber of pairs with overlapping confidence intervals: {len(overlaps_df)}")

    if not overlaps_df.empty:
        max_diff = overlaps_df['rank_diff'].max()
        print(f"Largest rank difference between overlapping pairs: {max_diff} positions")

    print("\nDetails of all finalists:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Participant':<30} {'Stability':<10} {'Rank Range':<12}")
    print("-" * 70)
    for _, row in stability_df.iterrows():
        rank_info = rank_changes_df[rank_changes_df['actual_rank'] == row['rank']].iloc[0]
        print(f"{row['rank']:<6} {row['participant']:<30} "
              f"{row['stability_score']:<10.1f} {rank_info['rank_range']:<12.1f}")


if __name__ == "__main__":
    main()