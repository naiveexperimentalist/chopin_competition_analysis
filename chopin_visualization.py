"""\nGeneral visualizations for the Chopin Competition analysis.

Plots score distributions by judge, scale usage, Judge tendencies,
judge alliances & correlations, leave-one-judge-out effects, and more.\n"""

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

#  Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ChopinVisualization:

    def __init__(self, processor, analyzer=None):
        self.processor = processor
        self.analyzer = analyzer
        self.judge_columns = processor.judge_columns
        
    def setup_figure(self, figsize=(12, 8)):
        """Helper function to create a figure"""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def visualize_score_distribution(self, stage: str = 'stage1', save_path: str = None):
        if stage not in self.processor.stages_data:
            print(f"No data for stage: {stage}")
            return
        
        df = self.processor.stages_data[stage]
        
        #  Prepare data
        scores_data = []
        for judge in self.judge_columns:
            scores = pd.to_numeric(df[judge], errors='coerce').dropna()
            for score in scores:
                scores_data.append({'Judge': judge, 'Score': score})
        
        scores_df = pd.DataFrame(scores_data)
        
        #  Plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        #  Violin plot
        ax1 = axes[0]
        sns.violinplot(data=scores_df, x='Judge', y='Score', ax=ax1, inner='box')
        ax1.set_title(f'Distribution of scores by judge — {stage}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        #  Box plot with data points
        ax2 = axes[1]
        sns.boxplot(data=scores_df, x='Judge', y='Score', ax=ax2)
        sns.stripplot(data=scores_df, x='Judge', y='Score', ax=ax2,
                     alpha=0.4, size=3, color='red')
        ax2.set_title(f'Box-and-whisker plot with data points (jittered) — {stage}', fontsize=16, fontweight='bold')
        ax2.set_xlabel('', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_scale_usage_comparison(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object. Run advanced_analyzer first.")
            return
        
        scale_usage = self.analyzer.analyze_scale_usage()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        scale_usage_sorted = scale_usage.sort_values('overall_range', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scale_usage_sorted)))
        bars1 = ax1.barh(scale_usage_sorted['judge'], scale_usage_sorted['overall_range'], color=colors)
        ax1.set_xlabel('Score range (max − min)', fontsize=12)
        ax1.set_title('Range of assigned scores per judge', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, scale_usage_sorted['overall_range']):
            ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                    va='center', fontsize=9)
        
        #  2. Percentage of scale usage
        ax2 = axes[0, 1]
        scale_usage_sorted2 = scale_usage.sort_values('scale_coverage', ascending=True)
        colors2 = plt.cm.viridis(np.linspace(0.3, 0.9, len(scale_usage_sorted2)))
        bars2 = ax2.barh(scale_usage_sorted2['judge'], scale_usage_sorted2['scale_coverage'], color=colors2)
        ax2.set_xlabel('Percentage of scale usage (1–25)', fontsize=12)
        ax2.set_title('Percentage of unique score values used (of 25)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, scale_usage_sorted2['scale_coverage']):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                    va='center', fontsize=9)
        
        ax3 = axes[1, 0]
        scatter = ax3.scatter(scale_usage['overall_mean'], scale_usage['overall_std'], 
                            s=scale_usage['overall_range']*20, alpha=0.6, c=range(len(scale_usage)),
                            cmap='coolwarm')
        
        for idx, row in scale_usage.iterrows():
            ax3.annotate(row['judge'].split()[-1], 
                        (row['overall_mean'], row['overall_std']),
                        fontsize=8, alpha=0.7)
        
        ax3.set_xlabel('Mean score', fontsize=12)
        ax3.set_ylabel('Standard deviation', fontsize=12)
        ax3.set_title('Mean vs. variability of scores (size = range)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        #  4. entropy heatmap by stage
        ax4 = axes[1, 1]
        entropy_cols = [col for col in scale_usage.columns if 'entropy' in col]
        if entropy_cols:
            entropy_data = scale_usage[['judge'] + entropy_cols].set_index('judge')
            entropy_data.columns = [col.replace('_entropy', '') for col in entropy_data.columns]
            
            sns.heatmap(entropy_data.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, 
                       cbar_kws={'label': "Entropy"})
            ax4.set_title('Score entropy (diversity) by stage', fontsize=14, fontweight='bold')
            ax4.set_xlabel('', fontsize=12)
            ax4.set_ylabel('', fontsize=12)
        
        plt.suptitle('Use of the 1–25 scale by judges', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_judge_tendencies(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object. Run advanced_analyzer first.")
            return
        
        tendencies = self.analyzer.analyze_judge_tendencies()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        tendencies_sorted = tendencies.sort_values('overall_harshness')
        colors = ['red' if x < 0 else 'green' for x in tendencies_sorted['overall_harshness']]
        bars = ax1.barh(tendencies_sorted['judge'], tendencies_sorted['overall_harshness'], color=colors)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Mean difference from the consensus', fontsize=12)
        ax1.set_title('Judge harshness (negative = harsher, positive = more lenient)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        #  2. Score consistency
        ax2 = axes[0, 1]
        tendencies_sorted2 = tendencies.sort_values('overall_consistency', ascending=False)
        bars2 = ax2.barh(tendencies_sorted2['judge'], tendencies_sorted2['overall_consistency'], 
                        color=plt.cm.plasma(np.linspace(0.2, 0.8, len(tendencies_sorted2))))
        ax2.set_xlabel('Standard deviation of scores', fontsize=12)
        ax2.set_title('Scoring variability (higher = more variable)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        #  3. correlation with the consensus
        ax3 = axes[1, 0]
        tendencies_sorted3 = tendencies.sort_values('consensus_correlation')
        colors3 = plt.cm.RdYlGn(tendencies_sorted3['consensus_correlation'])
        bars3 = ax3.barh(tendencies_sorted3['judge'], tendencies_sorted3['consensus_correlation'], 
                        color=colors3)
        ax3.set_xlabel('Correlation with the consensus', fontsize=12)
        ax3.set_title('Agreement with other judges', fontsize=14, fontweight='bold')
        ax3.set_xlim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        scatter = ax4.scatter(tendencies['overall_harshness'], 
                            tendencies['overall_consistency'],
                            s=tendencies['consensus_correlation']*500,
                            alpha=0.6, c=range(len(tendencies)), cmap='viridis')
        
        for idx, row in tendencies.iterrows():
            ax4.annotate(row['judge'].split()[-1], 
                        (row['overall_harshness'], row['overall_consistency']),
                        fontsize=8, alpha=0.7)
        
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(y=tendencies['overall_consistency'].mean(), color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Harshness (difference from the consensus mean)', fontsize=12)
        ax4.set_ylabel('Scoring variability', fontsize=12)
        ax4.set_title('Judge tendency map (size = agreement with the consensus)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        bbox_style = dict(boxstyle='round,pad=0.5', facecolor='peachpuff',
                          edgecolor='darkorange', alpha=0.8, linewidth=1.5)
        ax4.text(0.07, 0.93, 'Harsh\nStable', transform=ax4.transAxes,
                fontsize=9, va='top', alpha=1, bbox=bbox_style)
        ax4.text(0.93, 0.93, 'Lenient\nStable', transform=ax4.transAxes,
                fontsize=9, va='top', ha='right', alpha=1, bbox=bbox_style)
        ax4.text(0.07, 0.07, 'Harsh\nVariable', transform=ax4.transAxes,
                fontsize=9, alpha=1, bbox=bbox_style)
        ax4.text(0.93, 0.07, 'Lenient\nVariable', transform=ax4.transAxes,
                fontsize=9, ha='right', alpha=1, bbox=bbox_style)
        
        plt.suptitle('Judge tendency analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_judge_alliances(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object. Run advanced_analyzer first.")
            return

        correlation_matrix, alliances = self.analyzer.analyze_judge_alliances(threshold=0.6)

        has_alliances = not alliances.empty

        if has_alliances:
            fig, axes = plt.subplots(2, 2, figsize=(18, 16))
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            ax1 = axes[0]
            ax2 = axes[1]

        #  1. heatmap correlation
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0.5, vmin=0, vmax=1,
                    square=True, ax=ax1, cbar_kws={'label': "Correlation"})
        ax1.set_title('Judge–to–judge score correlation matrix', fontsize=14, fontweight='bold')

        #  2. Dendrogram - hierarchical clustering
        if len(correlation_matrix) > 1:
            distance_matrix = 1 - correlation_matrix.fillna(0)
            np.fill_diagonal(distance_matrix.values, 0)

            distance_matrix = (distance_matrix + distance_matrix.T) / 2

            #  Convert to the condensed form
            condensed_distances = squareform(distance_matrix)

            linkage_matrix = linkage(condensed_distances, method='ward')
            dendrogram(linkage_matrix, labels=correlation_matrix.index.tolist(),
                       ax=ax2, orientation='right')
            ax2.set_title('Dendrogram — judge clustering', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Distance')

        if has_alliances:
            #  3. Network graph alliancey
            strong_alliances = alliances[alliances['correlation'] > 0.7]

            ax3.set_title('Strong judge alliances (correlation > 0.7)', fontsize=14, fontweight='bold')

            n_judges = len(self.judge_columns)
            angles = np.linspace(0, 2 * np.pi, n_judges, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)

            ax3.scatter(x, y, s=500, c='lightblue', edgecolors='navy', linewidth=2, alpha=0.7)

            #  Add label
            for i, judge in enumerate(self.judge_columns):
                ax3.annotate(judge.split()[-1], (x[i] * 1.15, y[i] * 1.15),
                             ha='center', va='center', fontsize=9)

            judge_positions = {judge: (x[i], y[i]) for i, judge in enumerate(self.judge_columns)}

            for _, alliance in strong_alliances.iterrows():
                if alliance['judge1'] in judge_positions and alliance['judge2'] in judge_positions:
                    pos1 = judge_positions[alliance['judge1']]
                    pos2 = judge_positions[alliance['judge2']]

                    linewidth = (alliance['correlation'] - 0.7) * 10
                    alpha = alliance['correlation']

                    ax3.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                             'r-', linewidth=linewidth, alpha=alpha)

            ax3.set_xlim([-1.5, 1.5])
            ax3.set_ylim([-1.5, 1.5])
            ax3.set_aspect('equal')
            ax3.axis('off')

            #  4. Top Alliances
            ax4.axis('tight')
            ax4.axis('off')

            top_alliances = alliances.head(10)
            table_data = []
            for _, row in top_alliances.iterrows():
                table_data.append([
                    row['judge1'].split()[-1],
                    row['judge2'].split()[-1],
                    f"{row['correlation']:.3f}",
                    row['strength']
                ])

            table = ax4.table(cellText=table_data,
                              colLabels=['Judge 1', 'Judge 2', 'Correlation', 'Strength'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.25, 0.25, 0.25, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for i, row in enumerate(top_alliances.itertuples()):
                if row.strength == 'strong':
                    table[(i + 1, 3)].set_facecolor('#90EE90')
                else:
                    table[(i + 1, 3)].set_facecolor('#FFE4B5')

            ax4.set_title('Top 10 judge alliances', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Correlation and clustering analysis of judges', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_judge_removal_impact(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object. Run advanced_analyzer first.")
            return
        
        removal_impact = self.analyzer.simulate_judge_removal()
        
        if removal_impact.empty:
            print("No data about the impact of judge removal")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        removal_sorted = removal_impact.sort_values('avg_rank_change', ascending=False)
        colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, len(removal_sorted)))
        bars1 = ax1.barh(removal_sorted['judge_removed'], removal_sorted['avg_rank_change'], 
                        color=colors1)
        ax1.set_xlabel('Average change in rank', fontsize=12)
        ax1.set_title('Average change in rank when a judge is removed', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        #  2. Maksymalna zmiana ranking
        ax2 = axes[0, 1]
        removal_sorted2 = removal_impact.sort_values('max_rank_change', ascending=False)
        colors2 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(removal_sorted2)))
        bars2 = ax2.barh(removal_sorted2['judge_removed'], removal_sorted2['max_rank_change'], 
                        color=colors2)
        ax2.set_xlabel('Maximum change in rank', fontsize=12)
        ax2.set_title('Largest impact on a single contestant', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        removal_sorted3 = removal_impact.sort_values('participants_affected', ascending=False)
        colors3 = plt.cm.Purples(np.linspace(0.3, 0.9, len(removal_sorted3)))
        bars3 = ax3.barh(removal_sorted3['judge_removed'], removal_sorted3['participants_affected'], 
                        color=colors3)
        ax3.set_xlabel('Number of contestants who changed rank', fontsize=12)
        ax3.set_title('Number of contestants affected', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        removal_sorted4 = removal_impact.sort_values('top10_changes', ascending=False)
        colors4 = ['red' if x > 2 else 'orange' if x > 0 else 'green' 
                  for x in removal_sorted4['top10_changes']]
        bars4 = ax4.barh(removal_sorted4['judge_removed'], removal_sorted4['top10_changes'], 
                        color=colors4)
        ax4.set_xlabel('Number of changes among finalists', fontsize=12)
        ax4.set_title('Impact on the leading group', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Leave–one–judge–out simulation: impact on results',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_qualification_impact_after_removing_judge(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object")
            return

        qual_impact = self.analyzer.analyze_qualification_after_judge_removal()

        if qual_impact.empty:
            print("No data about the impact on qualifications")
            return

        participant_names = {}
        for stage_data in self.processor.stages_data.values():
            for _, row in stage_data.iterrows():
                participant_names[row['number']] = f"{row['firstname']} {row['lastname']}"

        stages = ['stage1_to_stage2', 'stage2_to_stage3', 'stage3_to_final']
        stage_labels = ['I→II', 'II→III', 'III→F']

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.25, height_ratios=[0.1, 1, 1], width_ratios=[7, 3])

        ax_table = fig.add_subplot(gs[:, 0])  #  Spans both rows in column 0
        ax_table.axis('off')

        #  Prepare data for the table
        table_data = []
        for _, row in qual_impact.iterrows():
            judge = row['judge_removed']
            row_text = [judge]

            for stage in stages:
                lost = row[stage]['lost_qualification']
                gained = row[stage]['gained_qualification']

                lost_names = [participant_names.get(nr, f"#{nr}") for nr in lost]
                gained_names = [participant_names.get(nr, f"#{nr}") for nr in gained]

                cell_text = ""
                if lost_names:
                    cell_text += "↓ " + ", ".join(lost_names)
                if gained_names:
                    if cell_text:
                        cell_text += "\n"
                    cell_text += "↑ " + ", ".join(gained_names)
                if not cell_text:
                    cell_text = "—"

                row_text.append(cell_text)
            table_data.append(row_text)

        fig.subplots_adjust(top=0.95)
        table = ax_table.table(cellText=table_data,
                               colLabels=['Judge'] + stage_labels,
                               cellLoc='left',
                               bbox=[0.0, 0.0, 1.0, 0.85],
                               colWidths=[0.16, 0.28, 0.28, 0.28])

        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 2.2)

        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(weight='bold', color='white', fontsize=9)
            cell.set_height(0.04)

        for i in range(1, len(table_data) + 1):
            table[(i, 0)].set_facecolor('#ecf0f1')
            table[(i, 0)].set_text_props(weight='bold', fontsize=8)

            for j in range(1, 4):
                cell_text = table_data[i - 1][j]
                if '↓' in cell_text and '↑' in cell_text:
                    table[(i, j)].set_facecolor('#fff3cd')
                elif '↓' in cell_text:
                    table[(i, j)].set_facecolor('#f8d7da')
                elif '↑' in cell_text:
                    table[(i, j)].set_facecolor('#d4edda')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        ax_table.text(0.5, 0.94, 'Detailed impact: contestants affected by removing a judge',
                      ha='center', va='top', transform=ax_table.transAxes,
                      fontsize=13, fontweight='bold', clip_on=False)
        ax_table.text(0.5, 0.90, '(↓ = would be eliminated, ↑ = would advance)',
                      ha='center', va='top', transform=ax_table.transAxes,
                      fontsize=10, style='italic', color='#555')

        ax3 = fig.add_subplot(gs[1, 1])

        lost_data = []
        gained_data = []
        for _, row in qual_impact.iterrows():
            lost_count = sum(len(row[stage]['lost_qualification']) for stage in stages)
            gained_count = sum(len(row[stage]['gained_qualification']) for stage in stages)
            lost_data.append(lost_count)
            gained_data.append(gained_count)

        total_changes = pd.Series([l + g for l, g in zip(lost_data, gained_data)],
                                  index=qual_impact['judge_removed'])
        total_sorted = total_changes.sort_values(ascending=False)

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(total_sorted)))
        ax3.barh(total_sorted.index, total_sorted.values, color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Total number of qualification changes', fontsize=11, fontweight='bold')
        ax3.set_title('Overall impact of removing a judge',
                      fontsize=12, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, axis='x')

        for i, (judge, value) in enumerate(total_sorted.items()):
            ax3.text(value + 0.15, i, f'{int(value)}',
                     va='center', fontsize=9, fontweight='bold')

        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')

        affected_participants = {}
        for _, row in qual_impact.iterrows():
            for stage in stages:
                for nr in row[stage]['lost_qualification']:
                    name = participant_names.get(nr, f"#{nr}")
                    if name not in affected_participants:
                        affected_participants[name] = {'up': 0, 'down': 0}
                    affected_participants[name]['down'] += 1

                for nr in row[stage]['gained_qualification']:
                    name = participant_names.get(nr, f"#{nr}")
                    if name not in affected_participants:
                        affected_participants[name] = {'up': 0, 'down': 0}
                    affected_participants[name]['up'] += 1

        sorted_participants = sorted(affected_participants.items(),
                                     key=lambda x: (-(x[1]['up'] + x[1]['down']), x[0]))

        text_content = "All contestants affected by the removal\nof any judge\n"
        text_content += "=" * 42 + "\n\n"

        for name, counts in sorted_participants:
            up = counts['up']
            down = counts['down']
            total = up + down

            impact_str = ""
            if up > 0:
                impact_str += f"{up}×↑"
            if down > 0:
                if impact_str:
                    impact_str += ", "
                impact_str += f"{down}×↓"

            text_content += f"• {name:<28} ({impact_str})\n"

        ax4.text(0.05, 0.95, text_content,
                 transform=ax4.transAxes,
                 fontsize=8.5,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, pad=1))

        ax4.set_title('Summary: all affected contestants',
                      fontsize=12, fontweight='bold', pad=10)

        plt.suptitle('Impact of removing a judge on stage qualifications',
                     fontsize=15, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        #  Print summary
        print(f"\nTotal affected contestants: {len(affected_participants)}")
        print(f"Most influential judge: {total_sorted.index[0]} ({int(total_sorted.iloc[0])} zmian)")

    def visualize_final_results_impact_after_removing_judge(self, save_path: str = None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        #  Fetch data from results_after_judge_removal
        results_df = self.analyzer.generate_results_after_judge_removal()

        judges = [col.replace('_change', '') for col in results_df.columns if col.endswith('_change')]

        n_judges = len(judges)
        n_cols = 5
        n_rows = (n_judges + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5.5 * n_rows))
        axes = axes.flatten()

        bg_colors = ['#FFEFD5', '#FFFFC2']

        for idx, judge in enumerate(judges):
            ax = axes[idx]

            rank_col = f"{judge}_rank"
            change_col = f"{judge}_change"

            bg_color = bg_colors[idx % 2]
            rect_bg = mpatches.Rectangle((0, 0), 1, 1,
                                         linewidth=0,
                                         facecolor=bg_color,
                                         transform=ax.transAxes, zorder=0)
            ax.add_patch(rect_bg)

            plot_data = results_df[['number', 'firstname', 'lastname', 'original_rank', rank_col, change_col]].copy()

            plot_data['sort_key'] = plot_data[rank_col].apply(lambda x: 999 if x == 'n/a' else int(x))
            plot_data = plot_data.sort_values('sort_key')

            ax.text(0.5, 0.96, f'Without {judge}',
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    transform=ax.transAxes)

            #  Table parameters
            y_start = 0.88
            line_height = 0.075
            n_participants = len(plot_data)

            #  Draw horizontal table lines
            for i in range(n_participants + 1):
                y_line = y_start - i * line_height
                ax.plot([0.03, 0.97], [y_line, y_line], 'k-', linewidth=0.5,
                        alpha=0.3, transform=ax.transAxes, zorder=1)

            #  Draw vertical table lines
            col_positions = [0.03, 0.12, 0.70, 0.97]  #  Pozycje kolumn: rank | participant | change |
            for x_pos in col_positions:
                ax.plot([x_pos, x_pos], [y_start, y_start - n_participants * line_height],
                        'k-', linewidth=0.5, alpha=0.3, transform=ax.transAxes, zorder=1)

            #  Draw rows
            for i, (_, row) in enumerate(plot_data.iterrows()):
                y_pos = y_start - (i + 0.5) * line_height

                participant_info = f"{row['number']}. {row['firstname']} {row['lastname']}"

                if row[rank_col] == 'n/a':
                    #  Disqualified
                    rank_text = "—"
                    color = 'darkgray'

                    #  Rysuj ranking
                    ax.text(0.075, y_pos, rank_text, ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold',
                            transform=ax.transAxes, zorder=2)

                    full_participant = f"{row['number']}. {row['firstname']} {row['lastname']}"

                    #  Rysuj numer (normalnie, czarny)
                    nr_text = f"{row['number']}. "
                    ax.text(0.13, y_pos, nr_text, ha='left', va='center',
                            fontsize=9, color='black',
                            transform=ax.transAxes, zorder=2)

                    name_text = f"{row['firstname']} {row['lastname']}"

                    nr_width = len(nr_text) * 0.0055
                    name_x_start = 0.15 + nr_width

                    ax.text(name_x_start, y_pos, name_text, ha='left', va='center',
                            fontsize=9, color=color, style='italic',
                            transform=ax.transAxes, zorder=2)

                    name_width = len(name_text) * 0.0055
                    name_x_end = name_x_start + 2.3 * name_width

                    ax.plot([name_x_start, min(name_x_end, 10.68)], [y_pos, y_pos],
                            'k-', linewidth=0.8, alpha=0.5, transform=ax.transAxes, zorder=3)

                else:
                    new_rank = int(row[rank_col])
                    orig_rank = int(row['original_rank'])

                    if row[change_col] == 'n/a':
                        change_symbol = ""
                        color = 'darkgray'
                    else:
                        change = int(row[change_col])

                        if change > 0:
                            change_symbol = f"↑{change}"
                            color = 'forestgreen'
                        elif change < 0:  #  Pogorszenie
                            change_symbol = f"↓{abs(change)}"
                            color = 'firebrick'
                        else:  #  No change
                            change_symbol = "—"
                            color = 'steelblue'

                    ax.text(0.075, y_pos, f"{new_rank}.", ha='center', va='center',
                            fontsize=9, color='black', fontweight='bold',
                            transform=ax.transAxes, zorder=2)

                    ax.text(0.13, y_pos, participant_info, ha='left', va='center',
                            fontsize=9, color='black',
                            transform=ax.transAxes, zorder=2)

                    ax.text(0.835, y_pos, change_symbol, ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold',
                            transform=ax.transAxes, zorder=2)

            rect = mpatches.Rectangle((0.03, y_start - n_participants * line_height),
                                      0.94, n_participants * line_height,
                                      linewidth=1.5, edgecolor='black',
                                      facecolor='none', transform=ax.transAxes, zorder=4)
            ax.add_patch(rect)

        for idx in range(n_judges, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Finalists’ placements under leave-one-judge-out scenarios',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=(0, 0, 1, 0.99))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
        plt.close()

    def visualize_participant_journey(self, participant_nr: int, save_path: str = None):
        """Visualization of the competition journey for the selected contestant"""
        progression = self.processor.get_participant_progression(participant_nr)
        
        if progression.empty:
            print(f"No data for contestant no. {participant_nr}")
            return
        
        participant_name = None
        for stage_data in self.processor.stages_data.values():
            if participant_nr in stage_data['number'].values:
                row = stage_data[stage_data['number'] == participant_nr].iloc[0]
                participant_name = f"{row['firstname']} {row['lastname']}"
                break
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        ax1 = axes[0, 0]
        stages = progression['stage'].tolist()
        scores = progression['score'].tolist()
        
        ax1.plot(stages, scores, 'o-', linewidth=2, markersize=10, color='navy')
        ax1.fill_between(range(len(stages)), scores, alpha=0.3)
        
        for i, (stage, score) in enumerate(zip(stages, scores)):
            ax1.annotate(f'{score:.2f}', (i, score), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Stage', fontsize=12)
        ax1.set_ylabel('Score (corrected)', fontsize=12)
        ax1.set_title(f'Score progression: {participant_name} (No. {participant_nr})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        
        judge_scores = {judge: [] for judge in self.judge_columns}
        stage_labels = []
        
        for stage in stages:
            df = self.processor.stages_data[stage]
            if participant_nr in df['number'].values:
                row = df[df['number'] == participant_nr].iloc[0]
                stage_labels.append(stage)
                
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    judge_scores[judge].append(score)
        
        #  Plot score
        x = np.arange(len(stage_labels))
        width = 0.8 / len(self.judge_columns)
        
        for i, judge in enumerate(self.judge_columns):
            offset = (i - len(self.judge_columns)/2) * width
            ax2.bar(x + offset, judge_scores[judge], width, label=judge.split()[-1], alpha=0.7)
        
        ax2.set_xlabel('Stage', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Scores by each judge', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_labels)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ranks = []
        cumulative_stages = []
        
        for cum_name, cum_df in self.processor.cumulative_scores.items():
            if participant_nr in cum_df['number'].values:
                rank = cum_df[cum_df['number'] == participant_nr]['rank'].iloc[0]
                ranks.append(rank)
                cumulative_stages.append(cum_name.replace('_cumulative', ''))
        
        if ranks:
            ax3.plot(cumulative_stages, ranks, 'o-', linewidth=2, markersize=10, color='red')

            for i, (stage, rank) in enumerate(zip(cumulative_stages, ranks)):
                ax3.annotate(f'{int(rank)}', (i, rank), textcoords="offset points", 
                           xytext=(0,-10), ha='center', fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('Stage (cumulative)', fontsize=12)
            ax3.set_ylabel('Rank position', fontsize=12)
            ax3.set_title('Position in cumulative ranking', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        all_scores = []
        for stage in stages:
            df = self.processor.corrected_data[stage]
            if participant_nr in df['number'].values:
                row = df[df['number'] == participant_nr].iloc[0]
                
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    if pd.notna(score):
                        all_scores.append(score)
        
        if all_scores:
            stats_text = f"""
            Contestant statistics: {participant_name}
            
            Number of stages: {len(stages)}
            
            Mean score: {np.mean(all_scores):.2f}
            Median score: {np.median(all_scores):.2f}
            Standard deviation: {np.std(all_scores):.2f}
            Min score: {min(all_scores):.0f}
            Max score: {max(all_scores):.0f}
            Range: {max(all_scores) - min(all_scores):.0f}
            
            Total number of scores: {len(all_scores)}
            """
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Competition journey analysis: {participant_name}',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_favorites_analysis(self, save_path: str = None):
        if not self.analyzer:
            print("No analyzer object. Run advanced_analyzer first.")
            return
        
        favorites = self.analyzer.find_judge_favorites(min_stages=3)
        
        if favorites.empty:
            print("No clear favorites found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        fav_counts = favorites[favorites['type'] == 'favorite']['judge'].value_counts()
        unfav_counts = favorites[favorites['type'] == 'unfavorite']['judge'].value_counts()
        
        judges_all = list(set(list(fav_counts.index) + list(unfav_counts.index)))
        judges_all.sort()
        
        x = np.arange(len(judges_all))
        width = 0.35
        
        favs = [fav_counts.get(j, 0) for j in judges_all]
        unfavs = [unfav_counts.get(j, 0) for j in judges_all]
        
        bars1 = ax1.bar(x - width/2, favs, width, label='Favorites', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, unfavs, width, label='Anti–favorites', color='red', alpha=0.7)
        
        ax1.set_ylabel('Number of contestants', fontsize=12)
        ax1.set_title('Number of favorites and anti–favorites per judge', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([j.split()[-1] for j in judges_all], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        
        judge_bias = favorites.groupby(['judge', 'type'])['avg_difference'].mean().unstack(fill_value=0)
        
        if 'favorite' in judge_bias.columns and 'unfavorite' in judge_bias.columns:
            judge_bias_diff = judge_bias['favorite'] + judge_bias['unfavorite']
            judge_bias_diff = judge_bias_diff.sort_values()
            
            colors = ['red' if x < 0 else 'green' for x in judge_bias_diff]
            bars = ax2.barh(range(len(judge_bias_diff)), judge_bias_diff, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(judge_bias_diff)))
            ax2.set_yticklabels([j.split()[-1] for j in judge_bias_diff.index])
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Average magnitude of favoring', fontsize=12)
            ax2.set_title('Favoring balance (+ favorites, − anti–favorites)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        #  3. Strongest favouritism cases
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        top_favorites = favorites.nlargest(10, 'avg_difference')[
            ['judge', 'participant_name', 'avg_difference', 'n_stages']
        ]
        
        if not top_favorites.empty:
            table_data = []
            for _, row in top_favorites.iterrows():
                table_data.append([
                    row['judge'],
                    row['participant_name'],
                    f"+{row['avg_difference']:.2f}",
                    f"{int(row['n_stages'])}"
                ])
            
            table = ax3.table(cellText=table_data,
                            colLabels=['Judge', 'Contestant', 'Mean difference', 'Stages'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            for i in range(len(table_data)):
                table[(i+1, 2)].set_facecolor('#90EE90')
        
        ax3.set_title('Top 10 cases of over–scoring', fontsize=14, fontweight='bold', pad=20)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        top_unfavorites = favorites.nsmallest(10, 'avg_difference')[
            ['judge', 'participant_name', 'avg_difference', 'n_stages']
        ]
        
        if not top_unfavorites.empty:
            table_data2 = []
            for _, row in top_unfavorites.iterrows():
                table_data2.append([
                    row['judge'],
                    row['participant_name'],
                    f"{row['avg_difference']:.2f}",
                    f"{int(row['n_stages'])}"
                ])
            
            table2 = ax4.table(cellText=table_data2,
                             colLabels=['Judge', 'Contestant', 'Mean difference', 'Stages'],
                             cellLoc='center',
                             loc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 2)
            
            for i in range(len(table_data2)):
                table2[(i+1, 2)].set_facecolor('#FFB6C1')
        
        ax4.set_title('Top 10 cases of under–scoring', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Analysis of judges’ favorites and anti–favorites',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, output_dir: str = 'visualizations'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating a comprehensive visual report...")
        
        for stage in ['stage1', 'stage2', 'stage3', 'final']:
            print(f"  - Score distribution {stage}")
            self.visualize_score_distribution(stage, f'{output_dir}/1_distribution_{stage}.png')
        
        #  2. Scale usage
        print("  - Scale usage analysis")
        self.visualize_scale_usage_comparison(f'{output_dir}/2_scale_usage.png')
        
        print("  - Judge tendencies")
        self.visualize_judge_tendencies(f'{output_dir}/3_judge_tendencies.png')
        
        #  4. Alliances
        print("  - Correlations and alliances")
        self.visualize_judge_alliances(f'{output_dir}/4_alliances.png')

        print("  - Simulating removal of judges")
        self.visualize_judge_removal_impact(f'{output_dir}/5_removal_impact.png')

        print("  - Qualification to the next round without one judge...")
        self.visualize_qualification_impact_after_removing_judge(
            f'{output_dir}/6_qualification_impact_after_removing_judge.png')

        print("  - Final results without one judge...")
        self.visualize_final_results_impact_after_removing_judge(f'{output_dir}/7_final_results_impact_after_removing_judge.png')
        
        print("  - Favorites analysis")
        self.visualize_favorites_analysis(f'{output_dir}/8_favorites.png')
        
        print(f"\nVisualizations saved in directory: {output_dir}")


if __name__ == "__main__":
    print("Run this module after loading data with chopin_data_processor.py")
