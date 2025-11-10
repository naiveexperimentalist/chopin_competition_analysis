"""\nHigh-level helpers to build report sections and compare normalisation methods.

Writes a markdown report, computes normalisation variants (Z-score, Min–Max,
rank-based), and renders side-by-side ranking tables for comparison.\n"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

from chopin_data_processor import ChopinCompetitionProcessor, process_competition_data
from chopin_advanced_analyzer import ChopinAdvancedAnalyzer, run_advanced_analysis
from chopin_visualization import ChopinVisualization

def create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis'):
    """Tworzy kompletny report analyze konkursu"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Chopin Competition 2025 — Analysis Report\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # 1. Basic statistics
        f.write("## 1. Basic statistics\n\n")
        
        f.write("### Number of contestants per stage:\n")
        for stage_name, df in processor.stages_data.items():
            f.write(f"- **{stage_name}**: {len(df)} uczestników\n")
        
        # 2. Judges analysis
        f.write("\n## 2. Judges analysis\n\n")
        
        judge_stats = analyzer.get_judge_statistics()
        if not judge_stats.empty:
            f.write("### Scale usage by judges:\n\n")
            scale_usage = analyzer.analyze_scale_usage()
            
            # Top 3 najbardziej liberalni w score
            top_range = scale_usage.nlargest(3, 'overall_range')
            f.write("**Judges using the widest scale:**\n")
            for _, row in top_range.iterrows():
                f.write(f"- {row['judge']}: rozpiętość {row['overall_range']:.1f} punktów\n")
            
            # Top 3 najbardziej konserwatywni
            bottom_range = scale_usage.nsmallest(3, 'overall_range')
            f.write("\n**Judges using the narrowest scale:**\n")
            for _, row in bottom_range.iterrows():
                f.write(f"- {row['judge']}: rozpiętość {row['overall_range']:.1f} punktów\n")
        
        # 3. Judge tendencies
        f.write("\n### Judge tendencies:\n\n")
        tendencies = analyzer.analyze_judge_tendencies()
        
        if not tendencies.empty:
            # Najbardziej surowi
            harsh = tendencies.nsmallest(3, 'overall_harshness')
            f.write("**Most harsh judges:**\n")
            for _, row in harsh.iterrows():
                f.write(f"- {row['judge']}: średnio {abs(row['overall_harshness']):.2f} punkta poniżej konsensusu\n")
            
            lenient = tendencies.nlargest(3, 'overall_harshness')
            f.write("\n**Most lenient judges:**\n")
            for _, row in lenient.iterrows():
                f.write(f"- {row['judge']}: średnio {row['overall_harshness']:.2f} punkta powyżej konsensusu\n")
        
        f.write("\n## 3. Alliances and correlations\n\n")
        correlation_matrix, alliances = analyzer.analyze_judge_alliances(threshold=0.7)
        
        if not alliances.empty:
            f.write("### Strongest alliances (korelacja > 0.7):\n")
            for _, row in alliances.head(5).iterrows():
                f.write(f"- **{row['judge1']}** i **{row['judge2']}**: korelacja {row['correlation']:.3f}\n")
        
        f.write("\n## 4. Impact of individual judges on results\n\n")
        removal_impact = analyzer.simulate_judge_removal()
        
        if not removal_impact.empty:
            most_influential = removal_impact.nlargest(3, 'avg_rank_change')
            f.write("### Judges with the largest impact on ranking:\n")
            for _, row in most_influential.iterrows():
                f.write(f"- **{row['judge_removed']}**: usunięcie zmienia ranking średnio o {row['avg_rank_change']:.2f} pozycji\n")
        
        # 6. Faworyci
        f.write("\n## 5. Favouritism analysis\n\n")
        favorites = analyzer.find_judge_favorites(min_stages=3)
        
        if not favorites.empty:
            # Strongest favouritism cases
            top_fav = favorites[favorites['type'] == 'favorite'].nlargest(3, 'avg_difference')
            if not top_fav.empty:
                f.write("### Strongest favouritism cases:\n")
                for _, row in top_fav.iterrows():
                    f.write(f"- **{row['judge']}** → {row['participant_name']}: +{row['avg_difference']:.2f} punkta średnio\n")
            
            # Strongest under-scoring cases
            top_unfav = favorites[favorites['type'] == 'unfavorite'].nsmallest(3, 'avg_difference')
            if not top_unfav.empty:
                f.write("\n### Strongest under-scoring cases:\n")
                for _, row in top_unfav.iterrows():
                    f.write(f"- **{row['judge']}** → {row['participant_name']}: {row['avg_difference']:.2f} punkta średnio\n")
        
        # 7. result finalne
        f.write("\n## 6. Final results\n\n")
        if 'final_cumulative' in processor.cumulative_scores:
            final_results = processor.cumulative_scores['final_cumulative']
            f.write("### Top 10 finalists:\n")
            for _, row in final_results.head(10).iterrows():
                f.write(f"{int(row['rank'])}. **{row['firstname']} {row['lastname']}** - {row['cumulative_score']:.2f} punktów\n")
        
        # 8. Wnioski
        f.write("\n## 7. Key takeaways\n\n")
        
        if not scale_usage.empty:
            avg_coverage = scale_usage['scale_coverage'].mean()
            f.write(f"- **Scale usage**: Średnio sędziowie używali {avg_coverage:.1f}% dostępnej skali (1-25)\n")
            
            if avg_coverage < 50:
                f.write("  - ⚠️ Niska dywersyfikacja ocen - sędziowie używają ograniczonego zakresu punktacji\n")
        
        if not tendencies.empty:
            avg_consensus = tendencies['consensus_correlation'].mean()
            f.write(f"- **Scoring agreement**: Średnia korelacja z konsensusem wynosi {avg_consensus:.3f}\n")
            
            if avg_consensus < 0.7:
                f.write("  - ⚠️ Znaczące różnice w kryteriach oceniania między sędziami\n")
        
        if not removal_impact.empty:
            max_impact = removal_impact['avg_rank_change'].max()
            if max_impact > 2:
                f.write(f"- **Impact of individual judges**: Maksymalny wpływ na ranking to {max_impact:.2f} pozycji\n")
                f.write("  - ⚠️ Niektórzy sędziowie mają nieproporcjonalnie duży wpływ na wyniki\n")
        
        f.write("\n---\n")
        f.write("*Report generated automatically*\n")
    
    print(f"Text report saved to: {report_path}")

def visualize_normalization_comparison(normalization_results: dict, save_path: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    norm_methods = ['zscore', 'minmax', 'rank']
    method_labels = {
        'zscore': "Z-Score",
        'minmax': "Min-Max",
        'rank': "Rank"
    }

    # Figura z 3 subplotami obok siebie
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))

    bg_colors = ['#F0FFFF', '#F0FFFF']

    for idx, norm_method in enumerate(norm_methods):
        ax = axes[idx]
        ax.axis('off')

        # Pobierz dane dla tej metody
        df = normalization_results[norm_method].copy()

        # 1. ZMIANA SORTOWANIA:
        df_sorted = df.sort_values(
            ['rank', 'cumulative_score'],
            ascending=[True, False]
        ).reset_index(drop=True)

        n_participants = len(df_sorted)

        bg_color = bg_colors[idx % 2]
        # Rect_bg = mpatches.Rectangle((0, 0), 1, 1,
        # Linewidth=0,
        # Facecolor=bg_color,
        # Transform=ax.transAxes, zorder=0)
        # Ax.add_patch(rect_bg)

        ax.text(0.5, 0.97, f'Normalization: {method_labels[norm_method]}',
                ha='center', va='top', fontsize=13, fontweight='bold',
                transform=ax.transAxes)

        # Parametry tabelki
        y_start = 0.91
        line_height = 0.082

        header_y = y_start + 0.015
        ax.text(0.10, header_y, 'Rank', ha='center', va='center',
                fontsize=10, fontweight='bold', transform=ax.transAxes)
        ax.text(0.42, header_y, 'Contestant', ha='left', va='center',
                fontsize=10, fontweight='bold', transform=ax.transAxes)
        # Ax.text(0.85, header_y, 'result', ha='center', va='center',
        # Fontsize=10, fontweight='bold', transform=ax.transAxes)

        # Rysuj linie poziome
        for i in range(n_participants + 1):
            y_line = y_start - i * line_height
            ax.plot([0.05, 0.95], [y_line, y_line], 'k-', linewidth=0.5,
                    alpha=0.3, transform=ax.transAxes, zorder=1)

        # Rysuj linie pionowe
        # Col_positions = [0.05, 0.18, 0.75, 0.95]
        col_positions = [0.05, 0.18, 0.95]
        for x_pos in col_positions:
            ax.plot([x_pos, x_pos], [y_start, y_start - n_participants * line_height],
                    'k-', linewidth=0.5, alpha=0.3, transform=ax.transAxes, zorder=1)

        # Colour dla podium
        rank_colors = {
            2: "#C0C0C0",  # Srebrny
        }

        # Rysuj wiersze z danymi
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            y_pos = y_start - (i + 0.5) * line_height

            rank = int(row['rank'])
            participant = f"{int(row['number'])}. {row['firstname']} {row['lastname']}"
            # Score = f"{row['cumulative_score']:.2f}"

            text_color = 'black'
            strikethrough_settings = {}
            rank_label = f"{rank}."

            if rank == 999:
                text_color = 'gray'  # Przyciemnij tekst
                # (Najprostsza i najczystsza metoda w Matplotlib)

                ax.plot([0.20, 0.43], [y_pos, y_pos],
                        'r-', linewidth=1.0, alpha=0.6, transform=ax.transAxes, zorder=3)

            else:
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

            # Pozycja w ranking
            ax.text(0.115, y_pos, rank_label, ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold' if rank != 999 else 'normal',
                    transform=ax.transAxes, zorder=2, **strikethrough_settings)

            # Contestant
            ax.text(0.20, y_pos, participant, ha='left', va='center',
                    fontsize=10, color=text_color,
                    transform=ax.transAxes, zorder=2, **strikethrough_settings)

            # # result skumulowany
            # Ax.text(0.85, y_pos, score, ha='center', va='center',
            # Fontsize=10, color=text_color, fontweight='bold',
            # Transform=ax.transAxes, zorder=2, **strikethrough_settings)

        rect = mpatches.Rectangle((0.05, y_start - n_participants * line_height),
                                  0.90, n_participants * line_height,
                                  linewidth=1.5, edgecolor='black',
                                  facecolor='none', transform=ax.transAxes, zorder=4)
        ax.add_patch(rect)

    fig.suptitle('Comparison of final rankings across normalization methods',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is None:
        save_path = 'normalization_comparison.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization of the normalization comparison saved in {save_path}")
    plt.close()

def analyze_normalization_impact(save_path):
    """Analysing normalization impact on the competition results"""
    print("\n=== NORMALIZATION IMPACT ANALYSIS ===\n")

    QUALIFICATION_THRESHOLDS = {
        'stage1': 40,
        'stage2': 20,
        'stage3': 10,
        'real_stage3': 11
    }
    # Load dane
    processor = ChopinCompetitionProcessor()
    processor.load_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    )
    
    processor.calculate_all_stages()
    processor.calculate_cumulative_scores()
    original_final = processor.cumulative_scores.get('final_cumulative')

    if original_final is None:
        print("No final results")
        return
    
    analyzer = ChopinAdvancedAnalyzer(processor)
    normalized_data = analyzer.normalize_scores()
    
    normalization_results = {}
    
    for norm_method in ['zscore', 'minmax', 'rank']:
        print(f"Processing normalization: {norm_method}")
        
        norm_processor = ChopinCompetitionProcessor()
        norm_processor.judge_columns = processor.judge_columns
        norm_processor.weights = processor.weights
        norm_processor.deviation_limits = processor.deviation_limits
        
        norm_processor.stages_data = {}
        for stage in ['stage1', 'stage2', 'stage3', 'final']:
            key = f'{stage}_{norm_method}'
            if key in normalized_data:
                norm_processor.stages_data[stage] = normalized_data[key]
        
        norm_processor.calculate_all_stages()
        norm_processor.calculate_cumulative_scores()

        exclude_from_finals = set()
        new_stage1 = norm_processor.cumulative_scores.get('stage1_cumulative')
        stage2_qualifiers = set(
            new_stage1.nsmallest(QUALIFICATION_THRESHOLDS['stage1'], 'rank')['number'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['number'] in stage2_qualifiers:
                exclude_from_finals.add(int(row['number']))

        new_stage2 = norm_processor.cumulative_scores.get('stage2_cumulative')
        stage3_qualifiers = set(
            new_stage2.nsmallest(QUALIFICATION_THRESHOLDS['stage2'], 'rank')['number'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['number'] in stage3_qualifiers:
                exclude_from_finals.add(int(row['number']))

        new_stage3 = norm_processor.cumulative_scores.get('stage3_cumulative')
        final_qualifiers = set(
            new_stage3.nsmallest(QUALIFICATION_THRESHOLDS['stage3'], 'rank')['number'].values
        )
        for i, (_, row) in enumerate(original_final.iterrows()):
            if not row['number'] in final_qualifiers:
                exclude_from_finals.add(int(row['number']))
        norm_processor.calculate_cumulative_scores(exclude_from_finals)

        if 'final_cumulative' in norm_processor.cumulative_scores:
            normalization_results[norm_method] = norm_processor.cumulative_scores['final_cumulative']

    visualize_normalization_comparison(normalization_results, save_path)
    print("\n=== Visualization of the normalization comparison ===\n")
    
    comparison_data = []
    
    for idx, row in original_final.iterrows():
        participant = f"{row['firstname']} {row['lastname']}"
        nr = row['number']
        
        comparison = {
            'contestant': participant,
            'number': nr,
            'Original_rank': row['rank'],
            'Original_score': row['cumulative_score']
        }
        
        for norm_method, norm_df in normalization_results.items():
            if nr in norm_df['number'].values:
                norm_row = norm_df[norm_df['number'] == nr].iloc[0]
                comparison[f'{norm_method}_rank'] = norm_row['rank']
                comparison[f'{norm_method}_score'] = norm_row['cumulative_score']
                comparison[f'{norm_method}_rank_change'] = row['rank'] - norm_row['rank']
        
        comparison_data.append(comparison)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("The most significant changes after Z-score normalization:")
    if 'zscore_rank_change' in comparison_df.columns:
        top_changes = comparison_df.nlargest(10, 'zscore_rank_change', keep='all')
        for _, row in top_changes.iterrows():
            print(f"  {row['contestant']}: {row['Original_rank']:.0f} → {row['zscore_rank']:.0f} "
                  f"(change: {row['zscore_rank_change']:+.0f})")
    
    comparison_df.to_csv(os.path.join(save_path, 'normalization_comparison.csv'), index=False)
    print("\nFull comparison saved in normalization_comparison.csv")
    
    # Visualisation zmian
    if 'zscore_rank_change' in comparison_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, norm_method in enumerate(['zscore', 'minmax', 'rank']):
            ax = axes[i]
            col = f'{norm_method}_rank_change'
            
            if col in comparison_df.columns:
                changes = comparison_df[col].abs()
                
                ax.hist(changes, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(changes.mean(), color='red', linestyle='--', 
                          label=f'Mean: {changes.mean():.2f}')
                ax.set_xlabel('Rank position change', fontsize=12)
                ax.set_ylabel('Number of contestants', fontsize=12)
                ax.set_title(f'Impact of {norm_method.upper()} normalization', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Normalization impact', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'normalization_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nChart saved in normalization_impact.png")
    
    return comparison_df

def main():
    print("=" * 60)
    print("CHOPIN COMPETITION ANALYSIS")
    print("=" * 60)
    
    required_files = [
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n⚠️ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nMake sure, all CSV files are present in the working directory.")
        return
    
    # 1. Przetworzenie podstawowych danych
    print("\n[1/6] Reading and processing data…")
    processor = process_competition_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv',
        output_dir='results'
    )
    
    print("\n[2/6] Advanced analysis…")
    analyzer = run_advanced_analysis(processor, output_dir='advanced_results')
    
    # 3. visualisation
    print("\n[3/6] Visualizations…")
    visualizer = ChopinVisualization(processor, analyzer)
    visualizer.create_comprehensive_report(output_dir='visualizations')
    
    # 4. analyze normalisation
    print("\n[4/6] Normalization impact analysis…")
    normalization_comparison = analyze_normalization_impact()
    
    # 5. report tekstowy
    print("\n[5/6] Report…")
    create_analysis_report(processor, analyzer, visualizer, output_dir='full_analysis')
    
    # 6. Podsumowanie
    print("\n[6/6] Finished!")
    print("\n" + "=" * 60)
    print("OUTPUT STRUCTURE:")
    print("=" * 60)
    print("📁 results/           - Basic results and corrected score")
    print("📁 advanced_results/  - Advanced statistical analyses")
    print("📁 visualizations/    - All charts and visualization")
    print("📁 full_analysis/     - Text report")
    print("\n📊 Key files:")
    print("  - results/final_cumulative.csv          - Final ranking")
    print("  - advanced_results/judge_tendencies.csv - Judge tendencies")
    print("  - advanced_results/judge_alliances.csv  - Judge alliances")
    print("  - normalization_comparison.csv          - Normalization impact")
    print("  - full_analysis/analysis_report.md      - Analysis report")
    
if __name__ == "__main__":
    main()
