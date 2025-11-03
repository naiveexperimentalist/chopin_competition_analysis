"""
run_full_analysis.py
====================

This script orchestrates the entire Chopin competition analysis workflow from
data processing through advanced, statistical, clustering and multistage
analyses.  All generated visualisations are consolidated into a single
directory with numbered prefixes indicating both the order in which they
should appear in a report and the group (general, normalisation, multistage,
statistical).  By running this script once you will obtain a complete set
of CSV outputs and PNG figures ready for inclusion in a final report.

Usage:

    python run_full_analysis.py --stage1 path/to/stage1.csv \
                                --stage2 path/to/stage2.csv \
                                --stage3 path/to/stage3.csv \
                                --final  path/to/final.csv   \
                                --output all_results

The script assumes the underlying modules (`chopin_data_processor`,
`chopin_advanced_analyzer`, `chopin_visualization`, `chopin_controversy_analyzer`,
`chopin_statistical_analyzer`, `chopin_clustering_analyzer`,
`chopin_multistage_clustering`, `chopin_advanced_visualizations`,
`chopin_statistical_visualization`, and `main_analysis`) are available in the
same repository.  It does not modify those modules; instead, it calls
their public interfaces and rearranges the outputs into a unified report
folder.
"""

import argparse
import os
import shutil

from chopin_data_processor import process_competition_data
from chopin_advanced_analyzer import run_advanced_analysis
from chopin_visualization import ChopinVisualization
from chopin_controversy_analyzer import ChopinControversyAnalyzer
from chopin_statistical_analyzer import run_statistical_analysis
from chopin_clustering_analyzer import run_clustering_analysis, ChopinClusteringAnalyzer
from chopin_multistage_clustering import run_multistage_analysis
from chopin_advanced_visualizations import run_advanced_visualizations
from chopin_statistical_visualization import ChopinStatisticalVisualization
from main_analysis import analyze_normalization_impact
from chopin_final_score_stability import FinalScoreStabilityAnalyzer, FinalScoreStabilityVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run full Chopin competition analysis")
    parser.add_argument('--stage1', default='chopin_2025_stage1_by_judge.csv', help='CSV file with stage 1 scores by judge')
    parser.add_argument('--stage2', default='chopin_2025_stage2_by_judge.csv', required=True, help='CSV file with stage 2 scores by judge')
    parser.add_argument('--stage3', default='chopin_2025_stage3_by_judge.csv', required=True, help='CSV file with stage 3 scores by judge')
    parser.add_argument('--final', default='chopin_2025_final_by_judge.csv', required=True, help='CSV file with final scores by judge')
    parser.add_argument('--output', default='full_analysis_results', help='Base output directory')
    args = parser.parse_args()

    # Create top-level output directories
    base_dir = args.output
    os.makedirs(base_dir, exist_ok=True)
    tmp_dir = os.path.join(base_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Data processing
    print("[1/9] Loading and processing competition data...")
    processor = process_competition_data(
        args.stage1, args.stage2, args.stage3, args.final,
        output_dir=os.path.join(tmp_dir, 'processed_data'))

    # 2. Advanced analysis and general visualisations
    print("[2/9] Running advanced analyses and generating general visualisations...")
    # Perform advanced analyses and save numerical outputs
    advanced_analyzer = run_advanced_analysis(processor, output_dir=os.path.join(tmp_dir, 'advanced_results'))
    # Generate general visualisations to a temporary folder with default numbering
    gen_viz_dir = os.path.join(tmp_dir, 'general_visualizations')
    vis = ChopinVisualization(processor, advanced_analyzer)
    vis.create_comprehensive_report(output_dir=gen_viz_dir)

    # 3. Normalisation impact analysis
    print("[3/9] Analysing impact of score normalisation...")
    norm_dir = os.path.join(tmp_dir, 'normalisation_analysis')
    os.makedirs(norm_dir, exist_ok=True)
    # The analyse_normalization_impact function prints and saves its own files
    analyze_normalization_impact(save_path = 'visualizations/normalization_comparison.png')

    # 4. Multistage clustering and advanced visualisations
    print("[4/9] Running multistage clustering analyses...")
    # Run clustering for each stage and return the cluster analyzer (for statistical viz)
    cluster_analyzer = run_clustering_analysis(processor, output_dir=os.path.join(tmp_dir, 'clustering_results'))
    # Prepare dictionary of raw data files for multistage analysis
    data_files = {
        'stage1': args.stage1,
        'stage2': args.stage2,
        'stage3': args.stage3,
        'final': args.final
    }
    # Run multistage analysis using original CSVs rather than the processor object
    multistage_analyzer = run_multistage_analysis(data_files, output_dir=os.path.join(tmp_dir, 'multistage_analysis'))
    # Generate advanced multistage visualisations
    adv_vis_dir = os.path.join(tmp_dir, 'advanced_visualizations')
    os.makedirs(adv_vis_dir, exist_ok=True)
    run_advanced_visualizations(multistage_analyzer, output_dir=adv_vis_dir)

    # 5. Controversy and statistical analyses
    print("[5/9] Performing controversy and statistical analyses...")
    controversy_analyzer = ChopinControversyAnalyzer(processor)
    stat_analyzer = run_statistical_analysis(processor, output_dir=os.path.join(tmp_dir, 'statistical_results'))

    # 6. Statistical visualisations (including clustering results & PCA) to temp folder
    print("[6/9] Generating statistical and clustering visualisations...")
    stat_vis_dir = os.path.join(tmp_dir, 'statistical_visualizations')
    os.makedirs(stat_vis_dir, exist_ok=True)
    stat_vis = ChopinStatisticalVisualization(
        processor,
        controversy_analyzer=controversy_analyzer,
        statistical_analyzer=stat_analyzer,
        clustering_analyzer=cluster_analyzer
    )
    stat_vis.create_comprehensive_statistical_report(output_dir=stat_vis_dir)

    # 7. Consolidate all visualisations into final directory with new naming scheme
    print("[7/9] Consolidating and renaming visualisations...")
    final_viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    # Define mapping from original filenames (without path) to new prefixed names
    # General visuals (originally numbered 1_x) -> 01-09
    rename_map = {
        # General visualisations
        '1_distribution_stage1.png': '01_distribution_stage1.png',
        '1_distribution_stage2.png': '02_distribution_stage2.png',
        '1_distribution_stage3.png': '03_distribution_stage3.png',
        '1_distribution_final.png': '04_distribution_final.png',
        '2_scale_usage.png': '05_scale_usage.png',
        '3_judge_tendencies.png': '06_judge_tendencies.png',
        '4_alliances.png': '07_alliances.png',
        '5_removal_impact.png': '08_removal_impact.png',
        '6_qualification_impact_after_removing_judge.png': '09_qualification_impact.png',
        '7_final_results_impact_after_removing_judge.png': '10_final_results_impact.png',
        '8_favorites.png': '11_favorites_analysis.png',
        # Normalisation visuals produced by main_analysis functions
        'normalization_comparison.png': '12_normalisation_comparison.png',
        'normalization_impact.png': '13_normalisation_impact.png',
        # Advanced multistage visuals (originally 10-15)
        '10_participant_pca.png': '14_participant_pca.png',
        '11_judge_pca.png': '15_judge_pca.png',
        '12_multistage_heatmap.png': '16_multistage_heatmap.png',
        '13_judge_consistency.png': '17_judge_consistency.png',
        '14_participant_trajectories.png': '18_participant_trajectories.png',
        '15_cluster_evolution.png': '19_cluster_evolution.png',
        # Statistical & clustering visuals (originally 16-23)
        '16_score_diversity_heatmap.png': '20_score_diversity_heatmap.png',
        '17_score_diversity_analysis.png': '21_score_diversity_analysis.png',
        '18_outliers_analysis.png': '22_outliers_analysis.png',
        '19_ranking_stability.png': '23_ranking_stability.png',
        '20_statistical_significance.png': '24_statistical_significance.png',
        '21_pairwise_agreement.png': '25_pairwise_agreement.png',
        '22_clustering_results.png': '26_clustering_results.png',
        '23_pca_judges.png': '27_pca_judges.png'
    }

    # Copy and rename files based on mapping
    # Search through all temporary visualization directories
    temp_dirs = [gen_viz_dir, norm_dir, adv_vis_dir, stat_vis_dir]
    for temp_dir in temp_dirs:
        for fname in os.listdir(temp_dir):
            new_name = rename_map.get(fname)
            if not new_name:
                # Skip files that are not mapped (e.g., CSVs or intermediate)
                continue
            src = os.path.join(temp_dir, fname)
            dst = os.path.join(final_viz_dir, new_name)
            shutil.copyfile(src, dst)

    # 8. Copy important CSV summaries to base_dir for convenience
    print("[8/9] Copying summary CSVs...")
    # Copy processed rankings and advanced results
    csv_extensions = ('.csv', '.xlsx')
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            if file.endswith(csv_extensions):
                src = os.path.join(root, file)
                dst = os.path.join(base_dir, file)
                # Avoid overwriting duplicates by prepending directory name
                if os.path.exists(dst):
                    dst = os.path.join(base_dir, f"{os.path.basename(root)}_{file}")
                shutil.copyfile(src, dst)

    print("[9/9] Analiza stabilności wyniku końcowego (bootstrap)...")
    stage_files = {
        'stage1': args.stage1,
        'stage2': args.stage2,
        'stage3': args.stage3,
        'final': args.final
    }
    stability_analyzer = FinalScoreStabilityAnalyzer(stage_files)
    bootstrap_results = stability_analyzer.bootstrap_final_scores(n_iterations=10000)

    stability_viz_dir = os.path.join(tmp_dir, 'final_score_stability')
    visualizer = FinalScoreStabilityVisualizer(stability_analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results,
        output_dir=stability_viz_dir
    )
    print("\nFull analysis complete. Visualisations available in:")
    print(f"  {final_viz_dir}")
    print("CSV summaries and results available in:")
    print(f"  {base_dir}\n")


if __name__ == '__main__':
    main()