"""
Runner/orchestrator for the full Chopin Competition analysis pipeline.

Invokes data processing, advanced analysis, visualisations (general,
normalisation, multistage, statistical) and final-score stability studies.
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
from chopin_score_perturbation_bootstrap import ScorePerturbationAnalyzer, PerturbationVisualizer

def parse_analyses_to_run(analyses_str):
    """
    Parse the analyses string to determine which analyses to run.

    Available options:
    A - Advanced analysis and general visualisation
    N - Normalisation impact analysis
    M - Multistage clustering and advanced visualisation
    C - Controversy and statistical analysis
    S - Statistical visualisation (including clustering results & PCA)
    B - Bootstrap-based final score stability analysis
    P - Score-perturbation-based final score stability analysis

    Default (empty or 'ALL'): run all analyses
    """
    if not analyses_str or analyses_str.upper() == 'ALL':
        return {'A', 'N', 'M', 'C', 'S', 'B', 'P'}

    # Convert to uppercase and create set of unique characters
    return set(analyses_str.upper())

def main():
    print("Running full Chopin competition analysis")
    parser = argparse.ArgumentParser(description="Run full Chopin competition analysis")
    parser.add_argument('--stage1', default='chopin_2025_stage1_by_judge.csv', help='CSV file with stage 1 scores by judge')
    parser.add_argument('--stage2', default='chopin_2025_stage2_by_judge.csv', help='CSV file with stage 2 scores by judge')
    parser.add_argument('--stage3', default='chopin_2025_stage3_by_judge.csv', help='CSV file with stage 3 scores by judge')
    parser.add_argument('--final', default='chopin_2025_final_by_judge.csv', help='CSV file with final scores by judge')
    parser.add_argument('--output', default='full_analysis_results', help='Base output directory')
    parser.add_argument('--analyses', default='ALL',
                       help='Analyses to run: A=Advanced, N=Normalisation, M=Multistage, C=Controversy, S=Statistical vis, B=Bootstrap, P=Perturbation. Default=ALL')
    args = parser.parse_args()

    # Parse which analyses to run
    analyses_to_run = parse_analyses_to_run(args.analyses)

    print(f"Analyses to run: {', '.join(sorted(analyses_to_run))}")
    if analyses_to_run == {'A', 'N', 'M', 'C', 'S', 'B', 'P'}:
        print("Running ALL analyses")
    else:
        analysis_names = {
            'A': 'Advanced analysis and general visualisation',
            'N': 'Normalisation impact analysis',
            'M': 'Multistage clustering and advanced visualisation',
            'C': 'Controversy and statistical analysis',
            'S': 'Statistical visualisation',
            'B': 'Bootstrap-based stability analysis',
            'P': 'Perturbation-based stability analysis'
        }
        print("Selected analyses:")
        for code in sorted(analyses_to_run):
            if code in analysis_names:
                print(f"  [{code}] {analysis_names[code]}")

    # Create top-level output directories
    base_dir = args.output
    os.makedirs(base_dir, exist_ok=True)
    tmp_dir = os.path.join(base_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    gen_viz_dir = os.path.join(tmp_dir, 'general_visualizations')
    adv_viz_dir = os.path.join(tmp_dir, 'advanced_visualizations')
    norm_dir = os.path.join(tmp_dir, 'normalisation_analysis')
    stat_viz_dir = os.path.join(tmp_dir, 'statistical_visualizations')
    stability_viz_dir = os.path.join(tmp_dir, 'final_score_stability')

    # 1. Data processing (always required)
    print("[1/X] Loading and processing competition data...")
    processor = process_competition_data(
        args.stage1, args.stage2, args.stage3, args.final,
        output_dir=os.path.join(tmp_dir, 'processed_data'))

    bootstrap_results = None

    # 2. Advanced analysis and general visualisation
    if 'A' in analyses_to_run:
        print("[A] Running advanced analyses and generating general visualisations...")
        # Perform advanced analysis and save numerical outputs
        advanced_analyzer = run_advanced_analysis(processor, output_dir=os.path.join(tmp_dir, 'advanced_results'))
        # Generate general visualisation to a temporary folder with default numbering
        vis = ChopinVisualization(processor, advanced_analyzer)
        os.makedirs(gen_viz_dir, exist_ok=True)
        vis.create_comprehensive_report(output_dir=gen_viz_dir)

    # 3. Normalisation impact analysis
    if 'N' in analyses_to_run:
        print("[N] Analysing impact of score normalisation...")
        # The analysis function prints and saves its own files
        os.makedirs(norm_dir, exist_ok=True)
        analyze_normalization_impact(save_path = norm_dir)

    # 4. Multistage clustering and advanced visualisation
    if 'M' in analyses_to_run:
        print("[M] Running multistage analysis...")
        data_files = {
            'stage1': args.stage1,
            'stage2': args.stage2,
            'stage3': args.stage3,
            'final': args.final
        }
        multistage_analyzer = run_multistage_analysis(data_files, output_dir=os.path.join(tmp_dir, 'multistage_analysis'))
        os.makedirs(adv_viz_dir, exist_ok=True)
        run_advanced_visualizations(multistage_analyzer, output_dir=adv_viz_dir)

    # 6. Statistical visualisation (including clustering results & PCA) to temp folder
    if 'S' in analyses_to_run:
        print("[S] Generating statistical and clustering visualisations...")

        controversy_analyzer = ChopinControversyAnalyzer(processor)
        stat_analyzer = run_statistical_analysis(processor, output_dir=os.path.join(tmp_dir, 'statistical_results'))
        cluster_analyzer = run_clustering_analysis(processor, output_dir=os.path.join(tmp_dir, 'clustering_results'))

        stat_vis = ChopinStatisticalVisualization(
            processor,
            controversy_analyzer=controversy_analyzer,
            statistical_analyzer=stat_analyzer,
            clustering_analyzer=cluster_analyzer
        )
        os.makedirs(stat_viz_dir, exist_ok=True)
        stat_vis.create_comprehensive_statistical_report(output_dir=stat_viz_dir)

    stage_files = {
        'stage1': args.stage1,
        'stage2': args.stage2,
        'stage3': args.stage3,
        'final': args.final
    }

    # 7. Bootstrap-based final score stability analysis
    if 'B' in analyses_to_run:
        print("[B] Final score stability analysis (bootstrap)...")
        stability_analyzer = FinalScoreStabilityAnalyzer(stage_files)
        bootstrap_results = stability_analyzer.bootstrap_final_scores(n_iterations=10000)
        visualizer = FinalScoreStabilityVisualizer(stability_analyzer)
        os.makedirs(stability_viz_dir, exist_ok=True)
        visualizer.create_full_stability_report(
            bootstrap_results,
            output_dir=stability_viz_dir
        )

    # 8. Score-perturbation-based stability analysis
    if 'P' in analyses_to_run:
        print("[P] Stability analysis via score perturbation...")
        perturbation_analyzer = ScorePerturbationAnalyzer(stage_files, perturbation_values=[-1.0, -0.5, 0.0, 0.5, 1.0])
        perturbation_results = perturbation_analyzer.bootstrap_with_perturbation(n_iterations=10000)
        perturbation_visualizer = PerturbationVisualizer(perturbation_analyzer)
        os.makedirs(stability_viz_dir, exist_ok=True)
        perturbation_visualizer.create_full_report(perturbation_results, output_dir=stability_viz_dir)

    # 9. Consolidate all visualisation into final directory with new naming scheme
    print("[X] Consolidating and renaming visualisations...")
    final_viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    # Define mapping from original filenames (without path) to new prefixed names
    rename_map = {
        '1_distribution_stage1.png': "01_distribution_stage1.png",
        '1_distribution_stage2.png': "02_distribution_stage2.png",
        '1_distribution_stage3.png': "03_distribution_stage3.png",
        '1_distribution_final.png': "04_distribution_final.png",
        '2_scale_usage.png': "05_scale_usage.png",
        '3_judge_tendencies.png': "06_judge_tendencies.png",
        '4_alliances.png': "07_alliances.png",
        '5_removal_impact.png': "08_removal_impact.png",
        '6_qualification_impact_after_removing_judge.png': "09_qualification_impact.png",
        '7_final_results_impact_after_removing_judge.png': "10_final_results_impact.png",
        '8_favorites.png': "11_favorites_analysis.png",
        'normalization_comparison.png': "12_normalisation_comparison.png",
        'normalization_impact.png': "13_normalisation_impact.png",
        '10_participant_pca.png': "14_participant_pca.png",
        '11_judge_pca.png': "15_judge_pca.png",
        '12_multistage_heatmap.png': "16_multistage_heatmap.png",
        '12b_multistage_heatmap_with_changed_margins.png': "16b_multistage_heatmap_with_changed_margins.png",
        '13_judge_consistency.png': "17_judge_consistency.png",
        '14_participant_trajectories.png': "18_participant_trajectories.png",
        '15_cluster_evolution.png': "19_cluster_evolution.png",
        '16_score_diversity_heatmap.png': "20_score_diversity_heatmap.png",
        '17_score_diversity_analysis.png': "21_score_diversity_analysis.png",
        '18_outliers_analysis.png': "22_outliers_analysis.png",
        '21_pairwise_agreement.png': "23_pairwise_agreement.png",
        '22_clustering_results.png': "24_clustering_results.png",
        '23_pca_judges.png': "25_pca_judges.png",
        '34_score_distributions.png': '34_score_distributions.png',
        '35_confidence_intervals.png': '35_confidence_intervals.png',
        '36_ranking_stability_matrix.png': '36_ranking_stability_matrix.png',
        '37_score_vs_uncertainty.png': '37_score_vs_uncertainty.png',
        '38_perturbation_distributions.png': '38_perturbation_distributions.png',
        '39_perturbation_confidence.png': '39_perturbation_confidence.png',
        '40_perturbation_ranking_matrix.png': '40_perturbation_ranking_matrix.png',
        '41_perturbation_score_vs_uncertainty.png': '41_perturbation_score_vs_uncertainty.png',
    }

    temp_dirs = [gen_viz_dir, norm_dir, adv_viz_dir, stat_viz_dir, stability_viz_dir]

    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for fname in os.listdir(temp_dir):
                new_name = rename_map.get(fname)
                if not new_name:
                    # Skip files that are not mapped (e.g., CSVs or intermediate)
                    continue
                src = os.path.join(temp_dir, fname)
                dst = os.path.join(final_viz_dir, new_name)
                shutil.copyfile(src, dst)

    # 10. Copy important CSV summaries to base_dir for convenience
    print("[X] Copying summary CSVs...")
    # Copy processed ranking and advanced results
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

    print("\nAnalysis complete. Visualisations available in:")
    print(f"  {final_viz_dir}")
    print("CSV summaries and results available in:")
    print(f"  {base_dir}\n")


if __name__ == '__main__':
    main()