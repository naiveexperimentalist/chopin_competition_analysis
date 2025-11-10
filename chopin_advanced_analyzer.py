"""
Module for advanced analysis of Chopin Competition 2025 results
Contains analyses: scale usage, normalization, judge tendencies, alliances
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import itertools

class ChopinAdvancedAnalyzer:
    """Advanced competition results analysis"""

    def __init__(self, processor):
        self.processor = processor
        self.judge_columns = processor.judge_columns
        self.stages_data = processor.stages_data
        self.corrected_data = processor.corrected_data

    def get_judge_statistics(self) -> pd.DataFrame:
        """Calculates statistics for each judge"""
        judge_stats = []

        for judge in self.judge_columns:
            judge_stat = {
                'judge': judge,
            }

            for stage_name, df in self.stages_data.items():
                scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                if len(scores) > 0:
                    judge_stat[f'{stage_name}_mean'] = scores.mean()
                    judge_stat[f'{stage_name}_std'] = scores.std()
                    judge_stat[f'{stage_name}_min'] = scores.min()
                    judge_stat[f'{stage_name}_max'] = scores.max()
                    judge_stat[f'{stage_name}_range'] = scores.max() - scores.min()
                    judge_stat[f'{stage_name}_count'] = len(scores)

            judge_stats.append(judge_stat)

        return pd.DataFrame(judge_stats)

    def analyze_scale_usage(self) -> pd.DataFrame:
        """
        Analyzes how judges use the 1-25 scale
        - score range
        - score histogram
        - distribution entropy
        """
        scale_usage = []

        for judge in self.judge_columns:
            judge_info = {'judge': judge}
            all_scores = []

            for stage_name, df in self.stages_data.items():
                scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                if len(scores) > 0:
                    all_scores.extend(scores.tolist())

                    # Stage statistics
                    judge_info[f'{stage_name}_range'] = scores.max() - scores.min()
                    judge_info[f'{stage_name}_unique_scores'] = scores.nunique()

                    # Entropy - measure of score diversity
                    value_counts = scores.value_counts()
                    probabilities = value_counts / len(scores)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    judge_info[f'{stage_name}_entropy'] = entropy

            if all_scores:
                # Overall statistics
                judge_info['overall_range'] = max(all_scores) - min(all_scores)
                judge_info['overall_mean'] = np.mean(all_scores)
                judge_info['overall_std'] = np.std(all_scores)
                judge_info['overall_unique_scores'] = len(set(all_scores))
                judge_info['overall_skewness'] = stats.skew(all_scores)
                judge_info['overall_kurtosis'] = stats.kurtosis(all_scores)

                # Percentage of scale usage (how many unique values out of 25)
                judge_info['scale_coverage'] = (len(set(all_scores)) / 25) * 100

            scale_usage.append(judge_info)

        return pd.DataFrame(scale_usage)

    def normalize_scores(self) -> Dict[str, pd.DataFrame]:
        """
        Normalizes judge scores using different methods:
        1. Z-score normalization (standardization)
        2. Min-max scaling
        3. Rank-based normalization
        """
        normalized_data = {}

        for stage_name, df in self.stages_data.items():
            df_norm = df.copy()

            # Z-score normalization for each judge
            df_zscore = df.copy()
            for judge in self.judge_columns:
                scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~scores.isna()
                if valid_mask.sum() > 0:
                    z_scores = stats.zscore(scores[valid_mask])
                    # Transform z-score to 1-25 scale
                    z_scaled = 13 + (z_scores * 4)  # Center at 13, std ~4
                    z_scaled = np.clip(z_scaled, 1, 25)
                    df_zscore.loc[valid_mask, judge] = z_scaled

            # Min-max scaling
            df_minmax = df.copy()
            for judge in self.judge_columns:
                scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~scores.isna()
                if valid_mask.sum() > 0:
                    min_score = scores[valid_mask].min()
                    max_score = scores[valid_mask].max()
                    if max_score > min_score:
                        scaled = 1 + ((scores[valid_mask] - min_score) / (max_score - min_score)) * 24
                        df_minmax.loc[valid_mask, judge] = scaled

            # Rank-based normalization
            df_rank = df.copy()
            for judge in self.judge_columns:
                scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~scores.isna()
                if valid_mask.sum() > 0:
                    ranks = scores[valid_mask].rank(method='average')
                    # Transform ranks to 1-25 scale
                    rank_scaled = 1 + ((ranks - 1) / (len(ranks) - 1)) * 24 if len(ranks) > 1 else [13] * len(ranks)
                    df_rank.loc[valid_mask, judge] = rank_scaled

            normalized_data[f'{stage_name}_original'] = df
            normalized_data[f'{stage_name}_zscore'] = df_zscore
            normalized_data[f'{stage_name}_minmax'] = df_minmax
            normalized_data[f'{stage_name}_rank'] = df_rank

        return normalized_data

    def analyze_judge_tendencies(self) -> pd.DataFrame:
        """
        Analyzes individual judge tendencies:
        - harshness (average scores vs. overall average)
        - favoring specific participants
        - scoring consistency
        """
        tendencies = []

        for judge in self.judge_columns:
            tendency = {'judge': judge}

            # Collect all judge scores and averages
            judge_scores_all = []
            avg_scores_all = []

            for stage_name, df in self.corrected_data.items():
                judge_scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                stage_avgs = df.loc[judge_scores.index, 'avg_before_correction']

                if len(judge_scores) > 0:
                    judge_scores_all.extend(judge_scores.tolist())
                    avg_scores_all.extend(stage_avgs.tolist())

                    # Average difference from mean (negative = harsh, positive = lenient)
                    diff = judge_scores - stage_avgs
                    tendency[f'{stage_name}_harshness'] = diff.mean()
                    tendency[f'{stage_name}_consistency'] = diff.std()

            if judge_scores_all and avg_scores_all:
                # Overall tendencies
                tendency['overall_harshness'] = np.mean(np.array(judge_scores_all) - np.array(avg_scores_all))
                tendency['overall_consistency'] = np.std(np.array(judge_scores_all) - np.array(avg_scores_all))

                # Correlation with average (high = agrees with consensus)
                tendency['consensus_correlation'] = np.corrcoef(judge_scores_all, avg_scores_all)[0, 1]

            tendencies.append(tendency)

        return pd.DataFrame(tendencies)

    def find_judge_favorites(self, min_stages: int = 3) -> pd.DataFrame:
        """
        Finds participants consistently rated high/low by judges
        (through at least min_stages stages)
        """
        favorites = []

        for judge in self.judge_columns:
            # Track each participant's scores through stages
            participant_tracking = {}

            for stage_name, df in self.corrected_data.items():
                for idx, row in df.iterrows():
                    participant_id = row['number']
                    judge_score = pd.to_numeric(row[judge], errors='coerce')
                    avg_score = row['avg_before_correction']

                    if pd.notna(judge_score) and pd.notna(avg_score):
                        if participant_id not in participant_tracking:
                            participant_tracking[participant_id] = {
                                'name': f"{row['firstname']} {row['lastname']}",
                                'scores': [],
                                'differences': []
                            }

                        participant_tracking[participant_id]['scores'].append(judge_score)
                        participant_tracking[participant_id]['differences'].append(judge_score - avg_score)

            # Analyze consistent tendencies
            for participant_id, data in participant_tracking.items():
                if len(data['differences']) >= min_stages:
                    avg_diff = np.mean(data['differences'])
                    consistency = np.std(data['differences'])

                    # If consistently high (>1) or low (<-1) with small variance
                    if abs(avg_diff) > 1 and consistency < 2:
                        favorites.append({
                            'judge': judge,
                            'participant_nr': participant_id,
                            'participant_name': data['name'],
                            'avg_difference': avg_diff,
                            'consistency': consistency,
                            'n_stages': len(data['differences']),
                            'type': 'favorite' if avg_diff > 0 else 'unfavorite'
                        })

        return pd.DataFrame(favorites)

    def simulate_judge_removal(self) -> pd.DataFrame:
        """
        Simulates removal of each judge and checks impact on final results
        """
        results = []

        # Baseline - results with all judges
        baseline_final = self.processor.cumulative_scores.get('final_cumulative')
        if baseline_final is None:
            print("No final results available for analysis")
            return pd.DataFrame()

        baseline_ranks = dict(zip(baseline_final['number'], baseline_final['rank']))

        # For each judge
        for judge_to_remove in self.judge_columns:
            try:
                # Create temporary processor without this judge
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Copy data without judge column
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Set judge_columns without removed judge
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Recalculate results
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                # Compare rankings
                if 'final_cumulative' in temp_processor.cumulative_scores:
                    new_final = temp_processor.cumulative_scores['final_cumulative']
                    new_ranks = dict(zip(new_final['number'], new_final['rank']))

                    # Calculate ranking changes
                    rank_changes = []
                    for participant_nr in baseline_ranks:
                        if participant_nr in new_ranks:
                            change = baseline_ranks[participant_nr] - new_ranks[participant_nr]
                            rank_changes.append(abs(change))

                    if rank_changes:  # Check if list is not empty
                        results.append({
                            'judge_removed': judge_to_remove,
                            'avg_rank_change': np.mean(rank_changes),
                            'max_rank_change': max(rank_changes),
                            'participants_affected': sum(1 for c in rank_changes if c > 0),
                            'top10_changes': sum(1 for nr in baseline_ranks
                                                 if baseline_ranks[nr] <= 10 and nr in new_ranks
                                                 and baseline_ranks[nr] != new_ranks[nr])
                        })
            except Exception as e:
                print(f"  Error when removing judge {judge_to_remove}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def analyze_qualification_after_judge_removal(self) -> pd.DataFrame:
        """
        Simulates removal of each judge and checks impact on participant qualification to next rounds
        """
        results = []

        # Baseline - results of subsequent rounds with all judges
        baseline_stage1 = self.processor.cumulative_scores.get('stage1_cumulative')
        baseline_stage2 = self.processor.cumulative_scores.get('stage2_cumulative')
        baseline_stage3 = self.processor.cumulative_scores.get('stage3_cumulative')
        baseline_final = self.processor.cumulative_scores.get('final_cumulative')
        if baseline_final is None:
            print("No final results available for analysis")
            return pd.DataFrame()

        baseline_stage1_ranks = dict(zip(baseline_stage1['number'], baseline_stage1['rank']))
        baseline_stage2_ranks = dict(zip(baseline_stage2['number'], baseline_stage2['rank']))
        baseline_stage3_ranks = dict(zip(baseline_stage3['number'], baseline_stage3['rank']))
        baseline_final_ranks = dict(zip(baseline_final['number'], baseline_final['rank']))

        QUALIFICATION_THRESHOLDS = {
            'stage1': 40,  # top 40 advance to stage2
            'stage2': 20,  # top 20 advance to stage3
            'stage3': 10   # top 10 advance to final
        }

        # For each judge
        for judge_to_remove in self.judge_columns:
            try:
                # Create temporary processor without this judge
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Copy data without judge column
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Set judge_columns without removed judge
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Recalculate results
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                new_stage1 = temp_processor.cumulative_scores['stage1_cumulative']
                new_stage2 = temp_processor.cumulative_scores['stage2_cumulative']
                new_stage3 = temp_processor.cumulative_scores['stage3_cumulative']
                new_final = temp_processor.cumulative_scores['final_cumulative']

                # Qualification analysis for each stage
                qualification_changes = {}

                # Stage 1 -> Stage 2
                stage1_baseline_qualified = set(
                    baseline_stage1[baseline_stage1['rank'] <= QUALIFICATION_THRESHOLDS['stage1']]['number'])
                stage1_new_qualified = set(new_stage1[new_stage1['rank'] <= QUALIFICATION_THRESHOLDS['stage1']]['number'])
                qualification_changes['stage1_to_stage2'] = {
                    'lost_qualification': list(stage1_baseline_qualified - stage1_new_qualified),
                    'gained_qualification': list(stage1_new_qualified - stage1_baseline_qualified)
                }

                # Stage 2 -> Stage 3
                stage2_baseline_qualified = set(
                    baseline_stage2[baseline_stage2['rank'] <= QUALIFICATION_THRESHOLDS['stage2']]['number'])
                stage2_new_qualified = set(new_stage2[new_stage2['rank'] <= QUALIFICATION_THRESHOLDS['stage2']]['number'])
                qualification_changes['stage2_to_stage3'] = {
                    'lost_qualification': list(stage2_baseline_qualified - stage2_new_qualified),
                    'gained_qualification': list(stage2_new_qualified - stage2_baseline_qualified)
                }

                # Stage 3 -> Final
                stage3_baseline_qualified = set(
                    baseline_stage3[baseline_stage3['rank'] <= QUALIFICATION_THRESHOLDS['stage3']]['number'])
                stage3_new_qualified = set(new_stage3[new_stage3['rank'] <= QUALIFICATION_THRESHOLDS['stage3']]['number'])
                qualification_changes['stage3_to_final'] = {
                    'lost_qualification': list(stage3_baseline_qualified - stage3_new_qualified),
                    'gained_qualification': list(stage3_new_qualified - stage3_baseline_qualified)
                }

                results.append({
                    'judge_removed': judge_to_remove,
                    **qualification_changes
                })
            except Exception as e:
                print(f"  Error when removing judge {judge_to_remove}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def generate_results_after_judge_removal(self) -> pd.DataFrame:
        """
        Simulates full competition (from stage1 to final) after removing each judge.
        Takes into account that a participant may not qualify for stage3 or final.

        Args:
            num_qualifiers_stage3: number of participants advancing to stage3 (default 20)
            num_qualifiers_final: number of participants advancing to final (default 10)

        Returns:
            DataFrame with columns:
            - number: participant number
            - firstname, lastname: participant data
            - original_rank: original final ranking
            - [judge_name]_rank: ranking after removing given judge (or 'n/a' if wouldn't advance to final)
            - [judge_name]_change: position change (positive = improvement, negative = decline, 'n/a' = disqualification)
        """
        QUALIFICATION_THRESHOLDS = {
            'stage1': 40,  # top 40 advance to stage2
            'stage2': 20,  # top 20 advance to stage3
            'stage3': 10,   # top 10 advance to final
            'real_stage3': 11  # top 11 advanced to final
        }

        # Get original final results
        orig_finalists = self.processor.cumulative_scores.get('final_cumulative')

        # Prepare result DataFrame
        result_df = orig_finalists[['number', 'firstname', 'lastname', 'rank']].copy()
        result_df = result_df.rename(columns={'rank': 'original_rank'})
        result_df['original_rank'] = result_df['original_rank'].astype(int)

        print(f"\nFinalists in original scenario (top {QUALIFICATION_THRESHOLDS['stage3']}):")
        for _, row in orig_finalists.iterrows():
            print(f"  {int(row['rank'])}: No. {row['number']} - {row['firstname']} {row['lastname']}")

        # For each judge simulate full competition
        for judge_to_remove in self.judge_columns:
            try:
                # Create temporary processor without this judge
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Copy data without judge column
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Set judge_columns without removed judge
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Recalculate results
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                exclude_from_finals = set()

                # Check who would advance to stage2
                new_stage1 = temp_processor.cumulative_scores.get('stage1_cumulative')
                stage2_qualifiers = set(
                    new_stage1.nsmallest(QUALIFICATION_THRESHOLDS['stage1'], 'rank')['number'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if row['number'] not in stage2_qualifiers:
                        exclude_from_finals.add(int(row['number']))

                # Check who would advance to stage3
                new_stage2 = temp_processor.cumulative_scores.get('stage2_cumulative')
                stage3_qualifiers = set(
                    new_stage2.nsmallest(QUALIFICATION_THRESHOLDS['stage2'], 'rank')['number'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if row['number'] not in stage3_qualifiers:
                        exclude_from_finals.add(int(row['number']))

                # Check who would advance to final
                new_stage3 = temp_processor.cumulative_scores.get('stage3_cumulative')
                final_qualifiers = set(
                    new_stage3.nsmallest(QUALIFICATION_THRESHOLDS['stage3'], 'rank')['number'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if row['number'] not in final_qualifiers:
                        exclude_from_finals.add(int(row['number']))

                temp_processor.calculate_cumulative_scores(exclude_from_finals)

                # Get final results
                new_final = temp_processor.cumulative_scores.get('final_cumulative')

                # Set column names
                judge_col_name = f"{judge_to_remove}_rank"
                change_col_name = f"{judge_to_remove}_change"

                result_df = result_df.merge(new_final[['number', 'rank']].rename(columns={'rank': judge_col_name}), on='number', how='left')

                result_df[judge_col_name] = result_df[judge_col_name].replace(999, 'n/a')

                result_df[change_col_name] = result_df.apply(
                    lambda row: 'n/a' if row[judge_col_name] == 'n/a'
                    else int(row['original_rank']) - int(row[judge_col_name]),
                    axis=1
                )
            except Exception as e:
                print(f"  ✗ Error when removing judge {judge_to_remove}: {str(e)}")
                # Add empty columns to maintain structure
                judge_col_name = f"{judge_to_remove}_rank"
                change_col_name = f"{judge_to_remove}_change"
                result_df[judge_col_name] = 'error'
                result_df[change_col_name] = 'error'
                continue

        print(f"\n{'=' * 60}")
        print("SIMULATION COMPLETED")
        print(f"{'=' * 60}")

        return result_df

    def analyze_judge_alliances(self, threshold: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes potential alliances between judges based on score correlations
        """
        # Collect all scores for each judge
        all_scores_by_judge = {judge: [] for judge in self.judge_columns}
        participant_stage_keys = []

        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                key = f"{row['number']}_{stage_name}"
                participant_stage_keys.append(key)

                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    all_scores_by_judge[judge].append(score)

        # Create DataFrame with all judge scores
        scores_df = pd.DataFrame(all_scores_by_judge)

        # Calculate correlations between judges
        correlation_matrix = scores_df.corr(method='pearson')

        # Find pairs with high correlation
        alliances = []
        for i, judge1 in enumerate(self.judge_columns):
            for judge2 in self.judge_columns[i+1:]:
                corr = correlation_matrix.loc[judge1, judge2]
                if pd.notna(corr) and corr > threshold:
                    alliances.append({
                        'judge1': judge1,
                        'judge2': judge2,
                        'correlation': corr,
                        'strength': 'strong' if corr > 0.8 else 'moderate'
                    })

        alliances_df = pd.DataFrame(alliances)
        if not alliances_df.empty:
            alliances_df = alliances_df.sort_values('correlation', ascending=False)

        return correlation_matrix, alliances_df

    def analyze_national_bias(self, nationality_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Analyzes potential national biases
        Requires providing mapping judge -> nationality
        """
        if nationality_mapping is None:
            # Example mapping - should be filled with actual data
            print("No judge nationality data. Provide nationality_mapping dictionary.")
            return pd.DataFrame()

        bias_results = []

        # For each judge and nationality
        for judge in self.judge_columns:
            if judge not in nationality_mapping:
                continue

            judge_nationality = nationality_mapping[judge]

            # Analyze scores for participants from different countries
            # (assuming we have participant nationality information)
            # This requires additional data...

            bias_results.append({
                'judge': judge,
                'nationality': judge_nationality,
                # Add bias analysis when participant nationality data available
            })

        return pd.DataFrame(bias_results)

    def calculate_judge_independence(self) -> pd.DataFrame:
        """
        Calculates degree of each judge's independence from consensus
        """
        independence_scores = []

        for judge in self.judge_columns:
            judge_data = {'judge': judge}

            # Collect all cases where judge differed from average
            total_deviations = []
            total_corrections = 0

            for stage_name, df in self.corrected_data.items():
                judge_scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~judge_scores.isna()

                if valid_mask.sum() > 0:
                    # Differences from average before correction
                    avg_scores = df.loc[valid_mask, 'avg_before_correction']
                    deviations = abs(judge_scores[valid_mask] - avg_scores)
                    total_deviations.extend(deviations.tolist())

                    # How many times scores were corrected
                    stage_corrections = df.loc[valid_mask, 'corrections_made'].sum()
                    total_corrections += stage_corrections

            if total_deviations:
                judge_data['avg_deviation'] = np.mean(total_deviations)
                judge_data['max_deviation'] = max(total_deviations)
                judge_data['independence_score'] = np.percentile(total_deviations, 75)  # 75th percentile of deviations
                judge_data['times_corrected'] = total_corrections

            independence_scores.append(judge_data)

        return pd.DataFrame(independence_scores)


# Helper functions for analyses
def run_advanced_analysis(processor, output_dir: str = 'advanced_results'):
    """Runs all advanced analyses"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    analyzer = ChopinAdvancedAnalyzer(processor)

    # 1. Scale usage analysis
    print("Analyzing 1-25 scale usage...")
    scale_usage = analyzer.analyze_scale_usage()
    scale_usage.to_csv(f'{output_dir}/scale_usage_analysis.csv', index=False)

    # 2. Score normalization
    print("Normalizing scores...")
    normalized = analyzer.normalize_scores()
    # Save selected normalizations
    for key, df in normalized.items():
        if 'zscore' in key:
            df.to_csv(f'{output_dir}/{key}.csv', index=False)

    # 3. Judge tendencies
    print("Analyzing judge tendencies...")
    tendencies = analyzer.analyze_judge_tendencies()
    tendencies.to_csv(f'{output_dir}/judge_tendencies.csv', index=False)

    # 4. Judge favorites
    print("Finding favorites...")
    favorites = analyzer.find_judge_favorites()
    if not favorites.empty:
        favorites.to_csv(f'{output_dir}/judge_favorites.csv', index=False)

    # 5. Judge removal simulation
    print("Simulating judge removal...")
    removal_impact = analyzer.simulate_judge_removal()
    if not removal_impact.empty:
        removal_impact.to_csv(f'{output_dir}/judge_removal_impact.csv', index=False)

    # 5.3. Qualifications to next round without one judge
    print("Qualifications to next round without one judge...")
    qualification_with_removed_judge = analyzer.analyze_qualification_after_judge_removal()
    if not qualification_with_removed_judge.empty:
        qualification_with_removed_judge.to_csv(f'{output_dir}/qualification_with_removed_judge.csv', index=False)

    # 5.5. Competition results without one judge
    print("Competition results without one judge...")
    results_with_removed_judge = analyzer.generate_results_after_judge_removal()
    if not results_with_removed_judge.empty:
        results_with_removed_judge.to_csv(f'{output_dir}/results_after_judge_removal.csv', index=False)

    # 6. Judge alliances
    print("Analyzing alliances...")
    correlation_matrix, alliances = analyzer.analyze_judge_alliances()
    correlation_matrix.to_csv(f'{output_dir}/judge_correlations.csv')
    if not alliances.empty:
        alliances.to_csv(f'{output_dir}/judge_alliances.csv', index=False)
    else:
        print("  No strong alliances found (correlation > 0.6)")

    # 7. Judge independence
    print("Calculating independence...")
    independence = analyzer.calculate_judge_independence()
    independence.to_csv(f'{output_dir}/judge_independence.csv', index=False)

    print(f"\nAnalyses saved in directory: {output_dir}")

    return analyzer


if __name__ == "__main__":
    # Example usage
    print("Run chopin_data_processor.py first, then this script with processor object")