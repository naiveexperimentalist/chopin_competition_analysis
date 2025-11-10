"""\nChopin Competition — score variability and outlier analysis.

This module computes per-contestant score variance, ranges, coefficients
of variation, and detects extreme judge deviations (outliers) per stage.
It exposes helper tables for visual heatmaps and summary plots.\n"""

import pandas as pd
import numpy as np


class ChopinControversyAnalyzer:
    """Analyzes the controversy per contestant"""
    
    def __init__(self, processor):
        self.processor = processor
        self.judge_columns = processor.judge_columns
        self.stages_data = processor.stages_data
        self.corrected_data = processor.corrected_data
    
    def analyze_participant_controversy(self) -> pd.DataFrame:
        """Analyzes the controversy per contestant"""
        controversy_data = []
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                participant_nr = row['number']
                participant_name = f"{row['firstname']} {row['lastname']}"
                
                scores = []
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    if pd.notna(score):
                        scores.append(score)
                
                    scores_array = np.array(scores)
                    
                    mean_score = np.mean(scores_array)
                    std_score = np.std(scores_array)
                    cv = (std_score / mean_score * 100) if mean_score > 0 else 0
                    
                    q1 = np.percentile(scores_array, 25)
                    q3 = np.percentile(scores_array, 75)
                    iqr = q3 - q1
                    
                    controversy_data.append({
                        'stage': stage_name,
                        'number': participant_nr,
                        'firstname': row['firstname'],
                        'lastname': row['lastname'],
                        'participant_name': participant_name,
                        'mean': mean_score,
                        'std': std_score,
                        'cv': cv,
                        'min': scores_array.min(),
                        'max': scores_array.max(),
                        'range': scores_array.max() - scores_array.min(),
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'n_judges': len(scores)
                    })
        
        controversy_df = pd.DataFrame(controversy_data)
        
        if not controversy_df.empty:
            for stage in controversy_df['stage'].unique():
                stage_mask = controversy_df['stage'] == stage
                controversy_df.loc[stage_mask, 'diversity_rank'] =\
                    controversy_df.loc[stage_mask, 'std'].rank(ascending=False)
            
            participant_overall = controversy_df.groupby('number').agg({
                'std': "mean",
                'range': "mean",
                'cv': "mean"
            }).reset_index()
            
            participant_overall.columns = ['number', 'avg_std', 'avg_range', 'avg_cv']
            participant_overall['overall_controversy_rank'] =\
                participant_overall['avg_std'].rank(ascending=False)
            
            controversy_df = controversy_df.merge(participant_overall, on='number', how='left')
        
        return controversy_df
    
    def find_most_controversial_participants(self, top_n: int = 10) -> pd.DataFrame:
        """Analyzes the controversy per contestant"""
        controversy = self.analyze_participant_controversy()
        
        if controversy.empty:
            return pd.DataFrame()
        
        latest_stage = controversy.sort_values('stage', ascending=False).groupby('number').first()
        
        # Analyzes the controversy per contestant
        most_controversial = latest_stage.nlargest(top_n, 'avg_std')[
            ['participant_name', 'stage', 'mean', 'std', 'range', 'cv', 'n_judges']
        ].reset_index()
        
        return most_controversial
    
    def create_controversy_heatmap_data(self, stage: str = None) -> pd.DataFrame:
        if stage:
            stages_to_process = [stage] if stage in self.stages_data else []
        else:
            stages_to_process = [list(self.stages_data.keys())[-1]]
        
        if not stages_to_process:
            print(f"No data for the stage: {stage}")
            return pd.DataFrame()
        
        stage_name = stages_to_process[0]
        df = self.stages_data[stage_name]
        
        participants = []
        heatmap_data = []
        
        for idx, row in df.iterrows():
            participant_label = f"{row['firstname']}\n{row['lastname']}"
            participants.append(participant_label)
            
            scores = []
            for judge in self.judge_columns:
                score = pd.to_numeric(row[judge], errors='coerce')
                scores.append(score)
            
            valid_scores = [s for s in scores if pd.notna(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                deviations = [s - mean_score if pd.notna(s) else np.nan for s in scores]
                heatmap_data.append(deviations)
            else:
                heatmap_data.append([np.nan] * len(self.judge_columns))
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                   index=participants,
                                   columns=self.judge_columns)
        
        return heatmap_df
    
    def analyze_outliers(self) -> pd.DataFrame:
        outliers = []
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                participant_nr = row['number']
                participant_name = f"{row['firstname']} {row['lastname']}"
                
                # Zbierz score
                scores = []
                judge_names = []
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    if pd.notna(score):
                        scores.append(score)
                        judge_names.append(judge)
                
                if len(scores) >= 4:  # Potrzeba przynajmniej 4 score
                    scores_array = np.array(scores)
                    
                    q1 = np.percentile(scores_array, 25)
                    q3 = np.percentile(scores_array, 75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    mean_score = np.mean(scores_array)
                    
                    for judge, score in zip(judge_names, scores):
                        if score < lower_bound or score > upper_bound:
                            outliers.append({
                                'stage': stage_name,
                                'number': participant_nr,
                                'participant_name': participant_name,
                                'judge': judge,
                                'score': score,
                                'mean': mean_score,
                                'deviation': score - mean_score,
                                'q1': q1,
                                'q3': q3,
                                'iqr': iqr,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound,
                                'outlier_type': "High" if score > upper_bound else 'low'
                            })
        
        outliers_df = pd.DataFrame(outliers)
        
        if not outliers_df.empty:
            outliers_df['extremeness'] = outliers_df['deviation'].abs()
            outliers_df = outliers_df.sort_values('extremeness', ascending=False)
        
        return outliers_df
    
    def get_outlier_statistics_by_judge(self) -> pd.DataFrame:
        outliers = self.analyze_outliers()
        
        if outliers.empty:
            return pd.DataFrame()
        
        judge_stats = []
        
        for judge in self.judge_columns:
            judge_outliers = outliers[outliers['judge'] == judge]
            
            if len(judge_outliers) > 0:
                high_outliers = len(judge_outliers[judge_outliers['outlier_type'] == 'high'])
                low_outliers = len(judge_outliers[judge_outliers['outlier_type'] == 'low'])
                
                judge_stats.append({
                    'judge': judge,
                    'total_outliers': len(judge_outliers),
                    'high_outliers': high_outliers,
                    'low_outliers': low_outliers,
                    'avg_deviation': judge_outliers['deviation'].abs().mean(),
                    'max_deviation': judge_outliers['deviation'].abs().max()
                })
        
        stats_df = pd.DataFrame(judge_stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('total_outliers', ascending=False)
        
        return stats_df
    
    def analyze_agreement_vs_controversy(self) -> pd.DataFrame:
        """Analyzes the controversy per contestant"""
        controversy = self.analyze_participant_controversy()
        
        if controversy.empty:
            return pd.DataFrame()
        
        agreement_data = []
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                participant_nr = row['number']
                
                scores = []
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    if pd.notna(score):
                        scores.append(score)
                
                if len(scores) >= 3:
                    controversy_row = controversy[
                        (controversy['number'] == participant_nr) & 
                        (controversy['stage'] == stage_name)
                    ]
                    
                    if not controversy_row.empty:
                        std = controversy_row.iloc[0]['std']
                        cv = controversy_row.iloc[0]['cv']
                        range_val = controversy_row.iloc[0]['range']
                        
                        agreement_data.append({
                            'stage': stage_name,
                            'number': participant_nr,
                            'participant_name': controversy_row.iloc[0]['participant_name'],
                            'std': std,
                            'cv': cv,
                            'range': range_val,
                            'n_scores': len(scores)
                        })
        
        return pd.DataFrame(agreement_data)


def run_controversy_analysis(processor, output_dir: str = 'score_diversity_results'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ChopinControversyAnalyzer(processor)
    
    print("analyze_participant_controversy…")
    controversy = analyzer.analyze_participant_controversy()
    controversy.to_csv(f'{output_dir}/participant_score_diversity.csv', index=False)
    
    print("find_most_controversial_participants…")
    most_controversial = analyzer.find_most_controversial_participants(top_n=20)
    if not most_controversial.empty:
        most_controversial.to_csv(f'{output_dir}/most_diverse_scores.csv', index=False)
    
    # 3. Dane do heatmap
    print("create_controversy_heatmap_data…")
    for stage in processor.stages_data.keys():
        heatmap_data = analyzer.create_controversy_heatmap_data(stage=stage)
        if not heatmap_data.empty:
            heatmap_data.to_csv(f'{output_dir}/heatmap_{stage}.csv')
    
    print("analyze_outliers…")
    outliers = analyzer.analyze_outliers()
    if not outliers.empty:
        outliers.to_csv(f'{output_dir}/outliers.csv', index=False)
    
    print("get_outlier_statistics_by_judge…")
    judge_outlier_stats = analyzer.get_outlier_statistics_by_judge()
    if not judge_outlier_stats.empty:
        judge_outlier_stats.to_csv(f'{output_dir}/judge_outlier_stats.csv', index=False)
    
    # Analyzes the controversy per contestant
    print("analyze_agreement_vs_controversy…")
    agreement_vs_controversy = analyzer.analyze_agreement_vs_controversy()
    if not agreement_vs_controversy.empty:
        agreement_vs_controversy.to_csv(f'{output_dir}/agreement_vs_controversy.csv', index=False)
    
    print(f"\nSaved in: {output_dir}")
    
    return analyzer
