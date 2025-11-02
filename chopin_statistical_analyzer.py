"""
Moduł do zaawansowanych analiz statystycznych konkursu Chopinowskiego
- Bootstrap confidence intervals dla stabilności rankingu
- Monte Carlo simulations
- Statistical significance testing
- Pair-wise agreement (Kendall's tau, Cohen's kappa)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ChopinStatisticalAnalyzer:
    """Zaawansowane analizy statystyczne"""
    
    def __init__(self, processor):
        self.processor = processor
        self.judge_columns = processor.judge_columns
        self.stages_data = processor.stages_data
        self.corrected_data = processor.corrected_data
    
    def bootstrap_ranking_confidence(self, stage: str = 'final', n_bootstrap: int = 1000, 
                                    confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Bootstrap confidence intervals dla rankingu
        Pokazuje jak stabilne są miejsca w rankingu
        
        Args:
            stage: etap do analizy ('stage1', 'stage2', 'stage3', 'final')
            n_bootstrap: liczba iteracji bootstrap
            confidence_level: poziom ufności (np. 0.95 dla 95% CI)
        """
        if stage not in self.corrected_data:
            print(f"Brak danych dla etapu: {stage}")
            return pd.DataFrame()
        
        df = self.corrected_data[stage]
        
        results = []
        alpha = 1 - confidence_level
        
        for idx, row in df.iterrows():
            participant_nr = row['Nr']
            participant_name = f"{row['imię']} {row['nazwisko']}"
            
            # Zbierz oceny od wszystkich sędziów
            scores = []
            for judge in self.judge_columns:
                score = pd.to_numeric(row[judge], errors='coerce')
                if pd.notna(score):
                    scores.append(score)
            
            if len(scores) < 3:
                continue
            
            scores_array = np.array(scores)
            original_mean = np.mean(scores_array)
            
            # Bootstrap
            bootstrap_means = []
            for _ in range(n_bootstrap):
                # Losuj z powtórzeniami
                bootstrap_sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Oblicz confidence interval
            ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
            ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
            
            results.append({
                'Nr': participant_nr,
                'imię': row['imię'],
                'nazwisko': row['nazwisko'],
                'participant_name': participant_name,
                'original_score': original_mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'std_bootstrap': np.std(bootstrap_means),
                'n_judges': len(scores)
            })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Oblicz ranking dla oryginalnych wyników
            results_df['rank'] = results_df['original_score'].rank(ascending=False, method='min')
            
            # Oblicz możliwe zakresy rankingu (bootstrap rankings)
            # Symuluj rankingi używając granic CI
            results_df['rank_ci_lower'] = results_df['ci_upper'].rank(ascending=False, method='min')
            results_df['rank_ci_upper'] = results_df['ci_lower'].rank(ascending=False, method='min')
            results_df['rank_uncertainty'] = results_df['rank_ci_upper'] - results_df['rank_ci_lower']
            
            results_df = results_df.sort_values('rank')
        
        return results_df
    
    def monte_carlo_ranking_stability(self, stage: str = 'final', n_simulations: int = 1000,
                                     noise_level: float = 0.1) -> pd.DataFrame:
        """
        Monte Carlo simulations - dodaje szum do ocen i sprawdza stabilność rankingu
        
        Args:
            stage: etap do analizy
            n_simulations: liczba symulacji
            noise_level: poziom szumu jako proporcja SD (0.1 = 10% SD)
        """
        if stage not in self.corrected_data:
            print(f"Brak danych dla etapu: {stage}")
            return pd.DataFrame()
        
        df = self.corrected_data[stage]
        
        # Przygotuj macierz ocen
        participants = []
        scores_matrix = []
        
        for idx, row in df.iterrows():
            participants.append({
                'Nr': row['Nr'],
                'imię': row['imię'],
                'nazwisko': row['nazwisko']
            })
            
            scores = []
            for judge in self.judge_columns:
                score = pd.to_numeric(row[judge], errors='coerce')
                scores.append(score if pd.notna(score) else np.nan)
            
            scores_matrix.append(scores)
        
        scores_matrix = np.array(scores_matrix)
        
        # Oryginalny ranking
        original_means = np.nanmean(scores_matrix, axis=1)
        original_ranks = rankdata(-original_means, method='min')
        
        # Monte Carlo simulations
        rank_distributions = [[] for _ in range(len(participants))]
        
        for _ in range(n_simulations):
            # Dodaj szum Gaussowski
            noisy_scores = scores_matrix.copy()
            for i in range(len(participants)):
                valid_scores = scores_matrix[i][~np.isnan(scores_matrix[i])]
                if len(valid_scores) > 0:
                    score_std = np.std(valid_scores)
                    noise = np.random.normal(0, score_std * noise_level, size=len(scores_matrix[i]))
                    noisy_scores[i] = np.where(~np.isnan(scores_matrix[i]), 
                                               scores_matrix[i] + noise, 
                                               np.nan)
            
            # Oblicz ranking dla zaszumionych danych
            noisy_means = np.nanmean(noisy_scores, axis=1)
            noisy_ranks = rankdata(-noisy_means, method='min')
            
            for i, rank in enumerate(noisy_ranks):
                rank_distributions[i].append(rank)
        
        # Przygotuj wyniki
        results = []
        for i, participant in enumerate(participants):
            rank_dist = np.array(rank_distributions[i])
            
            results.append({
                'Nr': participant['Nr'],
                'imię': participant['imię'],
                'nazwisko': participant['nazwisko'],
                'participant_name': f"{participant['imię']} {participant['nazwisko']}",
                'original_rank': int(original_ranks[i]),
                'mean_rank_mc': np.mean(rank_dist),
                'std_rank_mc': np.std(rank_dist),
                'min_rank_mc': int(np.min(rank_dist)),
                'max_rank_mc': int(np.max(rank_dist)),
                'rank_range_mc': int(np.max(rank_dist) - np.min(rank_dist)),
                'rank_95_lower': int(np.percentile(rank_dist, 2.5)),
                'rank_95_upper': int(np.percentile(rank_dist, 97.5))
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('original_rank')
        
        return results_df
    
    def statistical_significance_between_ranks(self, stage: str = 'final') -> pd.DataFrame:
        """
        Testuje czy różnice między sąsiednimi miejscami są statystycznie istotne
        Używa t-testu dla niezależnych prób
        """
        if stage not in self.corrected_data:
            print(f"Brak danych dla etapu: {stage}")
            return pd.DataFrame()
        
        df = self.corrected_data[stage].copy()
        
        # Dodaj ranking
        df['rank'] = df['stage_score'].rank(ascending=False, method='min')
        df = df.sort_values('rank')
        
        # Zbierz oceny dla każdego uczestnika
        participant_scores = {}
        for idx, row in df.iterrows():
            scores = []
            for judge in self.judge_columns:
                score = pd.to_numeric(row[judge], errors='coerce')
                if pd.notna(score):
                    scores.append(score)
            
            if len(scores) >= 3:
                participant_scores[row['Nr']] = np.array(scores)
        
        # Testuj różnice między sąsiednimi miejscami (pomijając ex aequo)
        significance_results = []
        
        df_sorted = df.sort_values('rank').reset_index(drop=True)
        
        for i in range(len(df_sorted) - 1):
            rank1 = df_sorted.iloc[i]['rank']
            rank2 = df_sorted.iloc[i+1]['rank']
            
            # Pomiń pary z tym samym miejscem (ex aequo)
            if rank1 == rank2:
                continue
            
            nr1 = df_sorted.iloc[i]['Nr']
            nr2 = df_sorted.iloc[i+1]['Nr']
            
            if nr1 in participant_scores and nr2 in participant_scores:
                scores1 = participant_scores[nr1]
                scores2 = participant_scores[nr2]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                
                significance_results.append({
                    'rank1': int(df_sorted.iloc[i]['rank']),
                    'participant1': f"{df_sorted.iloc[i]['imię']} {df_sorted.iloc[i]['nazwisko']}",
                    'score1': df_sorted.iloc[i]['stage_score'],
                    'rank2': int(df_sorted.iloc[i+1]['rank']),
                    'participant2': f"{df_sorted.iloc[i+1]['imię']} {df_sorted.iloc[i+1]['nazwisko']}",
                    'score2': df_sorted.iloc[i+1]['stage_score'],
                    'score_diff': df_sorted.iloc[i]['stage_score'] - df_sorted.iloc[i+1]['stage_score'],
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_05': p_value < 0.05,
                    'significant_01': p_value < 0.01,
                    'cohens_d': cohens_d,
                    'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
                })
        
        return pd.DataFrame(significance_results)
    
    def identify_statistical_ties(self, stage: str = 'final', alpha: float = 0.05) -> pd.DataFrame:
        """
        Identyfikuje grupy uczestników które są statystycznie nierozróżnialne (tie)
        Uwzględnia również ex aequo z oryginalnego rankingu
        """
        if stage not in self.corrected_data:
            print(f"Brak danych dla etapu: {stage}")
            return pd.DataFrame()
        
        df = self.corrected_data[stage].copy()
        df['rank'] = df['stage_score'].rank(ascending=False, method='min')
        df = df.sort_values('rank')
        
        # Zbierz oceny
        participant_data = {}
        for idx, row in df.iterrows():
            scores = []
            for judge in self.judge_columns:
                score = pd.to_numeric(row[judge], errors='coerce')
                if pd.notna(score):
                    scores.append(score)
            
            if len(scores) >= 3:
                participant_data[row['Nr']] = {
                    'scores': np.array(scores),
                    'rank': row['rank'],
                    'name': f"{row['imię']} {row['nazwisko']}",
                    'score': row['stage_score']
                }
        
        # Znajdź ties
        ties = []
        processed = set()
        
        participants = sorted(participant_data.keys(), 
                            key=lambda x: participant_data[x]['rank'])
        
        for i, nr1 in enumerate(participants):
            if nr1 in processed:
                continue
            
            tie_group = [nr1]
            rank1 = participant_data[nr1]['rank']
            
            for nr2 in participants[i+1:]:
                if nr2 in processed:
                    continue
                
                rank2 = participant_data[nr2]['rank']
                
                # Jeśli mają ten sam rank (ex aequo) - automatycznie tie
                if rank1 == rank2:
                    tie_group.append(nr2)
                    processed.add(nr2)
                    continue
                
                # Jeśli różne ranki - testuj statystycznie
                scores1 = participant_data[nr1]['scores']
                scores2 = participant_data[nr2]['scores']
                
                # T-test
                _, p_value = stats.ttest_ind(scores1, scores2)
                
                if p_value >= alpha:  # Nie ma istotnej różnicy
                    tie_group.append(nr2)
                    processed.add(nr2)
            
            if len(tie_group) > 1 or nr1 not in processed:
                # Pobierz unikalne ranki w grupie
                ranks_in_group = sorted(set([participant_data[nr]['rank'] for nr in tie_group]))
                
                tie_info = {
                    'tie_group_size': len(tie_group),
                    'ranks': ', '.join([str(int(r)) for r in ranks_in_group]),
                    'participants': ' / '.join([participant_data[nr]['name'] for nr in tie_group]),
                    'scores': ', '.join([f"{participant_data[nr]['score']:.2f}" for nr in tie_group]),
                    'score_range': max([participant_data[nr]['score'] for nr in tie_group]) - 
                                  min([participant_data[nr]['score'] for nr in tie_group])
                }
                ties.append(tie_info)
                processed.add(nr1)
        
        return pd.DataFrame(ties)
    
    def pairwise_agreement_kendall(self) -> pd.DataFrame:
        """
        Oblicza Kendall's tau dla każdej pary sędziów
        Pokazuje zgodność w rankingach (nie tylko korelację liniową)
        """
        # Zbierz rankingi od każdego sędziego dla wszystkich uczestników
        judge_rankings = {judge: [] for judge in self.judge_columns}
        participant_keys = []
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                key = f"{row['Nr']}_{stage_name}"
                participant_keys.append(key)
                
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    judge_rankings[judge].append(score)
        
        # Oblicz Kendall's tau dla każdej pary
        results = []
        
        for i, judge1 in enumerate(self.judge_columns):
            for judge2 in self.judge_columns[i+1:]:
                # Usuń pary z brakami
                scores1 = np.array(judge_rankings[judge1])
                scores2 = np.array(judge_rankings[judge2])
                
                valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))
                
                if valid_mask.sum() >= 3:
                    valid_scores1 = scores1[valid_mask]
                    valid_scores2 = scores2[valid_mask]
                    
                    # Kendall's tau
                    tau, p_value = stats.kendalltau(valid_scores1, valid_scores2)
                    
                    # Pearson dla porównania
                    pearson_r, _ = stats.pearsonr(valid_scores1, valid_scores2)
                    
                    results.append({
                        'judge1': judge1,
                        'judge2': judge2,
                        'kendall_tau': tau,
                        'tau_p_value': p_value,
                        'pearson_r': pearson_r,
                        'n_common': valid_mask.sum(),
                        'agreement_level': 'strong' if tau > 0.7 else ('moderate' if tau > 0.5 else 'weak')
                    })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('kendall_tau', ascending=False)
        
        return results_df
    
    def cohens_kappa_classification(self, threshold_top_n: int = 10) -> pd.DataFrame:
        """
        Oblicza Cohen's kappa dla klasyfikacji binarnej:
        czy uczestnik jest w TOP N czy nie
        
        Pokazuje zgodność sędziów w identyfikacji najlepszych uczestników
        """
        # Dla każdego etapu
        kappa_results = []
        
        for stage_name, df in self.stages_data.items():
            # Dla każdej pary sędziów
            for i, judge1 in enumerate(self.judge_columns):
                for judge2 in self.judge_columns[i+1:]:
                    # Pobierz oceny
                    scores1 = pd.to_numeric(df[judge1], errors='coerce')
                    scores2 = pd.to_numeric(df[judge2], errors='coerce')
                    
                    # Usuń brakujące
                    valid_mask = ~(scores1.isna() | scores2.isna())
                    
                    if valid_mask.sum() >= threshold_top_n * 2:  # Minimum 2x top_n uczestników
                        valid_scores1 = scores1[valid_mask]
                        valid_scores2 = scores2[valid_mask]
                        
                        # Klasyfikacja: top N vs reszta
                        class1 = (valid_scores1 >= valid_scores1.quantile(1 - threshold_top_n/len(valid_scores1))).astype(int)
                        class2 = (valid_scores2 >= valid_scores2.quantile(1 - threshold_top_n/len(valid_scores2))).astype(int)
                        
                        # Cohen's kappa
                        # Macierz pomyłek
                        agree = (class1 == class2).sum()
                        total = len(class1)
                        po = agree / total  # Observed agreement
                        
                        # Expected agreement
                        p1_yes = class1.sum() / total
                        p1_no = 1 - p1_yes
                        p2_yes = class2.sum() / total
                        p2_no = 1 - p2_yes
                        pe = (p1_yes * p2_yes) + (p1_no * p2_no)
                        
                        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
                        
                        kappa_results.append({
                            'stage': stage_name,
                            'judge1': judge1,
                            'judge2': judge2,
                            'cohens_kappa': kappa,
                            'observed_agreement': po,
                            'expected_agreement': pe,
                            'n_participants': total,
                            'interpretation': ('almost perfect' if kappa > 0.8 else 
                                             ('substantial' if kappa > 0.6 else
                                              ('moderate' if kappa > 0.4 else 'weak')))
                        })
        
        results_df = pd.DataFrame(kappa_results)
        if not results_df.empty:
            results_df = results_df.sort_values('cohens_kappa', ascending=False)
        
        return results_df


def run_statistical_analysis(processor, output_dir: str = 'statistical_results'):
    """Uruchamia wszystkie analizy statystyczne"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ChopinStatisticalAnalyzer(processor)
    
    # 1. Bootstrap confidence intervals
    print("Obliczam bootstrap confidence intervals...")
    for stage in ['stage1', 'stage2', 'stage3', 'final']:
        if stage in processor.corrected_data:
            bootstrap_ci = analyzer.bootstrap_ranking_confidence(stage=stage, n_bootstrap=1000)
            if not bootstrap_ci.empty:
                bootstrap_ci.to_csv(f'{output_dir}/bootstrap_ci_{stage}.csv', index=False)
    
    # 2. Monte Carlo simulations
    print("Przeprowadzam symulacje Monte Carlo...")
    for stage in ['stage1', 'stage2', 'stage3', 'final']:
        if stage in processor.corrected_data:
            mc_results = analyzer.monte_carlo_ranking_stability(stage=stage, n_simulations=1000)
            if not mc_results.empty:
                mc_results.to_csv(f'{output_dir}/monte_carlo_{stage}.csv', index=False)
    
    # 3. Statistical significance
    print("Testuję istotność różnic między miejscami...")
    for stage in ['stage1', 'stage2', 'stage3', 'final']:
        if stage in processor.corrected_data:
            significance = analyzer.statistical_significance_between_ranks(stage=stage)
            if not significance.empty:
                significance.to_csv(f'{output_dir}/significance_{stage}.csv', index=False)
    
    # 4. Statistical ties
    print("Identyfikuję grupy statystycznie nierozróżnialne...")
    for stage in ['stage1', 'stage2', 'stage3', 'final']:
        if stage in processor.corrected_data:
            ties = analyzer.identify_statistical_ties(stage=stage)
            if not ties.empty:
                ties.to_csv(f'{output_dir}/ties_{stage}.csv', index=False)
    
    # 5. Kendall's tau
    print("Obliczam Kendall's tau między sędziami...")
    kendall_results = analyzer.pairwise_agreement_kendall()
    if not kendall_results.empty:
        kendall_results.to_csv(f'{output_dir}/kendall_tau_pairwise.csv', index=False)
    
    # 6. Cohen's kappa
    print("Obliczam Cohen's kappa dla klasyfikacji...")
    for top_n in [5, 10, 15]:
        kappa_results = analyzer.cohens_kappa_classification(threshold_top_n=top_n)
        if not kappa_results.empty:
            kappa_results.to_csv(f'{output_dir}/cohens_kappa_top{top_n}.csv', index=False)
    
    print(f"\nAnalizy statystyczne zapisane w: {output_dir}")
    
    return analyzer


if __name__ == "__main__":
    print("Uruchom najpierw chopin_data_processor.py, a następnie ten skrypt z obiektem processor")
