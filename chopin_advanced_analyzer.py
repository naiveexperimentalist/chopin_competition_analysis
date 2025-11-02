"""
Moduł do zaawansowanych analiz wyników konkursu Chopinowskiego 2025
Zawiera analizy: wykorzystania skali, normalizacji, tendencji sędziów, sojuszy
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
    """Zaawansowane analizy wyników konkursu"""
    
    def __init__(self, processor):
        self.processor = processor
        self.judge_columns = processor.judge_columns
        self.stages_data = processor.stages_data
        self.corrected_data = processor.corrected_data
        
    def get_judge_statistics(self) -> pd.DataFrame:
        """Oblicza statystyki dla każdego sędziego"""
        judge_stats = []
        
        for judge in self.judge_columns:
            stats = {
                'judge': judge,
            }
            
            for stage_name, df in self.stages_data.items():
                scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                if len(scores) > 0:
                    stats[f'{stage_name}_mean'] = scores.mean()
                    stats[f'{stage_name}_std'] = scores.std()
                    stats[f'{stage_name}_min'] = scores.min()
                    stats[f'{stage_name}_max'] = scores.max()
                    stats[f'{stage_name}_range'] = scores.max() - scores.min()
                    stats[f'{stage_name}_count'] = len(scores)
            
            judge_stats.append(stats)
        
        return pd.DataFrame(judge_stats)
    
    def analyze_scale_usage(self) -> pd.DataFrame:
        """
        Analizuje jak sędziowie wykorzystują skalę 1-25
        - rozpiętość ocen
        - histogram ocen
        - entropia rozkładu
        """
        scale_usage = []
        
        for judge in self.judge_columns:
            judge_info = {'judge': judge}
            all_scores = []
            
            for stage_name, df in self.stages_data.items():
                scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                if len(scores) > 0:
                    all_scores.extend(scores.tolist())
                    
                    # Statystyki dla etapu
                    judge_info[f'{stage_name}_range'] = scores.max() - scores.min()
                    judge_info[f'{stage_name}_unique_scores'] = scores.nunique()
                    
                    # Entropia - miara różnorodności ocen
                    value_counts = scores.value_counts()
                    probabilities = value_counts / len(scores)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    judge_info[f'{stage_name}_entropy'] = entropy
            
            if all_scores:
                # Statystyki ogólne
                judge_info['overall_range'] = max(all_scores) - min(all_scores)
                judge_info['overall_mean'] = np.mean(all_scores)
                judge_info['overall_std'] = np.std(all_scores)
                judge_info['overall_unique_scores'] = len(set(all_scores))
                judge_info['overall_skewness'] = stats.skew(all_scores)
                judge_info['overall_kurtosis'] = stats.kurtosis(all_scores)
                
                # Procent wykorzystania skali (ile unikalnych wartości z 25)
                judge_info['scale_coverage'] = (len(set(all_scores)) / 25) * 100
            
            scale_usage.append(judge_info)
        
        return pd.DataFrame(scale_usage)
    
    def normalize_scores(self) -> Dict[str, pd.DataFrame]:
        """
        Normalizuje oceny sędziów używając różnych metod:
        1. Z-score normalization (standaryzacja)
        2. Min-max scaling
        3. Rank-based normalization
        """
        normalized_data = {}
        
        for stage_name, df in self.stages_data.items():
            df_norm = df.copy()
            
            # Z-score normalization dla każdego sędziego
            df_zscore = df.copy()
            for judge in self.judge_columns:
                scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~scores.isna()
                if valid_mask.sum() > 0:
                    z_scores = stats.zscore(scores[valid_mask])
                    # Przekształć z-score na skalę 1-25
                    z_scaled = 13 + (z_scores * 4)  # Środek na 13, std ~4
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
                    # Przekształć rangi na skalę 1-25
                    rank_scaled = 1 + ((ranks - 1) / (len(ranks) - 1)) * 24 if len(ranks) > 1 else [13] * len(ranks)
                    df_rank.loc[valid_mask, judge] = rank_scaled
            
            normalized_data[f'{stage_name}_original'] = df
            normalized_data[f'{stage_name}_zscore'] = df_zscore
            normalized_data[f'{stage_name}_minmax'] = df_minmax
            normalized_data[f'{stage_name}_rank'] = df_rank
        
        return normalized_data
    
    def analyze_judge_tendencies(self) -> pd.DataFrame:
        """
        Analizuje tendencje poszczególnych sędziów:
        - surowość (średnia ocen vs. średnia ogólna)
        - faworyzowanie konkretnych uczestników
        - konsystencja oceniania
        """
        tendencies = []
        
        for judge in self.judge_columns:
            tendency = {'judge': judge}
            
            # Zbierz wszystkie oceny sędziego i średnie
            judge_scores_all = []
            avg_scores_all = []
            
            for stage_name, df in self.corrected_data.items():
                judge_scores = pd.to_numeric(df[judge], errors='coerce').dropna()
                stage_avgs = df.loc[judge_scores.index, 'avg_before_correction']
                
                if len(judge_scores) > 0:
                    judge_scores_all.extend(judge_scores.tolist())
                    avg_scores_all.extend(stage_avgs.tolist())
                    
                    # Średnia różnica od średniej (ujemna = surowy, dodatnia = łagodny)
                    diff = judge_scores - stage_avgs
                    tendency[f'{stage_name}_harshness'] = diff.mean()
                    tendency[f'{stage_name}_consistency'] = diff.std()
            
            if judge_scores_all and avg_scores_all:
                # Ogólne tendencje
                tendency['overall_harshness'] = np.mean(np.array(judge_scores_all) - np.array(avg_scores_all))
                tendency['overall_consistency'] = np.std(np.array(judge_scores_all) - np.array(avg_scores_all))
                
                # Korelacja z średnią (wysoka = zgodny z konsensusem)
                tendency['consensus_correlation'] = np.corrcoef(judge_scores_all, avg_scores_all)[0, 1]
            
            tendencies.append(tendency)
        
        return pd.DataFrame(tendencies)
    
    def find_judge_favorites(self, min_stages: int = 3) -> pd.DataFrame:
        """
        Znajduje uczestników konsekwentnie wysoko/nisko ocenianych przez sędziów
        (przez co najmniej min_stages etapów)
        """
        favorites = []
        
        for judge in self.judge_columns:
            # Śledź oceny każdego uczestnika przez etapy
            participant_tracking = {}
            
            for stage_name, df in self.corrected_data.items():
                for idx, row in df.iterrows():
                    participant_id = row['Nr']
                    judge_score = pd.to_numeric(row[judge], errors='coerce')
                    avg_score = row['avg_before_correction']
                    
                    if pd.notna(judge_score) and pd.notna(avg_score):
                        if participant_id not in participant_tracking:
                            participant_tracking[participant_id] = {
                                'name': f"{row['imię']} {row['nazwisko']}",
                                'scores': [],
                                'differences': []
                            }
                        
                        participant_tracking[participant_id]['scores'].append(judge_score)
                        participant_tracking[participant_id]['differences'].append(judge_score - avg_score)
            
            # Analizuj konsystentne tendencje
            for participant_id, data in participant_tracking.items():
                if len(data['differences']) >= min_stages:
                    avg_diff = np.mean(data['differences'])
                    consistency = np.std(data['differences'])
                    
                    # Jeśli konsekwentnie wysoko (>1) lub nisko (<-1) z małą wariancją
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
        Symuluje usunięcie każdego sędziego i sprawdza wpływ na wyniki finalne
        """
        results = []

        # Baseline - wyniki z wszystkimi sędziami
        baseline_final = self.processor.cumulative_scores.get('final_cumulative')
        if baseline_final is None:
            print("Brak wyników finalnych do analizy")
            return pd.DataFrame()

        baseline_ranks = dict(zip(baseline_final['Nr'], baseline_final['rank']))

        # Dla każdego sędziego
        for judge_to_remove in self.judge_columns:
            try:
                # Stwórz tymczasowy procesor bez tego sędziego
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Kopiuj dane bez kolumny sędziego
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Ustaw judge_columns bez usuniętego sędziego
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Przelicz wyniki
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                # Porównaj rankingi
                if 'final_cumulative' in temp_processor.cumulative_scores:
                    new_final = temp_processor.cumulative_scores['final_cumulative']
                    new_ranks = dict(zip(new_final['Nr'], new_final['rank']))

                    # Oblicz zmiany w rankingu
                    rank_changes = []
                    for participant_nr in baseline_ranks:
                        if participant_nr in new_ranks:
                            change = baseline_ranks[participant_nr] - new_ranks[participant_nr]
                            rank_changes.append(abs(change))

                    if rank_changes:  # Sprawdź czy lista nie jest pusta
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
                print(f"  Błąd przy usuwaniu sędziego {judge_to_remove}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def analyze_qualification_after_judge_removal(self) -> pd.DataFrame:
        """
        Symuluje usunięcie każdego sędziego i sprawdza wpływ na kwalifikowanie się zawodników do kolejnych rund
        """
        results = []

        # Baseline - wyniki kolejnych run z wszystkimi sędziami
        baseline_stage1 = self.processor.cumulative_scores.get('stage1_cumulative')
        baseline_stage2 = self.processor.cumulative_scores.get('stage2_cumulative')
        baseline_stage3 = self.processor.cumulative_scores.get('stage3_cumulative')
        baseline_final = self.processor.cumulative_scores.get('final_cumulative')
        if baseline_final is None:
            print("Brak wyników finalnych do analizy")
            return pd.DataFrame()

        baseline_stage1_ranks = dict(zip(baseline_stage1['Nr'], baseline_stage1['rank']))
        baseline_stage2_ranks = dict(zip(baseline_stage2['Nr'], baseline_stage2['rank']))
        baseline_stage3_ranks = dict(zip(baseline_stage3['Nr'], baseline_stage3['rank']))
        baseline_final_ranks = dict(zip(baseline_final['Nr'], baseline_final['rank']))

        QUALIFICATION_THRESHOLDS = {
            'stage1': 40,  # top 40 przechodzi do stage2
            'stage2': 20,  # top 20 przechodzi do stage3
            'stage3': 10   # top 10 przechodzi do finału
        }

        # Dla każdego sędziego
        for judge_to_remove in self.judge_columns:
            try:
                # Stwórz tymczasowy procesor bez tego sędziego
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Kopiuj dane bez kolumny sędziego
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Ustaw judge_columns bez usuniętego sędziego
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Przelicz wyniki
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                new_stage1 = temp_processor.cumulative_scores['stage1_cumulative']
                new_stage2 = temp_processor.cumulative_scores['stage2_cumulative']
                new_stage3 = temp_processor.cumulative_scores['stage3_cumulative']
                new_final = temp_processor.cumulative_scores['final_cumulative']

                # Analiza kwalifikacji dla każdego etapu
                qualification_changes = {}

                # Stage 1 -> Stage 2
                stage1_baseline_qualified = set(
                    baseline_stage1[baseline_stage1['rank'] <= QUALIFICATION_THRESHOLDS['stage1']]['Nr'])
                stage1_new_qualified = set(new_stage1[new_stage1['rank'] <= QUALIFICATION_THRESHOLDS['stage1']]['Nr'])
                qualification_changes['stage1_to_stage2'] = {
                    'lost_qualification': list(stage1_baseline_qualified - stage1_new_qualified),
                    'gained_qualification': list(stage1_new_qualified - stage1_baseline_qualified)
                }

                # Stage 2 -> Stage 3
                stage2_baseline_qualified = set(
                    baseline_stage2[baseline_stage2['rank'] <= QUALIFICATION_THRESHOLDS['stage2']]['Nr'])
                stage2_new_qualified = set(new_stage2[new_stage2['rank'] <= QUALIFICATION_THRESHOLDS['stage2']]['Nr'])
                qualification_changes['stage2_to_stage3'] = {
                    'lost_qualification': list(stage2_baseline_qualified - stage2_new_qualified),
                    'gained_qualification': list(stage2_new_qualified - stage2_baseline_qualified)
                }

                # Stage 3 -> Final
                stage3_baseline_qualified = set(
                    baseline_stage3[baseline_stage3['rank'] <= QUALIFICATION_THRESHOLDS['stage3']]['Nr'])
                stage3_new_qualified = set(new_stage3[new_stage3['rank'] <= QUALIFICATION_THRESHOLDS['stage3']]['Nr'])
                qualification_changes['stage3_to_final'] = {
                    'lost_qualification': list(stage3_baseline_qualified - stage3_new_qualified),
                    'gained_qualification': list(stage3_new_qualified - stage3_baseline_qualified)
                }

                results.append({
                    'judge_removed': judge_to_remove,
                    **qualification_changes
                })
            except Exception as e:
                print(f"  Błąd przy usuwaniu sędziego {judge_to_remove}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def generate_results_after_judge_removal(self) -> pd.DataFrame:
        """
        Symuluje pełne zawody (od stage1 do finału) po usunięciu każdego sędziego.
        Uwzględnia że uczestnik może nie zakwalifikować się do stage3 lub finału.

        Args:
            num_qualifiers_stage3: liczba uczestników przechodzących do stage3 (domyślnie 20)
            num_qualifiers_final: liczba uczestników przechodzących do finału (domyślnie 10)

        Returns:
            DataFrame z kolumnami:
            - Nr: numer startowy uczestnika
            - imię, nazwisko: dane uczestnika
            - original_rank: oryginalny ranking końcowy
            - [judge_name]_rank: ranking po usunięciu danego sędziego (lub 'n/a' jeśli nie przeszedłby do finału)
            - [judge_name]_change: zmiana pozycji (dodatnia = poprawa, ujemna = pogorszenie, 'n/a' = dyskwalifikacja)
        """
        QUALIFICATION_THRESHOLDS = {
            'stage1': 40,  # top 40 przechodzi do stage2
            'stage2': 20,  # top 20 przechodzi do stage3
            'stage3': 10,   # top 10 przechodzi do finału
            'real_stage3': 11  # top 11 przeszło do finału
        }

        # # Pobierz oryginalne wyniki końcowe
        # baseline_final = self.processor.cumulative_scores.get('final_cumulative')
        #
        # # Weź finalistów z oryginalnego scenariusza
        # orig_finalists = baseline_final.nsmallest(QUALIFICATION_THRESHOLDS['stage3'], 'rank').copy()
        # orig_finalists = orig_finalists.sort_values('rank')

        # Pobierz oryginalne wyniki po finale
        orig_finalists = self.processor.cumulative_scores.get('final_cumulative')

        # Przygotuj DataFrame wynikowy
        result_df = orig_finalists[['Nr', 'imię', 'nazwisko', 'rank']].copy()
        result_df = result_df.rename(columns={'rank': 'original_rank'})
        result_df['original_rank'] = result_df['original_rank'].astype(int)

        print(f"\nFinaliści w oryginalnym scenariuszu (top {QUALIFICATION_THRESHOLDS['stage3']}):")
        for _, row in orig_finalists.iterrows():
            print(f"  {int(row['rank'])}: Nr {row['Nr']} - {row['imię']} {row['nazwisko']}")

        # Dla każdego sędziego symuluj pełne zawody
        for judge_to_remove in self.judge_columns:
            # print(f"\n{'=' * 60}")
            # print(f"Symulacja bez sędziego: {judge_to_remove}")
            # print(f"{'=' * 60}")

            try:
                # Stwórz tymczasowy procesor bez tego sędziego
                temp_processor = type(self.processor)()
                temp_processor.stages_data = {}

                # Kopiuj dane bez kolumny sędziego
                for stage_name, df in self.processor.stages_data.items():
                    temp_df = df.copy()
                    if judge_to_remove in temp_df.columns:
                        temp_df = temp_df.drop(columns=[judge_to_remove])
                    temp_processor.stages_data[stage_name] = temp_df

                # Ustaw judge_columns bez usuniętego sędziego
                temp_processor.judge_columns = [j for j in self.judge_columns if j != judge_to_remove]
                temp_processor.weights = self.processor.weights
                temp_processor.deviation_limits = self.processor.deviation_limits

                # Przelicz wyniki
                temp_processor.calculate_all_stages()
                temp_processor.calculate_cumulative_scores()

                exclude_from_finals = set()

                # Sprawdź kto przeszedłby do stage2
                new_stage1 = temp_processor.cumulative_scores.get('stage1_cumulative')
                stage2_qualifiers = set(
                    new_stage1.nsmallest(QUALIFICATION_THRESHOLDS['stage1'], 'rank')['Nr'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if not row['Nr'] in stage2_qualifiers:
                        exclude_from_finals.add(int(row['Nr']))

                # Sprawdź kto przeszedłby do stage3
                new_stage2 = temp_processor.cumulative_scores.get('stage2_cumulative')
                stage3_qualifiers = set(
                    new_stage2.nsmallest(QUALIFICATION_THRESHOLDS['stage2'], 'rank')['Nr'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if not row['Nr'] in stage3_qualifiers:
                        exclude_from_finals.add(int(row['Nr']))

                # Sprawdź kto przeszedłby do finału
                new_stage3 = temp_processor.cumulative_scores.get('stage3_cumulative')
                final_qualifiers = set(
                    new_stage3.nsmallest(QUALIFICATION_THRESHOLDS['stage3'], 'rank')['Nr'].values
                )
                for i, (_, row) in enumerate(orig_finalists.iterrows()):
                    if not row['Nr'] in final_qualifiers:
                        exclude_from_finals.add(int(row['Nr']))

                temp_processor.calculate_cumulative_scores(exclude_from_finals)

                # Pobierz wyniki końcowe
                new_final = temp_processor.cumulative_scores.get('final_cumulative')

                # Ustawienie nazw kolumn
                judge_col_name = f"{judge_to_remove}_rank"
                change_col_name = f"{judge_to_remove}_change"

                result_df = result_df.merge(new_final[['Nr', 'rank']].rename(columns={'rank': judge_col_name}), on='Nr', how='left')

                result_df[judge_col_name] = result_df[judge_col_name].replace(999, 'n/a')

                result_df[change_col_name] = result_df.apply(
                    lambda row: 'n/a' if row[judge_col_name] == 'n/a'
                    else int(row['original_rank']) - int(row[judge_col_name]),
                    axis=1
                )
            except Exception as e:
                print(f"  ✗ Błąd przy usuwaniu sędziego {judge_to_remove}: {str(e)}")
                # Dodaj puste kolumny żeby zachować strukturę
                judge_col_name = f"{judge_to_remove}_rank"
                change_col_name = f"{judge_to_remove}_change"
                result_df[judge_col_name] = 'error'
                result_df[change_col_name] = 'error'
                continue

        print(f"\n{'=' * 60}")
        print("SYMULACJA ZAKOŃCZONA")
        print(f"{'=' * 60}")

        return result_df

    def analyze_judge_alliances(self, threshold: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analizuje potencjalne sojusze między sędziami na podstawie korelacji ocen
        """
        # Zbierz wszystkie oceny każdego sędziego
        all_scores_by_judge = {judge: [] for judge in self.judge_columns}
        participant_stage_keys = []
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                key = f"{row['Nr']}_{stage_name}"
                participant_stage_keys.append(key)
                
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    all_scores_by_judge[judge].append(score)
        
        # Utwórz DataFrame z ocenami wszystkich sędziów
        scores_df = pd.DataFrame(all_scores_by_judge)
        
        # Oblicz korelacje między sędziami
        correlation_matrix = scores_df.corr(method='pearson')
        
        # Znajdź pary z wysoką korelacją
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
        Analizuje potencjalne krajowe sympatie/antypatie
        Wymaga dostarczenia mapowania sędzia -> narodowość
        """
        if nationality_mapping is None:
            # Przykładowe mapowanie - należy uzupełnić rzeczywistymi danymi
            print("Brak danych o narodowości sędziów. Podaj słownik nationality_mapping.")
            return pd.DataFrame()
        
        bias_results = []
        
        # Dla każdego sędziego i narodowości
        for judge in self.judge_columns:
            if judge not in nationality_mapping:
                continue
            
            judge_nationality = nationality_mapping[judge]
            
            # Analizuj oceny dla uczestników z różnych krajów
            # (zakładając, że mamy informacje o narodowości uczestników)
            # To wymaga dodatkowych danych...
            
            bias_results.append({
                'judge': judge,
                'nationality': judge_nationality,
                # Dodać analizę bias gdy dostępne dane o narodowości uczestników
            })
        
        return pd.DataFrame(bias_results)
    
    def calculate_judge_independence(self) -> pd.DataFrame:
        """
        Oblicza stopień niezależności każdego sędziego od konsensusu
        """
        independence_scores = []
        
        for judge in self.judge_columns:
            judge_data = {'judge': judge}
            
            # Zbierz wszystkie przypadki gdzie sędzia się różnił od średniej
            total_deviations = []
            total_corrections = 0
            
            for stage_name, df in self.corrected_data.items():
                judge_scores = pd.to_numeric(df[judge], errors='coerce')
                valid_mask = ~judge_scores.isna()
                
                if valid_mask.sum() > 0:
                    # Różnice od średniej przed korekcją
                    avg_scores = df.loc[valid_mask, 'avg_before_correction']
                    deviations = abs(judge_scores[valid_mask] - avg_scores)
                    total_deviations.extend(deviations.tolist())
                    
                    # Ile razy oceny były korygowane
                    stage_corrections = df.loc[valid_mask, 'corrections_made'].sum()
                    total_corrections += stage_corrections
            
            if total_deviations:
                judge_data['avg_deviation'] = np.mean(total_deviations)
                judge_data['max_deviation'] = max(total_deviations)
                judge_data['independence_score'] = np.percentile(total_deviations, 75)  # 75 percentyl odchyleń
                judge_data['times_corrected'] = total_corrections
            
            independence_scores.append(judge_data)
        
        return pd.DataFrame(independence_scores)


# Funkcje pomocnicze do analiz
def run_advanced_analysis(processor, output_dir: str = 'advanced_results'):
    """Uruchamia wszystkie zaawansowane analizy"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ChopinAdvancedAnalyzer(processor)
    
    # 1. Analiza wykorzystania skali
    print("Analizuję wykorzystanie skali 1-25...")
    scale_usage = analyzer.analyze_scale_usage()
    scale_usage.to_csv(f'{output_dir}/scale_usage_analysis.csv', index=False)
    
    # 2. Normalizacja ocen
    print("Normalizuję oceny...")
    normalized = analyzer.normalize_scores()
    # Zapisz wybrane normalizacje
    for key, df in normalized.items():
        if 'zscore' in key:
            df.to_csv(f'{output_dir}/{key}.csv', index=False)
    
    # 3. Tendencje sędziów
    print("Analizuję tendencje sędziów...")
    tendencies = analyzer.analyze_judge_tendencies()
    tendencies.to_csv(f'{output_dir}/judge_tendencies.csv', index=False)
    
    # 4. Faworyci sędziów
    print("Szukam faworytów...")
    favorites = analyzer.find_judge_favorites()
    if not favorites.empty:
        favorites.to_csv(f'{output_dir}/judge_favorites.csv', index=False)

    # 5. Symulacja usunięcia sędziego
    print("Symulacja usunięcia sędziów...")
    removal_impact = analyzer.simulate_judge_removal()
    if not removal_impact.empty:
        removal_impact.to_csv(f'{output_dir}/judge_removal_impact.csv', index=False)

    # 5.3. Kwalifikacje do kolejnej rundy bez jednego sędziego
    print("Kwalifikacje do kolejnej rundy bez jednego sędziego...")
    qualification_with_removed_judge = analyzer.analyze_qualification_after_judge_removal()
    if not qualification_with_removed_judge.empty:
        qualification_with_removed_judge.to_csv(f'{output_dir}/qualification_with_removed_judge.csv', index=False)

    # 5.5. Wyniki konkursu bez jednego sędziego
    print("Wyniki konkursu bez jednego sędziego...")
    results_with_removed_judge = analyzer.generate_results_after_judge_removal()
    if not results_with_removed_judge.empty:
        results_with_removed_judge.to_csv(f'{output_dir}/results_after_judge_removal.csv', index=False)
    
    # 6. Sojusze sędziowskie
    print("Analizuję sojusze...")
    correlation_matrix, alliances = analyzer.analyze_judge_alliances()
    correlation_matrix.to_csv(f'{output_dir}/judge_correlations.csv')
    if not alliances.empty:
        alliances.to_csv(f'{output_dir}/judge_alliances.csv', index=False)
    else:
        print("  Nie znaleziono silnych sojuszy (korelacja > 0.6)")
    
    # 7. Niezależność sędziów
    print("Obliczam niezależność...")
    independence = analyzer.calculate_judge_independence()
    independence.to_csv(f'{output_dir}/judge_independence.csv', index=False)
    
    print(f"\nAnalizy zapisane w katalogu: {output_dir}")
    
    return analyzer


if __name__ == "__main__":
    # Przykład użycia
    print("Uruchom najpierw chopin_data_processor.py, a następnie ten skrypt z obiektem processor")
