"""
Moduł do przetwarzania danych konkursu Chopinowskiego 2025
Implementuje algorytm korygowania ocen i obliczania wyników skumulowanych
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def custom_threshold_rank(series_to_rank: pd.Series, threshold: float = 0.03) -> pd.Series:
    """
    Oblicza niestandardowy ranking dla serii wyników:
    1. Wyniki różniące się o mniej niż 'threshold' otrzymują to samo miejsce (ex aequo).
    2. Numeracja rankingu jest kontynuowana (np. po dwóch 4. miejscach następuje 5.).

    Args:
        series_to_rank: Seria Pandas zawierająca wyniki (liczby) do rankowania.
        threshold: Maksymalna różnica, dla której wyniki są uznawane za ex aequo. Domyślnie 0.03.

    Returns:
        Seria Pandas z obliczonymi rankami (miejscami).
    """

    # Krok 1: Utwórz tymczasowy DataFrame i posortuj go malejąco
    temp_df = series_to_rank.to_frame(name='Wynik')
    temp_df_sorted = temp_df.sort_values(by='Wynik', ascending=False)

    # Krok 2: Przygotuj listę na końcowe rangi
    ranks = []

    if temp_df_sorted.empty:
        return pd.Series([], dtype='int')

    # Krok 3: Iteruj i obliczaj rangę

    # Pierwszy element zawsze ma rangę 1
    ranks.append(1)

    # Konwersja do listy w celu łatwego dostępu do wartości
    scores = temp_df_sorted['Wynik'].tolist()

    rank_i = 1
    for i in range(1, len(scores)):
        if scores[i] < 1:
            ranks.append(999)
            continue

        # Różnica wyniku poprzedniego i bieżącego
        diff = scores[i - 1] - scores[i]

        if diff < threshold:
            # Różnica mniejsza niż próg -> to samo miejsce co poprzednik
            ranks.append(ranks[-1])
        else:
            rank_i += 1
            # Różnica >= próg -> nowe miejsce.
            # Nowe miejsce to pozycja w posortowanym zbiorze + 1 (indeks 'i' + 1)
            ranks.append(rank_i)

    # Krok 4: Przypisz rangi z powrotem do posortowanego DF
    temp_df_sorted['Miejsce'] = ranks

    # Krok 5: Przywróć oryginalny indeks i zwróć kolumnę ranków
    # Zapewnia, że ranga jest przypisana do pierwotnego wiersza
    return temp_df_sorted.sort_index()['Miejsce']

class ChopinCompetitionProcessor:
    """Klasa do przetwarzania wyników konkursu Chopinowskiego"""
    
    def __init__(self):
        self.stages_data = {}
        self.corrected_data = {}
        self.cumulative_scores = {}
        self.judge_columns = []
        
        # Wagi dla poszczególnych etapów
        self.weights = {
            'stage1_cumulative': {'stage1': 1.0},
            'stage2_cumulative': {'stage1': 0.30, 'stage2': 0.70},
            'stage3_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.70},
            'final_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.35, 'final': 0.35}
        }
        
        # Limity odchyleń dla korekcji
        self.deviation_limits = {
            'stage1': 3,
            'stage2': 2,
            'stage3': 2,
            'final': 2
        }
    
    def load_data(self, stage1_path: str, stage2_path: str, 
                  stage3_path: str, final_path: str):
        """Wczytuje dane ze wszystkich etapów"""
        self.stages_data['stage1'] = pd.read_csv(stage1_path)
        self.stages_data['stage2'] = pd.read_csv(stage2_path)
        self.stages_data['stage3'] = pd.read_csv(stage3_path)
        self.stages_data['final'] = pd.read_csv(final_path)
        
        # Identyfikuj kolumny sędziów (wszystkie oprócz Nr, imię, nazwisko)
        for stage, df in self.stages_data.items():
            self.judge_columns = [col for col in df.columns if col not in ['Nr', 'imię', 'nazwisko']]
            break
    
    def process_scores(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Przetwarza oceny dla danego etapu:
        1. Oblicza średnią bez uwzględnienia 's' (student)
        2. Koryguje oceny odbiegające od średniej
        3. Oblicza średnią po korekcie
        """
        result_df = df.copy()
        
        # Kolumny z ocenami
        score_cols = self.judge_columns
        
        # Konwertuj oceny na numeric, 's' lub puste wartości na NaN
        for col in score_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        # Limit odchylenia dla tego etapu
        deviation_limit = self.deviation_limits[stage]
        
        # Dla każdego uczestnika
        averages_before = []
        averages_after = []
        corrections_made = []
        
        for idx in result_df.index:
            scores = result_df.loc[idx, score_cols].dropna()
            
            if len(scores) == 0:
                averages_before.append(np.nan)
                averages_after.append(np.nan)
                corrections_made.append(0)
                continue
            
            # Pierwsza średnia
            avg = scores.mean().round(2)
            averages_before.append(avg)
            
            # Korekta ocen
            corrections = 0
            corrected_scores = scores.copy()
            
            for judge in scores.index:
                score = scores[judge]
                if abs(score - avg) > deviation_limit:
                    corrections += 1
                    if score > avg:
                        corrected_scores[judge] = avg + deviation_limit
                    else:
                        corrected_scores[judge] = avg - deviation_limit
                    
                    # Zapisz skorygowaną wartość do DataFrame
                    result_df.loc[idx, judge] = corrected_scores[judge]
            
            # Średnia po korekcie
            final_avg = corrected_scores.mean().round(2)
            averages_after.append(final_avg)
            corrections_made.append(corrections)
        
        # Dodaj kolumny z wynikami
        result_df['avg_before_correction'] = averages_before
        result_df['avg_after_correction'] = averages_after
        result_df['corrections_made'] = corrections_made
        result_df['stage_score'] = result_df['avg_after_correction']
        
        return result_df
    
    def calculate_all_stages(self):
        """Oblicza wyniki dla wszystkich etapów"""
        for stage_name, df in self.stages_data.items():
            # print(f"Przetwarzanie {stage_name}...")
            self.corrected_data[stage_name] = self.process_scores(df, stage_name)
    
    def calculate_cumulative_scores(self, exclude_from_finals: List[int] = []):
        """Oblicza wyniki skumulowane według podanych wag"""

        # Etap I skumulowany (100% etap I)
        if 'stage1' in self.corrected_data:
            stage1_cum = self._merge_and_calculate_weighted(
                ['stage1'],
                self.weights['stage1_cumulative']
            )
            self.cumulative_scores['stage1_cumulative'] = stage1_cum

        # Etap II skumulowany (30% etap I + 70% etap II)
        if 'stage1' in self.corrected_data and 'stage2' in self.corrected_data:
            stage2_cum = self._merge_and_calculate_weighted(
                ['stage1', 'stage2'],
                self.weights['stage2_cumulative']
            )
            self.cumulative_scores['stage2_cumulative'] = stage2_cum
        
        # Etap III skumulowany (10% I + 20% II + 70% III)
        if all(stage in self.corrected_data for stage in ['stage1', 'stage2', 'stage3']):
            stage3_cum = self._merge_and_calculate_weighted(
                ['stage1', 'stage2', 'stage3'],
                self.weights['stage3_cumulative']
            )
            self.cumulative_scores['stage3_cumulative'] = stage3_cum
        
        # Finał skumulowany (10% I + 20% II + 35% III + 35% finał)
        if all(stage in self.corrected_data for stage in ['stage1', 'stage2', 'stage3', 'final']):
            final_cum = self._merge_and_calculate_weighted(
                ['stage1', 'stage2', 'stage3', 'final'],
                self.weights['final_cumulative'], exclude_from_finals
            )
            self.cumulative_scores['final_cumulative'] = final_cum
    
    def _merge_and_calculate_weighted(self, stages: List[str], weights: Dict[str, float], exclude_from_finals: List[int] = []) -> pd.DataFrame:
        """Łączy dane z różnych etapów i oblicza ważoną sumę"""
        
        # Rozpocznij od pierwszego etapu
        base_df = self.corrected_data[stages[0]][['Nr', 'imię', 'nazwisko', 'stage_score']].copy()
        base_df = base_df.rename(columns={'stage_score': f'{stages[0]}_score'})
        
        # Dodaj pozostałe etapy
        for stage in stages[1:]:
            stage_df = self.corrected_data[stage][['Nr', 'stage_score']].copy()
            stage_df = stage_df.rename(columns={'stage_score': f'{stage}_score'})
            base_df = base_df.merge(stage_df, on='Nr', how='inner')
        
        # Oblicz wynik ważony
        weighted_sum = 0
        for stage in stages:
            weight = weights[stage]
            base_df[f'{stage}_weighted'] = base_df[f'{stage}_score'] * weight
            weighted_sum += base_df[f'{stage}_weighted']
        
        base_df['cumulative_score'] = round(weighted_sum, 2)
        if len(stages) < 4:
            base_df['rank'] = base_df['cumulative_score'].rank(ascending=False, method='min')
        else:
            mask = base_df['Nr'].isin(exclude_from_finals)
            base_df.loc[mask, 'cumulative_score'] = 0.0
            base_df['rank'] = custom_threshold_rank(base_df['cumulative_score'])
        
        return base_df.sort_values('rank')
    
    def get_participant_progression(self, participant_nr: int) -> pd.DataFrame:
        """Śledzi progresję uczestnika przez wszystkie etapy"""
        progression = []
        
        for stage_name, df in self.corrected_data.items():
            if participant_nr in df['Nr'].values:
                row = df[df['Nr'] == participant_nr].iloc[0]
                progression.append({
                    'stage': stage_name,
                    'score': row['stage_score'],
                    'corrections': row['corrections_made']
                })
        
        return pd.DataFrame(progression)
    
    def get_judge_statistics(self) -> pd.DataFrame:
        """Oblicza statystyki dla każdego sędziego"""
        judge_stats = []
        
        for judge in self.judge_columns:
            stats = {
                'judge': judge,
            }
            
            for stage_name, df in self.corrected_data.items():
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
    
    def save_results(self, output_dir: str = '.'):
        """Zapisuje wszystkie wyniki do plików CSV"""
        import os
        
        # Utwórz katalog jeśli nie istnieje
        os.makedirs(output_dir, exist_ok=True)
        
        # Zapisz skorygowane dane
        for stage_name, df in self.corrected_data.items():
            df.to_csv(f'{output_dir}/{stage_name}_corrected.csv', index=False)
        
        # Zapisz wyniki skumulowane
        for cum_name, df in self.cumulative_scores.items():
            df.to_csv(f'{output_dir}/{cum_name}.csv', index=False)
        
        # Zapisz statystyki sędziów
        judge_stats = self.get_judge_statistics()
        judge_stats.to_csv(f'{output_dir}/judge_statistics.csv', index=False)
        
        print(f"Wyniki zapisane w katalogu: {output_dir}")


# Funkcje pomocnicze do szybkiego uruchomienia
def process_competition_data(stage1_path, stage2_path, stage3_path, final_path, output_dir='results'):
    """Główna funkcja do przetwarzania danych konkursu"""
    
    processor = ChopinCompetitionProcessor()
    processor.load_data(stage1_path, stage2_path, stage3_path, final_path)
    processor.calculate_all_stages()
    processor.calculate_cumulative_scores()
    processor.save_results(output_dir)
    
    return processor


if __name__ == "__main__":
    # Przykład użycia
    processor = process_competition_data(
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    )
    
    print("\nStatystyki sędziów:")
    print(processor.get_judge_statistics().head())
    
    print("\nWyniki finalne:")
    if 'final_cumulative' in processor.cumulative_scores:
        print(processor.cumulative_scores['final_cumulative'][['Nr', 'imię', 'nazwisko', 'cumulative_score', 'rank']].head(10))
