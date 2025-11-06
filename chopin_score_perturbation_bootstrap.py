"""
Analiza stabilności wyniku końcowego konkursu Chopinowskiego
Bootstrap przez perturbację ocen - dodawanie losowych wartości [-1, -0.5, 0, 0.5, 1]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChopinScoreCalculator:
    """Kalkulator wyników z korekcjami outlierów"""
    
    @staticmethod
    def calculate_corrected_average(scores: np.ndarray, threshold: float = 3.0) -> float:
        """
        Oblicza średnią z korekcją outlierów
        
        Args:
            scores: tablica ocen (bez NaN i 's')
            threshold: próg odchylenia od średniej (3 dla etapu 1, 2 dla pozostałych)
        
        Returns:
            skorygowana średnia zaokrąglona do 2 miejsc
        """
        if len(scores) == 0:
            return 0.0
        
        # Krok 1: oblicz średnią
        avg = np.mean(scores)
        
        # Krok 2: skoryguj outliery
        corrected = scores.copy()
        for i, score in enumerate(scores):
            if score > avg + threshold:
                corrected[i] = avg + threshold
            elif score < avg - threshold:
                corrected[i] = avg - threshold
        
        # Krok 3: oblicz średnią po korekcjach
        final_avg = np.mean(corrected)
        
        return round(final_avg, 2)


class ScorePerturbationAnalyzer:
    """Analiza stabilności przez perturbację ocen"""
    
    def __init__(self, stage_files: Dict[str, str], 
                 perturbation_values: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]):
        """
        Args:
            stage_files: słownik {nazwa_etapu: ścieżka_do_pliku}
            perturbation_values: możliwe wartości perturbacji
        """
        self.stage_files = stage_files
        self.perturbation_values = perturbation_values
        self.stages_data = {}
        self.judge_columns = []
        self.thresholds = {
            'stage1': 3.0,
            'stage2': 2.0,
            'stage3': 2.0,
            'final': 2.0
        }
        self.weights = {
            'stage1': 0.10,
            'stage2': 0.20,
            'stage3': 0.35,
            'final': 0.35
        }
        
        self._load_data()
    
    def _load_data(self):
        """Wczytuje dane ze wszystkich etapów"""
        print("Wczytuję dane z etapów...")
        
        for stage, filepath in self.stage_files.items():
            df = pd.read_csv(filepath)
            
            # Identyfikuj kolumny sędziów
            if not self.judge_columns:
                self.judge_columns = [col for col in df.columns 
                                     if col not in ['Nr', 'imię', 'nazwisko']]
            
            # Konwertuj oceny (zamień 's' na NaN)
            for judge in self.judge_columns:
                df[judge] = pd.to_numeric(df[judge], errors='coerce')
            
            self.stages_data[stage] = df
            print(f"  {stage}: {len(df)} uczestników, {len(self.judge_columns)} sędziów")
    
    def perturb_scores(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje losowe perturbacje do ocen
        
        Args:
            original_df: oryginalny DataFrame z ocenami
        
        Returns:
            DataFrame z perturowanymi ocenami (NaN pozostają jako NaN)
        """
        perturbed = original_df.copy()
        
        for judge in self.judge_columns:
            # Generuj perturbacje tylko dla niepustych ocen
            mask = perturbed[judge].notna()
            n_scores = mask.sum()
            
            if n_scores > 0:
                perturbations = np.random.choice(
                    self.perturbation_values, 
                    size=n_scores
                )
                
                # Dodaj perturbacje
                perturbed.loc[mask, judge] = perturbed.loc[mask, judge] + perturbations
                
                # Ogranicz do zakresu [1, 25]
                perturbed.loc[mask, judge] = perturbed.loc[mask, judge].clip(1, 25)
        
        return perturbed
    
    def calculate_stage_score_for_participant(self, stage: str, participant_nr: int,
                                              perturbed_data: Dict[str, pd.DataFrame] = None) -> float:
        """
        Oblicza wynik uczestnika w danym etapie
        
        Args:
            stage: nazwa etapu
            participant_nr: numer uczestnika
            perturbed_data: opcjonalnie perturowane dane (None = oryginalne)
        
        Returns:
            skorygowana średnia ocena
        """
        if perturbed_data:
            df = perturbed_data[stage]
        else:
            df = self.stages_data[stage]
        
        participant = df[df['Nr'] == participant_nr]
        
        if participant.empty:
            return 0.0
        
        # Pobierz oceny (pomijając NaN)
        scores = []
        for judge in self.judge_columns:
            score = participant[judge].iloc[0]
            if pd.notna(score):
                scores.append(score)
        
        if not scores:
            return 0.0
        
        # Oblicz skorygowaną średnią
        threshold = self.thresholds[stage]
        calculator = ChopinScoreCalculator()
        return calculator.calculate_corrected_average(np.array(scores), threshold)
    
    def calculate_final_weighted_score(self, participant_nr: int,
                                       perturbed_data: Dict[str, pd.DataFrame] = None) -> float:
        """
        Oblicza końcowy ważony wynik uczestnika
        
        Args:
            participant_nr: numer uczestnika
            perturbed_data: opcjonalnie perturowane dane
        
        Returns:
            końcowy ważony wynik
        """
        # Sprawdź, czy uczestnik wystąpił we wszystkich potrzebnych etapach
        data_to_use = perturbed_data if perturbed_data else self.stages_data
        
        available_stages = []
        for stage in self.stage_files.keys():
            if participant_nr in data_to_use[stage]['Nr'].values:
                available_stages.append(stage)
        
        if not available_stages:
            return 0.0
        
        # Oblicz wynik z każdego etapu
        stage_scores = {}
        for stage in available_stages:
            stage_scores[stage] = self.calculate_stage_score_for_participant(
                stage, participant_nr, perturbed_data
            )
        
        # Oblicz ważoną średnią
        weighted_sum = 0.0
        total_weight = 0.0
        
        for stage in available_stages:
            weight = self.weights[stage]
            weighted_sum += stage_scores[stage] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight * sum(self.weights.values())
    
    def bootstrap_with_perturbation(self, n_iterations: int = 10000) -> Dict[int, List[float]]:
        """
        Bootstrap z perturbacją ocen
        
        Args:
            n_iterations: liczba iteracji bootstrapu
        
        Returns:
            słownik {nr_uczestnika: lista_wyników_z_iteracji}
        """
        print(f"\nRozpoczynanie bootstrapu z perturbacją: {n_iterations} iteracji...")
        print(f"Wartości perturbacji: {self.perturbation_values}")
        
        # Znajdź wszystkich uczestników finału
        final_participants = self.stages_data['final']['Nr'].values
        
        bootstrap_results = {nr: [] for nr in final_participants}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteracja {iteration + 1}/{n_iterations}")
            
            # Dla każdego etapu wygeneruj perturbowane dane
            perturbed_data = {}
            for stage in self.stage_files.keys():
                perturbed_data[stage] = self.perturb_scores(self.stages_data[stage])
            
            # Oblicz wynik końcowy dla każdego uczestnika
            for participant_nr in final_participants:
                final_score = self.calculate_final_weighted_score(
                    participant_nr, perturbed_data
                )
                bootstrap_results[participant_nr].append(final_score)
        
        print("Bootstrap zakończony!")
        return bootstrap_results
    
    def get_actual_final_scores(self) -> pd.DataFrame:
        """
        Oblicza rzeczywiste wyniki końcowe (bez perturbacji)
        
        Returns:
            DataFrame z kolumnami: Nr, imię, nazwisko, final_score
        """
        final_df = self.stages_data['final'][['Nr', 'imię', 'nazwisko']].copy()
        
        final_scores = []
        for _, row in final_df.iterrows():
            score = self.calculate_final_weighted_score(row['Nr'])
            final_scores.append(score)
        
        final_df['final_score'] = final_scores
        final_df = final_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        final_df['rank'] = range(1, len(final_df) + 1)
        
        return final_df


class PerturbationVisualizer:
    """Wizualizacje stabilności z perturbacją"""
    
    def __init__(self, analyzer: ScorePerturbationAnalyzer):
        self.analyzer = analyzer
    
    def visualize_score_distributions(self, bootstrap_results: Dict[int, List[float]],
                                     save_path: str = None):
        """Violin plots rozkładów wyników z perturbacją"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Przygotuj dane
        plot_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            name = f"{row['rank']}. {row['nazwisko']}"
            actual_score = row['final_score']
            
            for score in bootstrap_results[nr]:
                plot_data.append({
                    'participant': name,
                    'rank': row['rank'],
                    'score': score,
                    'actual_score': actual_score
                })
        
        df = pd.DataFrame(plot_data)
        
        # Rysuj
        fig, ax = plt.subplots(figsize=(16, max(10, len(actual_scores) * 0.4)))
        
        order = [f"{r}. {n}" for r, n in 
                zip(actual_scores['rank'], actual_scores['nazwisko'])]
        
        sns.violinplot(data=df, y='participant', x='score', order=order, 
                      inner='quartile', ax=ax, palette='Set2')
        
        # Czerwone punkty dla rzeczywistych wyników
        for i, (_, row) in enumerate(actual_scores.iterrows()):
            ax.plot(row['final_score'], i, 'ro', markersize=8, 
                   label='Rzeczywisty wynik' if i == 0 else '')
        
        ax.set_xlabel('Wynik końcowy', fontsize=12)
        ax.set_ylabel('Uczestnik (ranga. nazwisko)', fontsize=12)
        ax.set_title('Stabilność wyniku końcowego - Perturbacja ocen\n' +
                    f'Rozkład możliwych wyników przy zmianach ocen o {self.analyzer.perturbation_values}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_confidence_intervals(self, bootstrap_results: Dict[int, List[float]],
                                      confidence: float = 0.95, save_path: str = None):
        """Przedziały ufności dla wyników z perturbacją"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Oblicz przedziały ufności
        ci_data = []
        alpha = (1 - confidence) / 2
        
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = bootstrap_results[nr]
            
            ci_low = np.percentile(scores, alpha * 100)
            ci_high = np.percentile(scores, (1 - alpha) * 100)
            ci_width = ci_high - ci_low
            std = np.std(scores)
            
            ci_data.append({
                'rank': row['rank'],
                'participant': f"{row['imię']} {row['nazwisko']}",
                'actual_score': row['final_score'],
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_width': ci_width,
                'std': std
            })
        
        ci_df = pd.DataFrame(ci_data)
        
        # Rysuj
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        
        # 1. Error bars
        ax1 = axes[0]
        y_pos = range(len(ci_df))
        
        ax1.errorbar(ci_df['actual_score'], y_pos, 
                    xerr=[ci_df['actual_score'] - ci_df['ci_low'],
                          ci_df['ci_high'] - ci_df['actual_score']],
                    fmt='o', markersize=6, capsize=5, capthick=2,
                    alpha=0.7, label=f'{confidence*100:.0f}% przedział ufności')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}" 
                            for _, row in ci_df.iterrows()], fontsize=8)
        ax1.set_xlabel('Wynik końcowy', fontsize=12)
        ax1.set_ylabel('Uczestnik', fontsize=12)
        ax1.set_title(f'Przedziały ufności ({confidence*100:.0f}%) - Perturbacja ocen\n' +
                     f'Zmiana ocen o {self.analyzer.perturbation_values}',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # 2. Szerokość przedziału
        ax2 = axes[1]
        top_unstable = ci_df.nlargest(30, 'ci_width')
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_unstable)))
        
        ax2.barh(range(len(top_unstable)), top_unstable['ci_width'], 
                color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_unstable)))
        ax2.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}" 
                            for _, row in top_unstable.iterrows()], fontsize=8)
        ax2.set_xlabel('Szerokość przedziału ufności (punkty)', fontsize=12)
        ax2.set_ylabel('Uczestnik', fontsize=12)
        ax2.set_title('Wrażliwość na małe zmiany w ocenach\n' +
                     '(im szerszy przedział, tym większa wrażliwość)',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_ranking_stability_matrix(self, bootstrap_results: Dict[int, List[float]],
                                          save_path: str = None):
        """Heatmapa prawdopodobieństwa pozycji w rankingu"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Oblicz ranking dla każdej iteracji
        n_iterations = len(list(bootstrap_results.values())[0])
        n_finalists = len(actual_scores)
        ranking_matrix = np.zeros((n_finalists, n_finalists))
        
        print("Obliczam macierz stabilności rankingu...")
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteracja {iteration + 1}/{n_iterations}")
            
            # Wyniki w tej iteracji
            iter_scores = []
            for _, row in actual_scores.iterrows():
                nr = row['Nr']
                iter_scores.append({
                    'nr': nr,
                    'score': bootstrap_results[nr][iteration]
                })
            
            # Posortuj i przypisz rangi
            iter_df = pd.DataFrame(iter_scores)
            iter_df = iter_df.sort_values('score', ascending=False).reset_index(drop=True)
            iter_df['bootstrap_rank'] = range(1, len(iter_df) + 1)
            
            # Zlicz wystąpienia
            for idx, (_, row) in enumerate(actual_scores.iterrows()):
                nr = row['Nr']
                bootstrap_rank = iter_df[iter_df['nr'] == nr]['bootstrap_rank'].iloc[0]
                ranking_matrix[idx, bootstrap_rank - 1] += 1
        
        # Normalizuj do prawdopodobieństw
        ranking_matrix = ranking_matrix / n_iterations * 100
        
        # Rysuj
        fig, ax = plt.subplots(figsize=(16, 12))
        
        participant_labels = [f"{row['rank']}. {row['nazwisko']}" 
                             for _, row in actual_scores.iterrows()]
        
        sns.heatmap(ranking_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=range(1, n_finalists + 1),
                   yticklabels=participant_labels,
                   cbar_kws={'label': 'Prawdopodobieństwo (%)'},
                   ax=ax)
        
        ax.set_xlabel('Możliwa pozycja w rankingu', fontsize=12)
        ax.set_ylabel('Uczestnik (rzeczywista pozycja. nazwisko)', fontsize=12)
        ax.set_title('Macierz stabilności rankingu - Perturbacja ocen\n' +
                    f'Prawdopodobieństwo pozycji przy zmianach ocen o {self.analyzer.perturbation_values}',
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_score_vs_uncertainty(self, bootstrap_results: Dict[int, List[float]],
                                      save_path: str = None):
        """Scatter: wynik vs niepewność"""
        actual_scores = self.analyzer.get_actual_final_scores()
        
        scatter_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            scores = bootstrap_results[nr]
            std = np.std(scores)
            
            scatter_data.append({
                'participant': f"{row['imię']} {row['nazwisko']}",
                'rank': row['rank'],
                'actual_score': row['final_score'],
                'std': std
            })
        
        df = pd.DataFrame(scatter_data)
        
        # Rysuj
        fig, ax = plt.subplots(figsize=(14, 10))
        
        scatter = ax.scatter(df['actual_score'], df['std'], 
                           s=150, alpha=0.6, c=df['rank'], 
                           cmap='viridis_r', edgecolors='black', linewidth=0.5)
        
        # Annotacje
        top_10 = df[df['rank'] <= 10]
        most_unstable = df.nlargest(5, 'std')
        to_annotate = pd.concat([top_10, most_unstable]).drop_duplicates()
        
        for _, row in to_annotate.iterrows():
            ax.annotate(f"{row['rank']}. {row['participant'].split()[-1]}", 
                       (row['actual_score'], row['std']),
                       fontsize=8, alpha=0.8, 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Rzeczywisty wynik końcowy', fontsize=12)
        ax.set_ylabel('Odchylenie standardowe (wrażliwość)', fontsize=12)
        ax.set_title('Wynik vs Wrażliwość na perturbacje\n' +
                    f'Zmiany ocen o {self.analyzer.perturbation_values}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Pozycja w rankingu')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_comparison_with_resampling(self, 
                                            perturbation_results: Dict[int, List[float]],
                                            resampling_results: Dict[int, List[float]] = None,
                                            save_path: str = None):
        """
        Porównanie stabilności: perturbacja vs resampling sędziów
        (jeśli dostępne są oba wyniki)
        """
        if resampling_results is None:
            print("Brak wyników z resamplingu - pomijam porównanie")
            return
        
        actual_scores = self.analyzer.get_actual_final_scores()
        
        comparison_data = []
        for _, row in actual_scores.iterrows():
            nr = row['Nr']
            
            pert_scores = perturbation_results[nr]
            resamp_scores = resampling_results[nr]
            
            comparison_data.append({
                'rank': row['rank'],
                'participant': f"{row['imię']} {row['nazwisko']}",
                'perturbation_std': np.std(pert_scores),
                'resampling_std': np.std(resamp_scores),
                'perturbation_ci_width': np.percentile(pert_scores, 97.5) - np.percentile(pert_scores, 2.5),
                'resampling_ci_width': np.percentile(resamp_scores, 97.5) - np.percentile(resamp_scores, 2.5)
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Rysuj
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        # 1. Scatter: perturbacja vs resampling (SD)
        ax1 = axes[0]
        scatter = ax1.scatter(comp_df['resampling_std'], comp_df['perturbation_std'],
                            s=150, alpha=0.6, c=comp_df['rank'], 
                            cmap='viridis_r', edgecolors='black', linewidth=0.5)
        
        # Linia 1:1
        max_val = max(comp_df['resampling_std'].max(), comp_df['perturbation_std'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        # Annotacje dla top 5
        for _, row in comp_df.head(5).iterrows():
            ax1.annotate(f"{row['rank']}", 
                        (row['resampling_std'], row['perturbation_std']),
                        fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('SD - Resampling sędziów', fontsize=12)
        ax1.set_ylabel('SD - Perturbacja ocen', fontsize=12)
        ax1.set_title('Porównanie wrażliwości\nOdchylenie standardowe', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar plot: szerokość CI
        ax2 = axes[1]
        
        top_10 = comp_df.head(10)
        x = np.arange(len(top_10))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, top_10['resampling_ci_width'], width,
                       label='Resampling sędziów', alpha=0.7, color='steelblue')
        bars2 = ax2.bar(x + width/2, top_10['perturbation_ci_width'], width,
                       label='Perturbacja ocen', alpha=0.7, color='coral')
        
        ax2.set_xlabel('Uczestnik', fontsize=12)
        ax2.set_ylabel('Szerokość 95% CI (punkty)', fontsize=12)
        ax2.set_title('Porównanie wrażliwości - Top 10\nSzerokość przedziału ufności', 
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{row['rank']}." for _, row in top_10.iterrows()])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.colorbar(scatter, ax=axes, label='Pozycja w rankingu')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_full_report(self, bootstrap_results: Dict[int, List[float]],
                          resampling_results: Dict[int, List[float]] = None,
                          output_dir: str = 'perturbation_stability'):
        """Tworzy kompletny raport stabilności z perturbacją"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENEROWANIE RAPORTU STABILNOŚCI - PERTURBACJA OCEN")
        print(f"{'='*60}\n")
        
        print("1/5 Rozkłady wyników...")
        self.visualize_score_distributions(
            bootstrap_results,
            save_path=f'{output_dir}/38_perturbation_distributions.png'
        )
        
        print("\n2/5 Przedziały ufności...")
        self.visualize_confidence_intervals(
            bootstrap_results, confidence=0.95,
            save_path=f'{output_dir}/39_perturbation_confidence.png'
        )
        
        print("\n3/5 Macierz stabilności rankingu...")
        self.visualize_ranking_stability_matrix(
            bootstrap_results,
            save_path=f'{output_dir}/40_perturbation_ranking_matrix.png'
        )
        
        print("\n4/5 Wynik vs wrażliwość...")
        self.visualize_score_vs_uncertainty(
            bootstrap_results,
            save_path=f'{output_dir}/41_perturbation_score_vs_uncertainty.png'
        )
        
        if resampling_results:
            print("\n5/5 Porównanie: perturbacja vs resampling...")
            self.visualize_comparison_with_resampling(
                bootstrap_results,
                resampling_results,
                save_path=f'{output_dir}/42_perturbation_vs_resampling.png'
            )
        else:
            print("\n5/5 Brak danych resamplingu - pomijam porównanie")
        
        print(f"\n{'='*60}")
        print(f"RAPORT ZAKOŃCZONY!")
        print(f"Wszystkie wizualizacje zapisane w: {output_dir}/")
        print(f"{'='*60}\n")


def main():
    """Główna funkcja - przykład użycia"""
    
    # Ścieżki do plików
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    # Inicjalizacja analizatora
    print("="*60)
    print("ANALIZA STABILNOŚCI - PERTURBACJA OCEN")
    print("="*60)
    
    # Możesz zmienić wartości perturbacji
    perturbation_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    analyzer = ScorePerturbationAnalyzer(stage_files, perturbation_values)
    
    # Bootstrap z perturbacją
    n_iterations = 10000
    bootstrap_results = analyzer.bootstrap_with_perturbation(n_iterations=n_iterations)
    
    # Wizualizacje
    visualizer = PerturbationVisualizer(analyzer)
    visualizer.create_full_report(
        bootstrap_results, 
        output_dir='visualizations_perturbation'
    )
    
    # Statystyki
    print("\nSTATYSTYKI WRAŻLIWOŚCI NA PERTURBACJE:")
    print("-" * 60)
    
    actual_scores = analyzer.get_actual_final_scores()
    
    for _, row in actual_scores.head(10).iterrows():
        nr = row['Nr']
        scores = bootstrap_results[nr]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_95 = [np.percentile(scores, 2.5), np.percentile(scores, 97.5)]
        
        print(f"\n{row['rank']:2d}. {row['imię']} {row['nazwisko']}")
        print(f"    Rzeczywisty wynik: {row['final_score']:.2f}")
        print(f"    Średnia z perturbacji: {mean_score:.2f} (SD: {std_score:.2f})")
        print(f"    95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
        print(f"    Maksymalna zmiana: ±{(ci_95[1] - ci_95[0])/2:.2f} punktów")


if __name__ == "__main__":
    main()
