"""
Analiza stabilności wyniku końcowego konkursu Chopinowskiego
Bootstrap resampling wszystkich etapów z obliczaniem wyniku końcowego
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


class FinalScoreStabilityAnalyzer:
    """Analiza stabilności wyniku końcowego metodą bootstrap"""
    
    def __init__(self, stage_files: Dict[str, str]):
        """
        Args:
            stage_files: słownik {nazwa_etapu: ścieżka_do_pliku}
                       np. {'stage1': 'chopin_2025_stage1_by_judge.csv', ...}
        """
        self.stage_files = stage_files
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
            
            # Identyfikuj kolumny sędziów (wszystkie oprócz Nr, imię, nazwisko)
            if not self.judge_columns:
                self.judge_columns = [col for col in df.columns 
                                     if col not in ['Nr', 'imię', 'nazwisko']]
            
            # Konwertuj oceny (zamień 's' na NaN)
            for judge in self.judge_columns:
                df[judge] = pd.to_numeric(df[judge], errors='coerce')
            
            self.stages_data[stage] = df
            print(f"  {stage}: {len(df)} uczestników, {len(self.judge_columns)} sędziów")
    
    def calculate_stage_score_for_participant(self, stage: str, participant_nr: int, 
                                              judge_subset: List[str] = None) -> float:
        """
        Oblicza wynik uczestnika w danym etapie
        
        Args:
            stage: nazwa etapu
            participant_nr: numer uczestnika
            judge_subset: lista sędziów do uwzględnienia (None = wszyscy)
        
        Returns:
            skorygowana średnia ocena
        """
        df = self.stages_data[stage]
        participant = df[df['Nr'] == participant_nr]
        
        if participant.empty:
            return 0.0
        
        judges = judge_subset if judge_subset else self.judge_columns
        
        # Pobierz oceny (pomijając NaN)
        scores = []
        for judge in judges:
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
                                       judge_subsets: Dict[str, List[str]] = None) -> float:
        """
        Oblicza końcowy ważony wynik uczestnika
        
        Args:
            participant_nr: numer uczestnika
            judge_subsets: słownik {etap: lista_sędziów} dla każdego etapu
        
        Returns:
            końcowy ważony wynik
        """
        if judge_subsets is None:
            judge_subsets = {stage: self.judge_columns for stage in self.stage_files.keys()}
        
        # Sprawdź, czy uczestnik wystąpił we wszystkich potrzebnych etapach
        available_stages = []
        for stage in self.stage_files.keys():
            if participant_nr in self.stages_data[stage]['Nr'].values:
                available_stages.append(stage)
        
        if not available_stages:
            return 0.0
        
        # Oblicz wynik z każdego etapu
        stage_scores = {}
        for stage in available_stages:
            stage_scores[stage] = self.calculate_stage_score_for_participant(
                stage, participant_nr, judge_subsets.get(stage)
            )
        
        # Oblicz ważoną średnią (tylko z dostępnych etapów)
        weighted_sum = 0.0
        total_weight = 0.0
        
        for stage in available_stages:
            weight = self.weights[stage]
            weighted_sum += stage_scores[stage] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalizuj do pełnej wagi
        return weighted_sum / total_weight * sum(self.weights.values())
    
    def bootstrap_final_scores(self, n_iterations: int = 10000) -> Dict[int, List[float]]:
        """
        Bootstrap resampling - losowanie sędziów w każdym etapie i obliczanie wyniku końcowego
        
        Args:
            n_iterations: liczba iteracji bootstrapu
        
        Returns:
            słownik {nr_uczestnika: lista_wyników_z_iteracji}
        """
        print(f"\nRozpoczynanie bootstrapu: {n_iterations} iteracji...")
        
        # Znajdź wszystkich uczestników finału (to oni mają wynik końcowy)
        final_participants = self.stages_data['final']['Nr'].values
        
        bootstrap_results = {nr: [] for nr in final_participants}
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteracja {iteration + 1}/{n_iterations}")
            
            # Dla każdego etapu wylosuj podzbiór sędziów (z powtórzeniami)
            judge_subsets = {}
            for stage in self.stage_files.keys():
                n_judges = len(self.judge_columns)
                # Losuj n_judges sędziów z powtórzeniami
                judge_subsets[stage] = list(np.random.choice(
                    self.judge_columns, size=n_judges, replace=True
                ))
            
            # Oblicz wynik końcowy dla każdego uczestnika przy tym losowaniu
            for participant_nr in final_participants:
                final_score = self.calculate_final_weighted_score(
                    participant_nr, judge_subsets
                )
                bootstrap_results[participant_nr].append(final_score)
        
        print("Bootstrap zakończony!")
        return bootstrap_results
    
    def get_actual_final_scores(self) -> pd.DataFrame:
        """
        Oblicza rzeczywiste wyniki końcowe (bez bootstrapu)
        
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


class FinalScoreStabilityVisualizer:
    """Wizualizacje stabilności wyniku końcowego"""
    
    def __init__(self, analyzer: FinalScoreStabilityAnalyzer):
        self.analyzer = analyzer
    
    def visualize_score_distributions(self, bootstrap_results: Dict[int, List[float]],
                                     save_path: str = None):
        """
        Violin plots rozkładów wyników końcowych dla wszystkich finalistów
        
        Args:
            bootstrap_results: wyniki bootstrapu
            save_path: ścieżka do zapisu
        """
        # Pobierz rzeczywiste wyniki i nazwy
        actual_scores = self.analyzer.get_actual_final_scores()
        top_participants = actual_scores  # wszyscy finaliści
        
        # Przygotuj dane do violin plot
        plot_data = []
        for _, row in top_participants.iterrows():
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
        fig, ax = plt.subplots(figsize=(16, max(10, len(top_participants) * 0.4)))
        
        # Sortuj po rankingu
        order = [f"{r}. {n}" for r, n in 
                zip(top_participants['rank'], top_participants['nazwisko'])]
        
        # Violin plot
        sns.violinplot(data=df, y='participant', x='score', order=order, 
                      inner='quartile', ax=ax, palette='Set2')
        
        # Dodaj czerwone punkty dla rzeczywistych wyników
        for i, (_, row) in enumerate(top_participants.iterrows()):
            ax.plot(row['final_score'], i, 'ro', markersize=8, 
                   label='Rzeczywisty wynik' if i == 0 else '')
        
        ax.set_xlabel('Wynik końcowy', fontsize=12)
        ax.set_ylabel('Uczestnik (ranga. nazwisko)', fontsize=12)
        ax.set_title('Stabilność wyniku końcowego - Finaliści\n' +
                    'Rozkład możliwych wyników z bootstrapu (losowanie sędziów we wszystkich etapach)',
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
        """
        Wizualizacja przedziałów ufności dla wyników końcowych
        
        Args:
            bootstrap_results: wyniki bootstrapu
            confidence: poziom ufności (np. 0.95)
            save_path: ścieżka do zapisu
        """
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
        
        # 1. Error bars dla przedziałów ufności
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
        ax1.set_title(f'Przedziały ufności ({confidence*100:.0f}%) dla wyniku końcowego\n' +
                     'Czerwony punkt = rzeczywisty wynik, linie = możliwy zakres',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # 2. Szerokość przedziału ufności (niestabilność)
        ax2 = axes[1]
        
        top_unstable = ci_df.nlargest(30, 'ci_width')
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_unstable)))
        
        bars = ax2.barh(range(len(top_unstable)), top_unstable['ci_width'], 
                       color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_unstable)))
        ax2.set_yticklabels([f"{row['rank']}. {row['participant'][:30]}" 
                            for _, row in top_unstable.iterrows()], fontsize=8)
        ax2.set_xlabel('Szerokość przedziału ufności (punkty)', fontsize=12)
        ax2.set_ylabel('Uczestnik', fontsize=12)
        ax2.set_title('Top 30 najbardziej niestabilnych wyników\n' +
                     '(im szerszy przedział, tym większa niepewność)',
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
        """
        Heatmapa prawdopodobieństwa zajęcia danej pozycji w rankingu
        
        Args:
            bootstrap_results: wyniki bootstrapu
            save_path: ścieżka do zapisu
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        top_participants = actual_scores  # wszyscy finaliści
        
        # Dla każdej iteracji bootstrapu, oblicz ranking
        n_iterations = len(list(bootstrap_results.values())[0])
        n_finalists = len(top_participants)
        ranking_matrix = np.zeros((n_finalists, n_finalists))
        
        print("Obliczam macierz stabilności rankingu...")
        
        for iteration in range(n_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"  Iteracja {iteration + 1}/{n_iterations}")
            
            # Pobierz wyniki wszystkich uczestników w tej iteracji
            iter_scores = []
            for _, row in top_participants.iterrows():
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
            for idx, (_, row) in enumerate(top_participants.iterrows()):
                nr = row['Nr']
                bootstrap_rank = iter_df[iter_df['nr'] == nr]['bootstrap_rank'].iloc[0]
                ranking_matrix[idx, bootstrap_rank - 1] += 1
        
        # Normalizuj do prawdopodobieństw
        ranking_matrix = ranking_matrix / n_iterations * 100
        
        # Rysuj heatmapę
        fig, ax = plt.subplots(figsize=(16, 12))
        
        participant_labels = [f"{row['rank']}. {row['nazwisko']}" 
                             for _, row in top_participants.iterrows()]
        
        sns.heatmap(ranking_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=range(1, n_finalists + 1),
                   yticklabels=participant_labels,
                   cbar_kws={'label': 'Prawdopodobieństwo (%)'},
                   ax=ax)
        
        ax.set_xlabel('Możliwa pozycja w rankingu', fontsize=12)
        ax.set_ylabel('Uczestnik (rzeczywista pozycja. nazwisko)', fontsize=12)
        ax.set_title('Macierz stabilności rankingu - Finaliści\n' +
                    'Prawdopodobieństwo zajęcia danej pozycji w bootstrapie',
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
        """
        Scatter plot: wynik końcowy vs niepewność (SD)
        
        Args:
            bootstrap_results: wyniki bootstrapu
            save_path: ścieżka do zapisu
        """
        actual_scores = self.analyzer.get_actual_final_scores()
        
        # Oblicz SD dla każdego uczestnika
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
        
        # Scatter plot z kolorami wg rankingu
        scatter = ax.scatter(df['actual_score'], df['std'], 
                           s=150, alpha=0.6, c=df['rank'], 
                           cmap='viridis_r', edgecolors='black', linewidth=0.5)
        
        # Annotacje dla top 10 i najbardziej niestabilnych
        top_10 = df[df['rank'] <= 10]
        most_unstable = df.nlargest(5, 'std')
        to_annotate = pd.concat([top_10, most_unstable]).drop_duplicates()
        
        for _, row in to_annotate.iterrows():
            ax.annotate(f"{row['rank']}. {row['participant'].split()[-1]}", 
                       (row['actual_score'], row['std']),
                       fontsize=8, alpha=0.8, 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Rzeczywisty wynik końcowy', fontsize=12)
        ax.set_ylabel('Odchylenie standardowe (niepewność)', fontsize=12)
        ax.set_title('Wynik końcowy vs Niepewność\n' +
                    'Kolor = pozycja w rankingu (ciemniejszy = wyżej)',
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
    
    def create_full_stability_report(self, bootstrap_results: Dict[int, List[float]],
                                    output_dir: str = 'final_score_stability'):
        """
        Tworzy kompletny raport stabilności wyniku końcowego
        
        Args:
            bootstrap_results: wyniki bootstrapu
            output_dir: katalog wyjściowy
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENEROWANIE RAPORTU STABILNOŚCI WYNIKU KOŃCOWEGO")
        print(f"{'='*60}\n")
        
        print("1/4 Rozkłady wyników (violin plots)...")
        self.visualize_score_distributions(
            bootstrap_results,
            save_path=f'{output_dir}/34_score_distributions.png'
        )
        
        print("\n2/4 Przedziały ufności...")
        self.visualize_confidence_intervals(
            bootstrap_results, confidence=0.95,
            save_path=f'{output_dir}/35_confidence_intervals.png'
        )
        
        print("\n3/4 Macierz stabilności rankingu...")
        self.visualize_ranking_stability_matrix(
            bootstrap_results,
            save_path=f'{output_dir}/36_ranking_stability_matrix.png'
        )
        
        print("\n4/4 Wynik vs niepewność...")
        self.visualize_score_vs_uncertainty(
            bootstrap_results,
            save_path=f'{output_dir}/37_score_vs_uncertainty.png'
        )
        
        print(f"\n{'='*60}")
        print(f"RAPORT ZAKOŃCZONY!")
        print(f"Wszystkie wizualizacje zapisane w: {output_dir}/")
        print(f"{'='*60}\n")


def main():
    """Główna funkcja - przykład użycia"""
    
    # Ścieżki do plików (dostosuj do swoich plików)
    stage_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    # Inicjalizacja analizatora
    print("="*60)
    print("ANALIZA STABILNOŚCI WYNIKU KOŃCOWEGO KONKURSU CHOPINOWSKIEGO")
    print("="*60)
    
    analyzer = FinalScoreStabilityAnalyzer(stage_files)
    
    # Bootstrap (10000 iteracji - może potrwać kilka minut)
    n_iterations = 10000
    bootstrap_results = analyzer.bootstrap_final_scores(n_iterations=n_iterations)
    
    # Wizualizacje
    visualizer = FinalScoreStabilityVisualizer(analyzer)
    visualizer.create_full_stability_report(
        bootstrap_results, 
        output_dir='visualizations'
    )
    
    # Wyświetl statystyki
    print("\nSTATYSTYKI STABILNOŚCI:")
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
        print(f"    Średnia bootstrap:  {mean_score:.2f} (SD: {std_score:.2f})")
        print(f"    95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
        print(f"    Szerokość CI: {ci_95[1] - ci_95[0]:.2f} punktów")


if __name__ == "__main__":
    main()
