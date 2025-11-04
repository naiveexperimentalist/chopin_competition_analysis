"""
Moduł do wizualizacji zaawansowanych analiz statystycznych i clusteringowych
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChopinStatisticalVisualization:
    """Wizualizacje dla analiz statystycznych i clusteringowych"""
    
    def __init__(self, processor, controversy_analyzer=None, 
                 statistical_analyzer=None, clustering_analyzer=None):
        self.processor = processor
        self.controversy_analyzer = controversy_analyzer
        self.statistical_analyzer = statistical_analyzer
        self.clustering_analyzer = clustering_analyzer
    
    def visualize_controversy_heatmap(self, stage: str = 'final', save_path: str = None):
        """
        Heatmapa zróżnicowania ocen: uczestnicy vs odchylenia od średniej dla każdego sędziego
        """
        if not self.controversy_analyzer:
            print("Brak controversy_analyzer")
            return
        
        heatmap_data = self.controversy_analyzer.create_controversy_heatmap_data(stage)
        
        if heatmap_data.empty:
            print("Brak danych do heatmapy")
            return
        
        # Oblicz SD dla każdego uczestnika
        participant_stds = heatmap_data.std(axis=1)
        
        # Sortuj uczestników po SD (największe zróżnicowanie na górze)
        heatmap_sorted = heatmap_data.loc[participant_stds.sort_values(ascending=False).index]
        
        fig, ax = plt.subplots(figsize=(16, max(12, len(heatmap_sorted) * 0.3)))
        
        # Heatmapa
        sns.heatmap(heatmap_sorted, cmap='RdBu_r', center=0, 
                   annot=False, fmt='.1f',
                   cbar_kws={'label': 'Odchylenie od średniej oceny uczestnika'},
                   ax=ax, vmin=-5, vmax=5)
        
        ax.set_title(f'Zróżnicowanie ocen uczestników - {stage}\n' + 
                    'Odchylenia ocen sędziów od średniej dla każdego uczestnika\n' +
                    '(im większe różnice, tym bardziej zróżnicowane opinie)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('Uczestnik', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_controversy_analysis(self, save_path: str = None):
        """
        Kompleksowa wizualizacja kontrowersyjności uczestników
        """
        if not self.controversy_analyzer:
            print("Brak controversy_analyzer")
            return
        
        controversy = self.controversy_analyzer.analyze_participant_controversy()
        
        if controversy.empty:
            print("Brak danych kontrowersyjności")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top uczestników z najbardziej zróżnicowanymi ocenami
        ax1 = axes[0, 0]
        top_controversial = controversy.nlargest(15, 'std')
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_controversial)))
        bars = ax1.barh(range(len(top_controversial)), top_controversial['std'], color=colors)
        ax1.set_yticks(range(len(top_controversial)))
        ax1.set_yticklabels([f"{row['imię']} {row['nazwisko']}" for _, row in top_controversial.iterrows()],
                           fontsize=9)
        ax1.set_xlabel('Odchylenie standardowe ocen', fontsize=12)
        ax1.set_title('Top 15 uczestników z najbardziej zróżnicowanymi ocenami\n(wysokie SD = duże różnice między sędziami)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Rozpiętość vs średnia ocena
        ax2 = axes[0, 1]
        scatter = ax2.scatter(controversy['mean'], controversy['range'],
                            s=controversy['std']*50, alpha=0.6,
                            c=controversy['std'], cmap='Reds')
        ax2.set_xlabel('Średnia ocena', fontsize=12)
        ax2.set_ylabel('Rozpiętość ocen (max - min)', fontsize=12)
        ax2.set_title('Średnia vs Rozpiętość\n(wielkość = SD)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='SD ocen')
        
        # 3. Rozkład kontrowersyjności
        ax3 = axes[1, 0]
        ax3.hist(controversy['std'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax3.axvline(controversy['std'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Średnia: {controversy["std"].mean():.2f}')
        ax3.axvline(controversy['std'].median(), color='blue', linestyle='--',
                   linewidth=2, label=f'Mediana: {controversy["std"].median():.2f}')
        ax3.set_xlabel('Odchylenie standardowe ocen', fontsize=12)
        ax3.set_ylabel('Liczba uczestników', fontsize=12)
        ax3.set_title('Rozkład kontrowersyjności wszystkich uczestników',
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Coefficient of variation
        ax4 = axes[1, 1]
        top_cv = controversy.nlargest(15, 'cv')
        colors4 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(top_cv)))
        bars4 = ax4.barh(range(len(top_cv)), top_cv['cv'], color=colors4)
        ax4.set_yticks(range(len(top_cv)))
        ax4.set_yticklabels([f"{row['participant_name']}" for _, row in top_cv.iterrows()],
                           fontsize=9)
        ax4.set_xlabel('Współczynnik zmienności (%)', fontsize=12)
        ax4.set_title('Top 15 uczestników wg CV\n(relatywna zmienność)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Analiza zróżnicowania ocen uczestników', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_outliers_analysis(self, save_path: str = None):
        """
        Wizualizacja analizy outlierów
        """
        if not self.controversy_analyzer:
            print("Brak controversy_analyzer")
            return
        
        outliers = self.controversy_analyzer.analyze_outliers()
        judge_outlier_stats = self.controversy_analyzer.get_outlier_statistics_by_judge()
        
        if outliers.empty:
            print("Brak outlierów")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Outliery per sędzia
        ax1 = axes[0, 0]
        if not judge_outlier_stats.empty:
            stats_sorted = judge_outlier_stats.sort_values('total_outliers', ascending=True)
            colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(stats_sorted)))
            
            y_pos = range(len(stats_sorted))
            ax1.barh(y_pos, stats_sorted['high_outliers'], label='High outliers', 
                    color='red', alpha=0.7)
            ax1.barh(y_pos, -stats_sorted['low_outliers'], label='Low outliers',
                    color='blue', alpha=0.7)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([j.split()[-1] for j in stats_sorted['judge']])
            ax1.set_xlabel('Liczba outlierów', fontsize=12)
            ax1.set_title('Liczba outlierów per sędzia\n(czerwone=wysokie, niebieskie=niskie)',
                         fontsize=14, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Najekstremalniejsze oceny
        ax2 = axes[0, 1]
        top_outliers = outliers.nlargest(20, 'extremeness')
        
        colors2 = ['red' if x == 'high' else 'blue' for x in top_outliers['outlier_type']]
        bars2 = ax2.barh(range(len(top_outliers)), top_outliers['deviation'].abs(),
                        color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(top_outliers)))
        labels = [f"{row['participant_name']} ({row['judge'].split()[0]} {row['judge'].split()[1][0]}.)" 
                 for _, row in top_outliers.iterrows()]
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('Wielkość odchylenia od średniej', fontsize=12)
        ax2.set_title('Top 20 najbardziej ekstremalnych ocen\n(czerwone=za wysokie, niebieskie=za niskie)',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rozkład odchyleń outlierów
        ax3 = axes[1, 0]
        ax3.hist(outliers['deviation'], bins=40, edgecolor='black', alpha=0.7,
                color='orange')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Odchylenie od średniej', fontsize=12)
        ax3.set_ylabel('Liczba outlierów', fontsize=12)
        ax3.set_title('Rozkład odchyleń dla wszystkich outlierów',
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Outliery przez etapy
        ax4 = axes[1, 1]
        
        # Prawidłowa kolejność etapów
        stage_order = ['stage1', 'stage2', 'stage3', 'final']
        stage_counts = outliers['stage'].value_counts()
        
        # Posortuj według zdefiniowanej kolejności
        ordered_stages = [s for s in stage_order if s in stage_counts.index]
        ordered_counts = [stage_counts[s] for s in ordered_stages]
        
        colors4 = plt.cm.viridis(np.linspace(0.2, 0.8, len(ordered_stages)))
        bars4 = ax4.bar(range(len(ordered_stages)), ordered_counts, color=colors4)
        ax4.set_xticks(range(len(ordered_stages)))
        ax4.set_xticklabels(ordered_stages, rotation=45)
        ax4.set_ylabel('Liczba outlierów', fontsize=12)
        ax4.set_title('Liczba outlierów w poszczególnych etapach',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Analiza outlierów', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_ranking_stability(self, stage: str = 'final', save_path: str = None):
        """
        Wizualizacja stabilności rankingu (bootstrap CI + Monte Carlo)
        """
        if not self.statistical_analyzer:
            print("Brak statistical_analyzer")
            return
        
        bootstrap_ci = self.statistical_analyzer.bootstrap_ranking_confidence(stage=stage, n_bootstrap=1000)
        mc_stability = self.statistical_analyzer.monte_carlo_ranking_stability(stage=stage, n_simulations=1000)
        
        if bootstrap_ci.empty or mc_stability.empty:
            print("Brak danych stabilności")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Bootstrap confidence intervals dla finalistów
        ax1 = axes[0, 0]
        # Weź wszystkich jeśli jest 15 lub mniej, inaczej top 15
        n_to_show = min(15, len(bootstrap_ci))
        top_boot = bootstrap_ci.head(n_to_show)
        
        y_pos = range(len(top_boot))
        ax1.errorbar(top_boot['original_score'], y_pos,
                    xerr=[(top_boot['original_score'] - top_boot['ci_lower']).values,
                          (top_boot['ci_upper'] - top_boot['original_score']).values],
                    fmt='o', capsize=5, capthick=2, markersize=6, color='blue', alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{int(row['rank'])}. {row['imię']} {row['nazwisko']}" 
                            for _, row in top_boot.iterrows()],
                           fontsize=9)
        ax1.set_xlabel('Ocena (95% CI)', fontsize=12)
        ax1.set_title(f'Bootstrap Confidence Intervals\nFinaliści (pokazano {n_to_show})',
                     fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 2. Niepewność rankingu
        ax2 = axes[0, 1]
        top_mc = mc_stability.head(n_to_show)
        
        colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_mc)))
        bars2 = ax2.barh(range(len(top_mc)), top_mc['rank_range_mc'], color=colors2)
        ax2.set_yticks(range(len(top_mc)))
        ax2.set_yticklabels([f"{int(row['original_rank'])}. {row['imię']} {row['nazwisko']}"
                            for _, row in top_mc.iterrows()],
                           fontsize=9)
        ax2.set_xlabel('Zakres możliwych miejsc (Monte Carlo)', fontsize=12)
        ax2.set_title(f'Niepewność pozycji w rankingu\nFinaliści (pokazano {n_to_show})',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Dodaj wartości na słupkach
        for i, (bar, val) in enumerate(zip(bars2, top_mc['rank_range_mc'])):
            if val > 0:
                ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{int(val)}', va='center', fontsize=8)
        
        # 3. Szerokość CI przez rankingi
        ax3 = axes[1, 0]
        ax3.scatter(bootstrap_ci['rank'], bootstrap_ci['ci_width'],
                   s=80, alpha=0.6, c=bootstrap_ci['rank'], cmap='coolwarm')
        ax3.set_xlabel('Miejsce w rankingu', fontsize=12)
        ax3.set_ylabel('Szerokość przedziału ufności', fontsize=12)
        ax3.set_title('Stabilność vs Pozycja w rankingu\n(mniejsza szerokość = większa pewność)',
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monte Carlo rank distributions dla TOP 5
        ax4 = axes[1, 1]
        top5_mc = mc_stability.head(5)
        
        for i, (_, row) in enumerate(top5_mc.iterrows()):
            # Symuluj rozkład
            mean_rank = row['mean_rank_mc']
            std_rank = row['std_rank_mc']
            
            # Aproximacja rozkładu normalnego
            x = np.linspace(max(1, mean_rank - 3*std_rank), 
                          mean_rank + 3*std_rank, 100)
            from scipy.stats import norm
            y = norm.pdf(x, mean_rank, std_rank)
            
            label = f"{int(row['original_rank'])}. {row['participant_name'].split()[-1]}"
            ax4.plot(x, y, label=label, linewidth=2, alpha=0.7)
            ax4.axvline(row['original_rank'], color=f'C{i}', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Miejsce w rankingu', fontsize=12)
        ax4.set_ylabel('Gęstość prawdopodobieństwa', fontsize=12)
        ax4.set_title('Rozkład możliwych miejsc (Monte Carlo)\nTOP 5',
                     fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Analiza stabilności rankingu - {stage}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_statistical_significance(self, stage: str = 'final', save_path: str = None):
        """
        Wizualizacja istotności statystycznej różnic między miejscami
        """
        if not self.statistical_analyzer:
            print("Brak statistical_analyzer")
            return
        
        significance = self.statistical_analyzer.statistical_significance_between_ranks(stage=stage)
        ties = self.statistical_analyzer.identify_statistical_ties(stage=stage, alpha=0.05)
        
        if significance.empty:
            print("Brak danych istotności")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. P-values dla kolejnych par
        ax1 = axes[0, 0]
        x_labels = [f"{int(row['rank1'])}-{int(row['rank2'])}" 
                   for _, row in significance.head(30).iterrows()]
        colors1 = ['green' if p < 0.05 else 'red' for p in significance.head(30)['p_value']]
        
        bars1 = ax1.bar(range(len(x_labels)), significance.head(30)['p_value'],
                       color=colors1, alpha=0.7)
        ax1.axhline(y=0.05, color='blue', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.set_xticks(range(len(x_labels)))
        ax1.set_xticklabels(x_labels, rotation=90, fontsize=8)
        ax1.set_ylabel('P-value', fontsize=12)
        ax1.set_title('Istotność różnic między sąsiednimi miejscami\n(zielone=istotne, czerwone=nieistotne)',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Effect size (Cohen's d)
        ax2 = axes[0, 1]
        cohens_d = significance.head(30)['cohens_d'].abs()
        colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cohens_d)))
        
        bars2 = ax2.bar(range(len(x_labels)), cohens_d, color=colors2, alpha=0.7)
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='small')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='medium')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='large')
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels, rotation=90, fontsize=8)
        ax2.set_ylabel("Cohen's d (effect size)", fontsize=12)
        ax2.set_title('Wielkość efektu różnic\n(wyższe = większa różnica)',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Różnice wyników vs p-value
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(significance['score_diff'], significance['p_value'],
                              s=100, alpha=0.6, 
                              c=significance['cohens_d'].abs(), cmap='viridis')
        ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax3.set_xlabel('Różnica wyników', fontsize=12)
        ax3.set_ylabel('P-value', fontsize=12)
        ax3.set_title('Różnica wyników vs Istotność statystyczna\n(kolor = effect size)',
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label="Cohen's d")
        
        # 4. Tabela ties
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if not ties.empty:
            table_data = []
            for _, row in ties.head(10).iterrows():
                table_data.append([
                    f"{row['tie_group_size']}",
                    row['ranks'],
                    row['participants'][:40] + '...' if len(row['participants']) > 40 else row['participants'],
                    f"{row['score_range']:.2f}"
                ])
            
            table = ax4.table(cellText=table_data,
                            colLabels=['Rozmiar\ngrupy', 'Miejsca', 'Uczestnicy', 'Zakres\nwyników'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.1, 0.15, 0.55, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            ax4.set_title('Grupy statystycznie nierozróżnialne (ties)\np > 0.05',
                         fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'Analiza istotności statystycznej - {stage}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_pairwise_agreement(self, save_path: str = None):
        """
        Wizualizacja pair-wise agreement (Kendall's tau)
        """
        if not self.statistical_analyzer:
            print("Brak statistical_analyzer")
            return
        
        kendall_results = self.statistical_analyzer.pairwise_agreement_kendall()
        
        if kendall_results.empty:
            print("Brak danych Kendall tau")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Kendall's tau vs Pearson r
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(kendall_results['pearson_r'], kendall_results['kendall_tau'],
                              s=100, alpha=0.6, c=kendall_results['n_common'], cmap='viridis')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax1.set_xlabel("Pearson's r", fontsize=12)
        ax1.set_ylabel("Kendall's tau", fontsize=12)
        ax1.set_title("Porównanie Kendall's tau vs Pearson's r\n(kolor = liczba wspólnych ocen)",
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='N wspólnych')
        
        # 2. Top agreements
        ax2 = axes[0, 1]
        top_agreements = kendall_results.head(15)
        
        colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_agreements)))
        bars2 = ax2.barh(range(len(top_agreements)), top_agreements['kendall_tau'],
                        color=colors2)
        ax2.set_yticks(range(len(top_agreements)))
        labels2 = [f"{row['judge1'].split()[-1]} - {row['judge2'].split()[-1]}"
                  for _, row in top_agreements.iterrows()]
        ax2.set_yticklabels(labels2, fontsize=9)
        ax2.set_xlabel("Kendall's tau", fontsize=12)
        ax2.set_title('Top 15 par sędziów o największej zgodności',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rozkład Kendall's tau
        ax3 = axes[1, 0]
        ax3.hist(kendall_results['kendall_tau'], bins=30, edgecolor='black',
                alpha=0.7, color='skyblue')
        ax3.axvline(kendall_results['kendall_tau'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Średnia: {kendall_results["kendall_tau"].mean():.3f}')
        ax3.axvline(kendall_results['kendall_tau'].median(), color='blue',
                   linestyle='--', linewidth=2,
                   label=f'Mediana: {kendall_results["kendall_tau"].median():.3f}')
        ax3.set_xlabel("Kendall's tau", fontsize=12)
        ax3.set_ylabel('Liczba par sędziów', fontsize=12)
        ax3.set_title("Rozkład zgodności między wszystkimi parami sędziów",
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Macierz agreement levels
        ax4 = axes[1, 1]
        
        # Zlicz agreement levels
        agreement_counts = kendall_results['agreement_level'].value_counts()
        colors4 = {'strong': 'green', 'moderate': 'orange', 'weak': 'red'}
        
        bars4 = ax4.bar(range(len(agreement_counts)), agreement_counts.values,
                       color=[colors4.get(level, 'gray') for level in agreement_counts.index],
                       alpha=0.7)
        ax4.set_xticks(range(len(agreement_counts)))
        ax4.set_xticklabels(agreement_counts.index)
        ax4.set_ylabel('Liczba par sędziów', fontsize=12)
        ax4.set_title('Klasyfikacja poziomu zgodności\n(strong: τ>0.7, moderate: τ>0.5, weak: τ≤0.5)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Dodaj liczby na słupkach
        for bar, val in zip(bars4, agreement_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.suptitle("Analiza zgodności rankingowej między sędziami (Kendall's tau)",
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_clustering_results(self, stage: str = 'final', save_path: str = None,
                                     multistage_finalists: bool = False):
        """
        Wizualizacja wyników clusteringu uczestników
        
        Args:
            stage: etap konkursu
            save_path: ścieżka zapisu
            multistage_finalists: jeśli True, używa ocen finalistów ze WSZYSTKICH etapów
        """
        if not self.clustering_analyzer:
            print("Brak clustering_analyzer")
            return
        
        try:
            clusters_df, cluster_stats, cluster_centers, scaler = \
                self.clustering_analyzer.kmeans_clustering_participants(
                    stage, n_clusters=5, multistage_finalists=multistage_finalists)
        except Exception as e:
            print(f"Błąd w clusteringu: {e}")
            return
        
        if clusters_df.empty:
            print("Brak danych clusteringu")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Rozmiary klastrów
        ax1 = axes[0, 0]
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        
        bars1 = ax1.bar(cluster_counts.index, cluster_counts.values, color=colors1, alpha=0.7)
        ax1.set_xlabel('Klaster', fontsize=12)
        ax1.set_ylabel('Liczba uczestników', fontsize=12)
        ax1.set_title('Rozmiary klastrów uczestników',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Dodaj liczby
        for bar, val in zip(bars1, cluster_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Średnie wyniki per klaster
        ax2 = axes[0, 1]
        cluster_means = clusters_df.groupby('cluster')['mean_score'].mean().sort_index()
        cluster_stds = clusters_df.groupby('cluster')['mean_score'].std().sort_index()
        
        bars2 = ax2.bar(cluster_means.index, cluster_means.values,
                       yerr=cluster_stds.values, capsize=5,
                       color=colors1, alpha=0.7)
        ax2.set_xlabel('Klaster', fontsize=12)
        ax2.set_ylabel('Średnia ocena', fontsize=12)
        ax2.set_title('Średnie oceny per klaster\n(error bars = SD)',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter: mean score vs std score, colored by cluster
        ax3 = axes[1, 0]
        for cluster_id in sorted(clusters_df['cluster'].unique()):
            cluster_data = clusters_df[clusters_df['cluster'] == cluster_id]
            ax3.scatter(cluster_data['mean_score'], cluster_data['std_score'],
                       label=f'Klaster {cluster_id}', s=100, alpha=0.6)
        
        ax3.set_xlabel('Średnia ocena uczestnika', fontsize=12)
        ax3.set_ylabel('SD ocen uczestnika', fontsize=12)
        ax3.set_title('Średnia vs Zmienność ocen (kolorowane po klastrach)',
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Lista wszystkich uczestników w każdym klastrze (alfabetycznie)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Przygotuj tekst z listą uczestników dla każdego klastra
        y_position = 0.95
        x_position = 0.05
        line_height = 0.05
        
        for cluster_id in sorted(clusters_df['cluster'].unique()):
            cluster_data = clusters_df[clusters_df['cluster'] == cluster_id]
            
            # Pobierz nazwiska i posortuj alfabetycznie
            participants = []
            for _, row in cluster_data.iterrows():
                # Wyciągnij samo nazwisko (część po ":")
                full_name = row['participant']
                if ':' in full_name:
                    name = full_name.split(':', 1)[1].strip()
                else:
                    name = full_name
                participants.append(name)
            
            participants.sort()  # Sortowanie alfabetyczne
            
            # Nagłówek klastra
            ax4.text(x_position, y_position, f"Klaster {cluster_id}:",
                    fontsize=11, fontweight='bold', 
                    transform=ax4.transAxes, va='top')
            y_position -= line_height
            
            # Lista uczestników
            for participant in participants:
                ax4.text(x_position + 0.02, y_position, f"• {participant}",
                        fontsize=9, transform=ax4.transAxes, va='top')
                y_position -= line_height * 0.8
            
            y_position -= line_height * 0.3  # Odstęp między klastrami
        
        ax4.set_title('Uczestnicy w każdym klastrze (alfabetycznie)',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Zmień tytuł w zależności od trybu
        if multistage_finalists:
            title = 'Analiza clusteringu uczestników - finaliści (wszystkie etapy)'
        else:
            title = f'Analiza clusteringu uczestników - {stage}'
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_pca_judges(self, save_path: str = None):
        """
        Wizualizacja PCA profili sędziów
        """
        if not self.clustering_analyzer:
            print("Brak clustering_analyzer")
            return
        
        try:
            pca, pca_df, variance_ratio = self.clustering_analyzer.pca_judge_profiles()
        except Exception as e:
            print(f"Błąd w PCA: {e}")
            return
        
        if pca_df.empty:
            print("Brak danych PCA")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scree plot - explained variance
        ax1 = axes[0, 0]
        n_components = len(variance_ratio)
        ax1.bar(range(1, n_components+1), variance_ratio, alpha=0.7, color='skyblue',
               label='Indywidualna')
        ax1.plot(range(1, n_components+1), np.cumsum(variance_ratio), 'ro-',
                linewidth=2, label='Skumulowana')
        ax1.set_xlabel('Komponent główny', fontsize=12)
        ax1.set_ylabel('Procent wyjaśnionej wariancji', fontsize=12)
        ax1.set_title('Scree Plot - Wyjaśniona wariancja',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Biplot PC1 vs PC2
        ax2 = axes[0, 1]
        ax2.scatter(pca_df['PC1'], pca_df['PC2'], s=100, alpha=0.6, c='coral')
        
        for _, row in pca_df.iterrows():
            ax2.annotate(row['judge'].split()[-1], 
                        (row['PC1'], row['PC2']),
                        fontsize=9, alpha=0.7)
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% wariancji)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}% wariancji)', fontsize=12)
        ax2.set_title('Biplot PC1 vs PC2\nProfile sędziów w przestrzeni głównych komponentów',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. PC1 vs PC3 (jeśli istnieje)
        ax3 = axes[1, 0]
        if 'PC3' in pca_df.columns:
            ax3.scatter(pca_df['PC1'], pca_df['PC3'], s=100, alpha=0.6, c='lightgreen')
            
            for _, row in pca_df.iterrows():
                ax3.annotate(row['judge'].split()[-1],
                            (row['PC1'], row['PC3']),
                            fontsize=9, alpha=0.7)
            
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% wariancji)', fontsize=12)
            ax3.set_ylabel(f'PC3 ({variance_ratio[2]*100:.1f}% wariancji)', fontsize=12)
            ax3.set_title('Biplot PC1 vs PC3',
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Tabela komponentów
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        cumsum = 0
        for i, var in enumerate(variance_ratio[:5]):
            cumsum += var
            table_data.append([
                f'PC{i+1}',
                f'{var*100:.2f}%',
                f'{cumsum*100:.2f}%'
            ])
        
        table = ax4.table(cellText=table_data,
                        colLabels=['Komponent', 'Wariancja', 'Skumulowana'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        ax4.set_title('Wyjaśniona wariancja per komponent',
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Analiza PCA profili sędziów',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_statistical_report(self, output_dir: str = 'visualizations'):
        """
        Tworzy kompletny raport wizualny dla analiz statystycznych
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generowanie kompleksowego raportu wizualnego analiz statystycznych...")
        
        # Zróżnicowanie ocen
        if self.controversy_analyzer:
            print("  - Heatmapa zróżnicowania ocen")
            self.visualize_controversy_heatmap('final', f'{output_dir}/16_score_diversity_heatmap.png')
            
            print("  - Analiza zróżnicowania ocen")
            self.visualize_controversy_analysis(f'{output_dir}/17_score_diversity_analysis.png')
            
            print("  - Analiza outlierów")
            self.visualize_outliers_analysis(f'{output_dir}/18_outliers_analysis.png')
        
        # Stabilność rankingu
        if self.statistical_analyzer:
            # print("  - Stabilność rankingu")
            # self.visualize_ranking_stability('final', f'{output_dir}/19_ranking_stability.png')
            
            # print("  - Istotność statystyczna")
            # self.visualize_statistical_significance('final', f'{output_dir}/20_statistical_significance.png')
            
            print("  - Pair-wise agreement")
            self.visualize_pairwise_agreement(f'{output_dir}/21_pairwise_agreement.png')
        
        # Clustering
        if self.clustering_analyzer:
            print("  - Wyniki clusteringu")
            self.visualize_clustering_results('final', f'{output_dir}/22_clustering_results.png')
            
            print("  - PCA sędziów")
            self.visualize_pca_judges(f'{output_dir}/23_pca_judges.png')

        print(f"\nWizualizacje statystyczne zapisane w: {output_dir}")


if __name__ == "__main__":
    print("Uruchom ten moduł po wczytaniu danych i analiz")
