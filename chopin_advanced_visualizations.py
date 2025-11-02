"""
Zaawansowane wizualizacje analizy wieloetapowej konkursu Chopinowskiego

Wizualizacje PCA, heatmapy, trajektorie uczestników i głębsze analizy klastrów
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_pca_participants(analyzer, n_components=3, min_stages=2, 
                               save_path='participant_pca.png'):
    """
    PCA dla uczestników - projekcja w przestrzeni 2D/3D
    Pokazuje klastry uczestników w zredukowanej przestrzeni
    """
    profile_df, participant_info = analyzer.build_participant_multistage_matrix()
    
    # Filtruj uczestników
    valid_participants = [nr for nr, info in participant_info.items() 
                         if len(info['stages']) >= min_stages]
    profile_df = profile_df.loc[valid_participants]
    profile_df = profile_df.dropna(axis=1, how='all')
    
    # Usuń uczestników z zbyt małą liczbą ocen
    min_valid_scores = len(profile_df.columns) * 0.3
    valid_rows = profile_df.count(axis=1) >= min_valid_scores
    profile_df = profile_df[valid_rows]
    
    if len(profile_df) < 5:
        print("Za mało uczestników dla PCA")
        return None, None, pd.DataFrame()
    
    # Wypełnij braki
    profile_df_filled = profile_df.fillna(profile_df.mean())
    if profile_df_filled.isnull().any().any():
        profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())
    
    # Standaryzacja
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df_filled)
    
    # Sprawdź wartości nieskończone
    if not np.isfinite(profile_scaled).all():
        print("UWAGA: Zamiana wartości nieskończonych na 0")
        profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PCA
    n_components = min(n_components, len(profile_df_filled), len(profile_df_filled.columns))
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(profile_scaled)
    
    # K-means dla kolorowania
    from sklearn.cluster import KMeans
    n_clusters = min(5, len(profile_df_filled) // 2)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
    else:
        clusters = np.zeros(len(profile_df_filled), dtype=int)
    
    # Wizualizacja
    if n_components >= 2:
        fig = plt.figure(figsize=(18, 6))
        
        # 2D scatter (PC1 vs PC2)
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(pca_components[:, 0], pca_components[:, 1],
                            c=clusters, cmap='Set1', s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
        ax1.set_title('PCA: PC1 vs PC2', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Klaster')
        
        # Dodaj etykiety dla wybranych punktów (finaliści)
        finalists = [nr for nr in valid_participants 
                    if 'final' in participant_info[nr]['stages']]
        for i, nr in enumerate(profile_df_filled.index):
            if nr in finalists:
                info = participant_info[nr]
                ax1.annotate(f"{nr}", 
                           (pca_components[i, 0], pca_components[i, 1]),
                           fontsize=8, alpha=0.7)
        
        if n_components >= 3:
            # 2D scatter (PC1 vs PC3)
            ax2 = fig.add_subplot(132)
            scatter2 = ax2.scatter(pca_components[:, 0], pca_components[:, 2],
                                 c=clusters, cmap='Set1', s=100, alpha=0.7, edgecolors='black')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
            ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} wariancji)', fontsize=12)
            ax2.set_title('PCA: PC1 vs PC3', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Klaster')
            
            # 2D scatter (PC2 vs PC3)
            ax3 = fig.add_subplot(133)
            scatter3 = ax3.scatter(pca_components[:, 1], pca_components[:, 2],
                                 c=clusters, cmap='Set1', s=100, alpha=0.7, edgecolors='black')
            ax3.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
            ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} wariancji)', fontsize=12)
            ax3.set_title('PCA: PC2 vs PC3', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax3, label='Klaster')
    
    plt.suptitle('Analiza PCA uczestników\n' +
                'Projekcja profili wieloetapowych w przestrzeni głównych komponentów',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano PCA uczestników: {save_path}")
    
    # Zwróć wyjaśnioną wariancję
    variance_df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(n_components)],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })
    
    return pca, pca_components, variance_df


def visualize_pca_judges(analyzer, save_path='judge_pca.png'):
    """
    PCA dla sędziów - pokazuje podobieństwa w stylach oceniania
    """
    profile_df = analyzer.build_judge_multistage_matrix()
    profile_df = profile_df.dropna(axis=1, how='all')
    
    # Usuń sędziów z zbyt małą liczbą ocen
    min_valid_scores = len(profile_df.columns) * 0.3
    valid_judges = profile_df.count(axis=1) >= min_valid_scores
    profile_df = profile_df[valid_judges]
    
    if len(profile_df) < 3:
        print("Za mało sędziów dla PCA")
        return
    
    # Wypełnij braki
    profile_df_filled = profile_df.fillna(profile_df.mean(axis=1), axis=0)
    if profile_df_filled.isnull().any().any():
        profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())
    
    # Standaryzacja
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df_filled)
    
    # Sprawdź wartości nieskończone
    if not np.isfinite(profile_scaled).all():
        print("UWAGA: Zamiana wartości nieskończonych na 0")
        profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PCA
    n_components = min(3, len(profile_df_filled))
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(profile_scaled)
    
    # K-means dla kolorowania
    from sklearn.cluster import KMeans
    n_clusters = min(3, len(profile_df_filled) // 2)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
    else:
        clusters = np.zeros(len(profile_df_filled), dtype=int)
    
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(14, 10))
    
    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                        c=clusters, cmap='Set1', s=300, alpha=0.7, edgecolors='black')
    
    # Etykiety sędziów
    for i, judge in enumerate(profile_df_filled.index):
        ax.annotate(judge, 
                   (pca_components[i, 0], pca_components[i, 1]),
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)', fontsize=12)
    ax.set_title('Analiza PCA sędziów\n' +
                'Podobieństwa w stylach oceniania przez wszystkie etapy',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Klaster')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano PCA sędziów: {save_path}")


def visualize_multistage_heatmap(analyzer, save_path='multistage_heatmap.png'):
    """
    Heatmapa wszystkich ocen ze wszystkich etapów
    Wiersze: uczestnicy, Kolumny: sędzia_etap
    """
    profile_df, participant_info = analyzer.build_participant_multistage_matrix()
    
    # Ogranicz do uczestników którzy przeszli przez >= 2 etapy
    valid_participants = [nr for nr, info in participant_info.items() 
                         if len(info['stages']) >= 2]
    profile_df = profile_df.loc[valid_participants]
    
    # Sortuj kolumny według etapu i sędziego
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    
    def sort_key(col):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            judge, stage = parts
            stage_idx = stage_order.index(stage) if stage in stage_order else 999
            return (stage_idx, judge)
        return (999, col)
    
    sorted_columns = sorted(profile_df.columns, key=sort_key)
    profile_df = profile_df[sorted_columns]
    
    # Etykiety uczestników
    row_labels = [f"{nr}: {participant_info[nr]['nazwisko']}" 
                 for nr in profile_df.index]
    
    # Sortuj uczestników po średniej ocenie
    row_means = profile_df.mean(axis=1)
    sorted_indices = row_means.sort_values(ascending=False).index
    profile_df = profile_df.loc[sorted_indices]
    row_labels = [f"{nr}: {participant_info[nr]['nazwisko']}" 
                 for nr in sorted_indices]
    
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(20, max(12, len(profile_df) * 0.3)))
    
    sns.heatmap(profile_df, cmap='RdYlGn', center=15, 
               cbar_kws={'label': 'Ocena'},
               yticklabels=row_labels,
               xticklabels=[col.replace('_', '\n') for col in profile_df.columns],
               ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_title('Heatmapa wszystkich ocen ze wszystkich etapów\n' +
                '(uczestnicy posortowani według średniej oceny)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sędzia_Etap', fontsize=12)
    ax.set_ylabel('Uczestnik', fontsize=12)
    
    # Dodaj linie separujące etapy
    stage_boundaries = []
    current_stage = None
    for i, col in enumerate(profile_df.columns):
        stage = col.rsplit('_', 1)[-1]
        if stage != current_stage:
            stage_boundaries.append(i)
            current_stage = stage
    
    for boundary in stage_boundaries[1:]:
        ax.axvline(x=boundary, color='blue', linewidth=3, alpha=0.5)
    
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano heatmapę wieloetapową: {save_path}")


def visualize_judge_consistency(analyzer, save_path='judge_consistency.png'):
    """
    Analiza konsystencji sędziów między etapami
    Pokazuje jak stabilne są oceny sędziów dla tych samych uczestników
    """
    profile_df = analyzer.build_judge_multistage_matrix()
    
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    
    # Dla każdego sędziego, oblicz korelacje między etapami
    judge_consistency = []
    
    for judge in analyzer.judge_columns:
        judge_row = profile_df.loc[judge]
        
        # Zbierz oceny z każdego etapu
        stage_scores = {}
        for stage in stage_order:
            stage_cols = [col for col in judge_row.index if col.endswith(f'_{stage}')]
            if len(stage_cols) > 0:
                stage_scores[stage] = judge_row[stage_cols].dropna()
        
        # Oblicz średnią korelację między wszystkimi parami etapów
        correlations = []
        for i, stage1 in enumerate(stage_order):
            for stage2 in stage_order[i+1:]:
                if stage1 in stage_scores and stage2 in stage_scores:
                    # Znajdź wspólnych uczestników
                    participants1 = set([col.split('_')[0] for col in stage_scores[stage1].index])
                    participants2 = set([col.split('_')[0] for col in stage_scores[stage2].index])
                    common = participants1 & participants2
                    
                    if len(common) >= 3:
                        scores1 = []
                        scores2 = []
                        for p in common:
                            col1 = f"{p}_{stage1}"
                            col2 = f"{p}_{stage2}"
                            if col1 in stage_scores[stage1].index and col2 in stage_scores[stage2].index:
                                scores1.append(stage_scores[stage1][col1])
                                scores2.append(stage_scores[stage2][col2])
                        
                        if len(scores1) >= 3:
                            corr = np.corrcoef(scores1, scores2)[0, 1]
                            correlations.append(corr)
        
        if correlations:
            judge_consistency.append({
                'judge': judge,
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'n_comparisons': len(correlations),
                'mean_score': judge_row.mean(),
                'std_score': judge_row.std()
            })
    
    if not judge_consistency:
        print("Brak danych do analizy konsystencji")
        return
    
    consistency_df = pd.DataFrame(judge_consistency)
    consistency_df = consistency_df.sort_values('mean_correlation', ascending=False)
    
    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Ranking konsystencji
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(consistency_df)))
    bars = ax1.barh(range(len(consistency_df)), consistency_df['mean_correlation'], color=colors)
    ax1.set_yticks(range(len(consistency_df)))
    ax1.set_yticklabels(consistency_df['judge'])
    ax1.set_xlabel('Średnia korelacja między etapami', fontsize=12)
    ax1.set_title('Konsystencja sędziów\n(wyższa korelacja = bardziej konsystentne oceny)',
                 fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Próg 0.5')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Konsystencja vs średnia ocena
    ax2 = axes[0, 1]
    scatter = ax2.scatter(consistency_df['mean_score'], 
                         consistency_df['mean_correlation'],
                         s=200, alpha=0.6, edgecolors='black')
    for idx, row in consistency_df.iterrows():
        ax2.annotate(row['judge'], 
                    (row['mean_score'], row['mean_correlation']),
                    fontsize=8, alpha=0.7)
    ax2.set_xlabel('Średnia ocena (przez wszystkie etapy)', fontsize=12)
    ax2.set_ylabel('Konsystencja (korelacja)', fontsize=12)
    ax2.set_title('Średnia ocena vs konsystencja', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Konsystencja vs zmienność
    ax3 = axes[1, 0]
    scatter2 = ax3.scatter(consistency_df['std_score'], 
                          consistency_df['mean_correlation'],
                          s=200, alpha=0.6, edgecolors='black', color='orange')
    for idx, row in consistency_df.iterrows():
        ax3.annotate(row['judge'], 
                    (row['std_score'], row['mean_correlation']),
                    fontsize=8, alpha=0.7)
    ax3.set_xlabel('Zmienność ocen (SD)', fontsize=12)
    ax3.set_ylabel('Konsystencja (korelacja)', fontsize=12)
    ax3.set_title('Zmienność vs konsystencja', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Rozkład konsystencji
    ax4 = axes[1, 1]
    ax4.hist(consistency_df['mean_correlation'], bins=10, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax4.axvline(x=consistency_df['mean_correlation'].median(), 
               color='red', linestyle='--', linewidth=2, label='Mediana')
    ax4.axvline(x=consistency_df['mean_correlation'].mean(), 
               color='green', linestyle='--', linewidth=2, label='Średnia')
    ax4.set_xlabel('Konsystencja (korelacja)', fontsize=12)
    ax4.set_ylabel('Liczba sędziów', fontsize=12)
    ax4.set_title('Rozkład konsystencji', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Analiza konsystencji sędziów między etapami\n' +
                '(jak stabilne są oceny dla tych samych uczestników)',
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano analizę konsystencji: {save_path}")
    
    return consistency_df


def visualize_participant_trajectories(analyzer, top_n=20, save_path='participant_trajectories.png'):
    """
    Wizualizacja "trajektorii" uczestników w przestrzeni PCA przez etapy
    Pokazuje jak uczestnicy "poruszają się" w przestrzeni ocen
    """
    # Buduj profile dla każdego etapu osobno
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    stage_labels = {'stage1': 'I', 'stage2': 'II', 'stage3': 'III', 'final': 'F'}
    
    # Zbierz uczestników którzy przeszli przez >= 3 etapy
    progression_df = analyzer.analyze_participant_progression()
    participant_counts = progression_df.groupby('Nr').size()
    multi_stage = participant_counts[participant_counts >= 3].index
    
    if len(multi_stage) < 3:
        print("Za mało uczestników z >= 3 etapami")
        return
    
    # Ogranicz do top N uczestników (według średniej z finału)
    final_scores = progression_df[progression_df['stage'] == 'final']
    if len(final_scores) > 0:
        top_participants = final_scores.nlargest(top_n, 'mean_score')['Nr'].values
        selected = [nr for nr in top_participants if nr in multi_stage]
    else:
        selected = multi_stage[:top_n].tolist()
    
    if len(selected) < 3:
        selected = multi_stage[:min(top_n, len(multi_stage))].tolist()
    
    # Dla każdego etapu, zbuduj macierz ocen i wykonaj PCA
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected)))
    
    for i, participant_nr in enumerate(selected):
        trajectory_points = []
        trajectory_stages = []
        
        for stage in stage_order:
            if stage in analyzer.stages_data:
                df = analyzer.stages_data[stage]
                participant_row = df[df['Nr'] == participant_nr]
                
                if len(participant_row) > 0:
                    # Zbierz oceny
                    scores = []
                    for judge in analyzer.judge_columns:
                        score_val = participant_row.iloc[0][judge]
                        if pd.notna(score_val) and str(score_val).strip().lower() != 's':
                            try:
                                scores.append(float(score_val))
                            except:
                                scores.append(np.nan)
                        else:
                            scores.append(np.nan)
                    
                    # Użyj pierwszych 2 ocen jako "współrzędne" (uproszczona wizualizacja)
                    # W prawdziwej implementacji użylibyśmy PCA na pełnym profilu
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if len(valid_scores) >= 2:
                        # Użyj średniej i SD jako "współrzędnych"
                        x = np.mean(valid_scores)
                        y = np.std(valid_scores)
                        trajectory_points.append((x, y))
                        trajectory_stages.append(stage_labels[stage])
        
        if len(trajectory_points) >= 2:
            # Rysuj trajektorię
            xs = [p[0] for p in trajectory_points]
            ys = [p[1] for p in trajectory_points]
            
            participant_info = progression_df[progression_df['Nr'] == participant_nr].iloc[0]
            label = f"{participant_nr}: {participant_info['nazwisko']}"
            
            ax.plot(xs, ys, marker='o', linewidth=2, markersize=8, 
                   color=colors[i], label=label, alpha=0.7)
            
            # Dodaj etykiety etapów
            for j, (x, y, stage_label) in enumerate(zip(xs, ys, trajectory_stages)):
                ax.annotate(stage_label, (x, y), fontsize=8, 
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Średnia ocena', fontsize=12)
    ax.set_ylabel('Odchylenie standardowe ocen', fontsize=12)
    ax.set_title('Trajektorie uczestników przez etapy\n' +
                '(średnia vs zmienność ocen w każdym etapie)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano trajektorie uczestników: {save_path}")


def visualize_cluster_evolution(analyzer, save_path='cluster_evolution.png'):
    """
    Pokazuje jak przypisania klastrów zmieniają się między etapami
    (dla uczestników którzy przeszli przez wiele etapów)
    """
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    
    # Dla każdego etapu, wykonaj klasterowanie
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    stage_clusters = {}
    
    for stage in stage_order:
        if stage not in analyzer.stages_data:
            continue
            
        df = analyzer.stages_data[stage]
        
        # Buduj macierz ocen
        participant_scores = {}
        for idx, row in df.iterrows():
            nr = int(row['Nr'])
            scores = []
            for judge in analyzer.judge_columns:
                score_val = row[judge]
                if pd.notna(score_val) and str(score_val).strip().lower() != 's':
                    try:
                        scores.append(float(score_val))
                    except:
                        pass
            
            if len(scores) > 0:
                participant_scores[nr] = scores
        
        # Standaryzacja i klasterowanie
        if len(participant_scores) >= 5:
            score_matrix = []
            participant_nrs = []
            
            for nr, scores in participant_scores.items():
                # Uzupełnij do stałej długości (17 sędziów)
                padded = scores + [np.nan] * (17 - len(scores))
                score_matrix.append(padded[:17])
                participant_nrs.append(nr)
            
            score_df = pd.DataFrame(score_matrix, index=participant_nrs)
            score_df_filled = score_df.fillna(score_df.mean())
            
            scaler = StandardScaler()
            scores_scaled = scaler.fit_transform(score_df_filled)
            
            kmeans = KMeans(n_clusters=min(5, len(scores_scaled)), random_state=42, n_init=20)
            clusters = kmeans.fit_predict(scores_scaled)
            
            stage_clusters[stage] = dict(zip(participant_nrs, clusters))
    
    # Znajdź uczestników którzy są w >= 3 etapach
    participant_stages = {}
    for stage, clusters_dict in stage_clusters.items():
        for nr in clusters_dict.keys():
            if nr not in participant_stages:
                participant_stages[nr] = []
            participant_stages[nr].append(stage)
    
    multi_stage_participants = {nr: stages for nr, stages in participant_stages.items() 
                               if len(stages) >= 3}
    
    if len(multi_stage_participants) == 0:
        print("Brak uczestników z >= 3 etapami do analizy ewolucji klastrów")
        return
    
    # Przygotuj dane do wizualizacji
    evolution_data = []
    for nr, stages in multi_stage_participants.items():
        row = {'Nr': nr}
        for stage in stage_order:
            if stage in stage_clusters and nr in stage_clusters[stage]:
                row[stage] = stage_clusters[stage][nr]
            else:
                row[stage] = np.nan
        evolution_data.append(row)
    
    evolution_df = pd.DataFrame(evolution_data)
    evolution_df = evolution_df.dropna(thresh=3)  # Usuń jeśli < 3 etapy
    
    if len(evolution_df) == 0:
        print("Brak danych do wizualizacji ewolucji")
        return
    
    # Sortuj według klastra w finale
    if 'final' in evolution_df.columns:
        evolution_df = evolution_df.sort_values('final')
    
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(14, max(10, len(evolution_df) * 0.4)))
    
    # Twórz heatmapę klastrów
    stage_cols = [s for s in stage_order if s in evolution_df.columns]
    heatmap_data = evolution_df[stage_cols].values
    
    im = ax.imshow(heatmap_data, cmap='Set1', aspect='auto', vmin=0, vmax=4)
    
    # Etykiety
    ax.set_xticks(np.arange(len(stage_cols)))
    ax.set_yticks(np.arange(len(evolution_df)))
    ax.set_xticklabels(stage_cols)
    ax.set_yticklabels([f"Uczestnik {nr}" for nr in evolution_df['Nr']])
    
    # Dodaj wartości w komórkach
    for i in range(len(evolution_df)):
        for j in range(len(stage_cols)):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, int(value),
                             ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Ewolucja przypisań do klastrów przez etapy\n' +
                '(dla uczestników którzy przeszli przez 3+ etapów)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Etap', fontsize=12)
    ax.set_ylabel('Uczestnik', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Numer klastra')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano ewolucję klastrów: {save_path}")


def run_advanced_visualizations(analyzer, output_dir='multistage_advanced'):
    """
    Uruchamia wszystkie zaawansowane wizualizacje
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ZAAWANSOWANE WIZUALIZACJE WIELOETAPOWE")
    print("=" * 80)
    
    # 1. PCA uczestników
    print("\n1. PCA uczestników...")
    try:
        pca, pca_components, variance_df = visualize_pca_participants(
            analyzer, n_components=3, save_path=f'{output_dir}/10_participant_pca.png')
        variance_df.to_csv(f'{output_dir}/pca_variance_explained.csv', 
                          index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    # 2. PCA sędziów
    print("\n2. PCA sędziów...")
    try:
        visualize_pca_judges(analyzer, save_path=f'{output_dir}/11_judge_pca.png')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    # 3. Heatmapa wieloetapowa
    print("\n3. Heatmapa wszystkich ocen...")
    try:
        visualize_multistage_heatmap(analyzer, 
                                    save_path=f'{output_dir}/12_multistage_heatmap.png')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    # 4. Konsystencja sędziów
    print("\n4. Analiza konsystencji sędziów...")
    try:
        consistency_df = visualize_judge_consistency(
            analyzer, save_path=f'{output_dir}/13_judge_consistency.png')
        if consistency_df is not None:
            consistency_df.to_csv(f'{output_dir}/judge_consistency.csv', 
                                 index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    # 5. Trajektorie uczestników
    print("\n5. Trajektorie uczestników...")
    try:
        visualize_participant_trajectories(analyzer, top_n=20,
                                           save_path=f'{output_dir}/14_participant_trajectories.png')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    # 6. Ewolucja klastrów
    print("\n6. Ewolucja klastrów...")
    try:
        visualize_cluster_evolution(analyzer, 
                                    save_path=f'{output_dir}/15_cluster_evolution.png')
    except Exception as e:
        print(f"  Błąd: {e}")
    
    print("\n" + "=" * 80)
    print(f"Zaawansowane wizualizacje zapisane w: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    print("Import tego modułu i użyj run_advanced_visualizations(analyzer)")
