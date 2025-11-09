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


def visualize_pca_participants(analyzer, n_components=3, min_stages=4,
                               save_path='participant_pca.png'):
    """
    PCA dla uczestników - projekcja w przestrzeni 2D/3D
    Pokazuje klastry uczestników w zredukowanej przestrzeni
    """
    from matplotlib.colors import ListedColormap

    profile_df, participant_info = analyzer.build_participant_multistage_matrix()

    # Filtruj uczestników
    valid_participants = [nr for nr, info in participant_info.items()
                          if len(info['stages']) >= min_stages]
    profile_df = profile_df.loc[valid_participants]
    profile_df = profile_df.dropna(axis=1, how='all')

    # Usuń uczestników ze zbyt małą liczbą ocen
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

    # Przygotuj dyskretną paletę kolorów
    colors = sns.color_palette("Set1", n_clusters)
    cmap = ListedColormap(colors)

    # Wizualizacja z tabelą klastrów pod wykresami
    if n_components >= 2:
        import matplotlib.gridspec as gridspec
        # Przygotuj GridSpec: 2 wiersze, 3 kolumny (górny wiersz na wykresy, dolny na tabelę)
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # dwa wiersze: górny na wykresy, dolny na tabelę
        # pierwszy wiersz dzielimy na 3 kolumny
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0])
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1])

        # 2D scatter (PC1 vs PC2)
        ax1 = fig.add_subplot(gs_top[0, 0])
        scatter = ax1.scatter(pca_components[:, 0], pca_components[:, 1],
                              c=clusters, cmap=cmap, s=100, alpha=0.7, edgecolors='black',
                              vmin=0, vmax=max(n_clusters - 1, 0))
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
        ax1.set_title('PCA: PC1 vs PC2', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if n_clusters > 0:
            cbar1 = plt.colorbar(scatter, ax=ax1, label='Klaster')
            cbar1.set_ticks(range(n_clusters))
        # Etykiety finalistów
        finalists = [nr for nr in valid_participants if 'final' in participant_info[nr]['stages']]
        for i, nr in enumerate(profile_df_filled.index):
            if nr in finalists:
                ax1.annotate(f"{nr}", (pca_components[i, 0]+0.16, pca_components[i, 1]+0.16), fontsize=8, alpha=0.7)

        # Drugi i trzeci wykres w górnym rzędzie
        if n_components >= 3:
            # PC1 vs PC3
            ax2 = fig.add_subplot(gs_top[0, 1])
            scatter2 = ax2.scatter(pca_components[:, 0], pca_components[:, 2],
                                   c=clusters, cmap=cmap, s=100, alpha=0.7, edgecolors='black',
                                   vmin=0, vmax=max(n_clusters - 1, 0))
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
            ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance explained)', fontsize=12)
            ax2.set_title('PCA: PC1 vs PC3', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            if n_clusters > 0:
                cbar2 = plt.colorbar(scatter2, ax=ax2, label='Klaster')
                cbar2.set_ticks(range(n_clusters))
            # PC2 vs PC3
            ax3 = fig.add_subplot(gs_top[0, 2])
            scatter3 = ax3.scatter(pca_components[:, 1], pca_components[:, 2],
                                   c=clusters, cmap=cmap, s=100, alpha=0.7, edgecolors='black',
                                   vmin=0, vmax=max(n_clusters - 1, 0))
            ax3.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
            ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance explained)', fontsize=12)
            ax3.set_title('PCA: PC2 vs PC3', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            if n_clusters > 0:
                cbar3 = plt.colorbar(scatter3, ax=ax3, label='Klaster')
                cbar3.set_ticks(range(n_clusters))
        else:
            # Jeśli mamy tylko 2 komponenty, rozciągnij wykres na pozostałe kolumny
            ax2 = fig.add_subplot(gs_top[0, 1:])
            scatter2 = ax2.scatter(pca_components[:, 0], pca_components[:, 1],
                                   c=clusters, cmap=cmap, s=100, alpha=0.7, edgecolors='black',
                                   vmin=0, vmax=max(n_clusters - 1, 0))
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
            ax2.set_title('PCA: PC1 vs PC2', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            if n_clusters > 0:
                cbar2 = plt.colorbar(scatter2, ax=ax2, label='Cluster')
                cbar2.set_ticks(range(n_clusters))

        # Buduj tablicę klastrów: dla każdego klastra lista uczestników "imie nazwisko (nr)"
        cluster_members = {cid: [] for cid in range(n_clusters)}
        for i, nr in enumerate(profile_df_filled.index):
            cid = int(clusters[i])
            info = participant_info.get(nr, {})
            first_name = info.get('imię') or info.get('imie') or ''
            last_name = info.get('nazwisko') or ''
            cluster_members[cid].append(f"{first_name} {last_name} ({nr})".strip())
        # Przygotuj wiersze do tabeli
        table_rows = []
        for cid in sorted(cluster_members.keys()):
            members = cluster_members[cid]
            if members:
                # Podziel listę na linie co trzy nazwiska, aby uniknąć bardzo długich wierszy
                # chunks = [', '.join(members[i:i+3]) for i in range(0, len(members), 3)]
                # members_text = '\n'.join(chunks)
                members_text = ', '.join(members)
            else:
                members_text = '-'
            table_rows.append([f"Cluster {cid}", members_text])
        # Dodaj tabelę w dolnym wierszu zajmującym wszystkie kolumny
        ax_table = fig.add_subplot(gs_bottom[0, 0])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=table_rows,
            colLabels=["Cluster", "Contestants"],
            cellLoc="left",
            loc="center",
            colWidths=[0.13, 0.87]
        )
        # Zredukuj marginesy w komórkach (dotyczy wszystkich kolumn)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.PAD = 0.02
            if col_idx == 0:
                cell.loc = "center"

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        # table.auto_set_column_width(col=[0, 3])

        plt.suptitle('PCA of contestants\n' +
                     'Projection of multi–stage profiles in principal–component space',
                     fontsize=16, fontweight='bold', y=0.97)
        # rect param sprawia, że tytuł nie nakłada się na wykresy
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Zapisano PCA uczestników: {save_path}")

    # Zwróć wyjaśnioną wariancję
    variance_df = pd.DataFrame({
        'component': [f'PC{i + 1}' for i in range(n_components)],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })

    return pca, pca_components, variance_df


def visualize_pca_judges(analyzer, save_path='judge_pca.png'):
    """
    PCA dla sędziów - pokazuje podobieństwa w stylach oceniania
    """
    from matplotlib.colors import ListedColormap

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

    # Przygotuj dyskretną paletę kolorów
    colors = sns.color_palette("Set1", n_clusters)
    cmap = ListedColormap(colors)

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                         c=clusters, cmap=cmap, s=300, alpha=0.7, edgecolors='black',
                         vmin=0, vmax=n_clusters - 1)

    # Etykiety sędziów
    for i, judge in enumerate(profile_df_filled.index):
        print(f'x-{judge}-x')
        x_offset = 0
        if str(judge) == 'K. Popowa-Zydroń':
            x_offset = 1.3
        elif str(judge) == 'Krzysztof Jabłoński':
            x_offset = 0.5
        ax.annotate(judge,
                    (pca_components[i, 0] + x_offset, pca_components[i, 1] + 0.42),
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
    ax.set_title('PCA of judges\n' +
                 'Similarities in scoring styles across all stages',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
    cbar.set_ticks(range(n_clusters))

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

    self_weights = {
        'stage1_cumulative': {'stage1': 1.0},
        'stage2_cumulative': {'stage1': 0.30, 'stage2': 0.70},
        'stage3_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.70},
        'final_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.35, 'final': 0.35}
    }

    # Definicje zakresów kolumn
    stage_cols = {
        'stage1': profile_df.columns[0:17],  # Indeksy 1 do 17 (17 kolumn)
        'stage2': profile_df.columns[17:34],  # Indeksy 18 do 34 (17 kolumn)
        'stage3': profile_df.columns[34:51],  # Indeksy 35 do 51 (17 kolumn)
        'final': profile_df.columns[51:68]  # Indeksy 52 do 68 (17 kolumn)
    }

    stage_margin = {
        'stage1': 3,
        'stage2': 2,
        'stage3': 2,
        'final': 2
    }

    def calculate_trimmed_mean_from_processor(row: pd.Series, margin: float) -> float:
        """
        Adoptowana logika z ChopinCompetitionProcessor.process_scores.
        Oblicza skorygowaną średnią dla jednego wiersza (uczestnika) ocen.
        """
        # Usuwamy NaN (może reprezentować brak oceny lub 's')
        scores = row.dropna()

        if len(scores) == 0:
            return np.nan

        # Krok 1: Pierwsza średnia (klucz do korekty)
        initial_mean = scores.mean().round(2)

        # Przygotowanie skopiowanej serii do korekty
        corrected_scores = scores.copy()
        # Krok 2: Korekta ocen
        for judge in scores.index:
            score = scores[judge]

            # Sprawdzanie odchylenia
            if abs(score - initial_mean) > margin:
                if score > initial_mean:
                    corrected_scores[judge] = initial_mean + margin
                else:
                    corrected_scores[judge] = initial_mean - margin

        # Krok 3: Średnia po korekcie
        final_avg = corrected_scores.mean().round(2)

        return final_avg

    stage_means = {}

    for stage_name, cols in stage_cols.items():
        stage_df = profile_df[cols]
        margin = stage_margin[stage_name]

        # Zastosuj funkcję do KAŻDEGO WIERSZA (axis=1)
        stage_means[stage_name] = stage_df.apply(
            calculate_trimmed_mean_from_processor,
            axis=1,
            margin=margin
        ).round(2)

    # 2. Oblicz skumulowane, ważone oceny
    for cumulative_key, weights in self_weights.items():
        weighted_score = pd.Series(0.0, index=profile_df.index)
        for stage_name, weight in weights.items():
            if stage_name in stage_means:
                # Uwzględnianie tylko dostępnych średnich etapowych
                # .fillna(0) w tym miejscu nie jest konieczne, bo Nan*waga da Nan,
                # a suma z Nan da Nan, co jest poprawne, bo nie chcemy sztucznie sumować etapów
                weighted_score += stage_means[stage_name] * weight

        # Wyniki skumulowane są dodawane do profile_df
        profile_df[cumulative_key] = round(weighted_score, 2)

    # 3. Obliczanie klucza sortowania (OSTATNI DOSTĘPNY WYNIK * MNOŻNIK)
    final_ranking_logic = {
        'final_cumulative': 4,
        'stage3_cumulative': 3,
        'stage2_cumulative': 2,
        'stage1_cumulative': 1,
    }

    # Inicjalizacja klucza sortowania zerami
    profile_df['total_weighted_score'] = pd.Series(0.0, index=profile_df.index)
    cumulative_keys = list(final_ranking_logic.keys())

    # Iteracja po wierszach (uczestnikach)
    for index in profile_df.index:
        # Iteracja od najwyższego etapu do najniższego (final_cumulative -> stage1_cumulative)
        for key, multiplier in final_ranking_logic.items():
            cumulative_score = profile_df.loc[index, key]

            # Sprawdzamy, czy wynik JEST dostępny (nie NaN)
            if pd.notna(cumulative_score):
                # Przypisanie wyniku z najwyższego osiągniętego etapu, pomnożonego przez wagę etapu
                final_score = cumulative_score * multiplier
                profile_df.loc[index, 'total_weighted_score'] = round(final_score, 2)

                # Ważne: Przypisano, więc przerywamy wewnętrzną pętlę i przechodzimy do kolejnego uczestnika
                break

    # 4. Sortowanie DataFrame
    sorted_profile_df = profile_df.sort_values(
        by='total_weighted_score',
        ascending=False
    )
    # print(sorted_profile_df)
    # exit(0)

    # 5. Oczyszczenie z kolumn pomocniczych
    columns_to_drop = cumulative_keys + ['total_weighted_score']

    final_profile_df = sorted_profile_df.drop(
        columns=columns_to_drop,
        axis=1
    )

    # 6. Generowanie etykiet wierszy dla posortowanego DataFrame
    # Używamy indeksów (numerów uczestników) z posortowanej ramki danych,
    # aby uzyskać poprawne etykiety
    row_labels = [
        f"{nr}: {participant_info[nr]['imię']} {participant_info[nr]['nazwisko']}"
        for nr in final_profile_df.index
    ]
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(20, max(12, len(final_profile_df) * 0.3)))

    sns.heatmap(final_profile_df, cmap='RdYlGn', center=15,
               cbar_kws={'label': 'Ocena'},
               yticklabels=row_labels,
               xticklabels=[col.split('_')[0] for col in final_profile_df.columns],
               ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('Heatmap of all scores across all stages\n' +
                '(contestants sorted by final placement)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    
    # Dodaj linie separujące etapy
    stage_boundaries = []
    current_stage = None

    for i, col in enumerate(final_profile_df.columns):
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
    ax1.set_xlabel('Mean inter–stage correlation', fontsize=12)
    ax1.set_title('Judges’ consistency\n(higher correlation = more consistent scoring)',
                 fontsize=14, fontweight='bold')
    # ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold 0.5')
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
    ax2.set_xlabel('Mean score (across all stages)', fontsize=12)
    ax2.set_ylabel('Consistency (correlation)', fontsize=12)
    ax2.set_title('Mean score vs consistency', fontsize=14, fontweight='bold')
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
    ax3.set_xlabel('Score variability (SD)', fontsize=12)
    ax3.set_ylabel('Consistency (correlation)', fontsize=12)
    ax3.set_title('Variability vs consistency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Rozkład konsystencji
    ax4 = axes[1, 1]
    ax4.hist(consistency_df['mean_correlation'], bins=10, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax4.axvline(x=consistency_df['mean_correlation'].median(), 
               color='red', linestyle='--', linewidth=2, label='Median')
    ax4.axvline(x=consistency_df['mean_correlation'].mean(), 
               color='green', linestyle='--', linewidth=2, label='Mean')
    ax4.set_xlabel('Consistency (correlation)', fontsize=12)
    ax4.set_ylabel('Number of judges', fontsize=12)
    ax4.set_title('Distribution of consistency', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Consistency of judges across stages',
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
            label = f"{participant_info['imię']} {participant_info['nazwisko']}"
            
            ax.plot(xs, ys, marker='o', linewidth=2, markersize=8, 
                   color=colors[i], label=label, alpha=0.7)
            
            # Dodaj etykiety etapów
            for j, (x, y, stage_label) in enumerate(zip(xs, ys, trajectory_stages)):
                ax.annotate(stage_label, (x, y), fontsize=8, 
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Mean score', fontsize=12)
    ax.set_ylabel('Standard deviation of scores', fontsize=12)
    ax.set_title('Contestants’ trajectories across stages\n' +
                '(mean vs variability of scores in each stage)',
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
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap

    stage_order = ['stage1', 'stage2', 'stage3', 'final']

    # Zbierz informacje o uczestnikach (imię, nazwisko)
    participant_info = {}
    for stage in stage_order:
        if stage in analyzer.stages_data:
            df = analyzer.stages_data[stage]
            for idx, row in df.iterrows():
                nr = int(row['Nr'])
                if nr not in participant_info:
                    participant_info[nr] = {
                        'imie': row['imię'],
                        'nazwisko': row['nazwisko']
                    }
    # Dla każdego etapu, wykonaj klasterowanie
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

    # Przygotuj dane do heatmapy
    stage_cols = [s for s in stage_order if s in evolution_df.columns]
    heatmap_data = evolution_df[stage_cols].copy()
    # heatmap_data.index = [f"Uczestnik {nr}" for nr in evolution_df['Nr']]
    # Użyj imion i nazwisk zamiast numerów
    heatmap_data.index = [
        f"{participant_info[nr]['imie']} {participant_info[nr]['nazwisko']}"
        if nr in participant_info
        else f"Uczestnik {nr}"
        for nr in evolution_df['Nr']
    ]

    colors = sns.color_palette("Set1", 5)
    cmap = ListedColormap(colors)

    # Wizualizacja z seaborn
    fig, ax = plt.subplots(figsize=(14, max(10, len(evolution_df) * 0.4)))

    # Użyj seaborn heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,  # Pokazuj wartości w komórkach
        fmt='.0f',  # Format liczb (bez miejsc po przecinku)
        cbar_kws={'label': 'Cluster number', 'ticks': [0, 1, 2, 3, 4]},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=4,
        ax=ax
    )

    # Etykiety i tytuł
    ax.set_title('Evolution of cluster assignments across stages\n' +
                 '(for contestants who advanced through 3+ stages)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    # Popraw etykiety osi X (nazwy etapów)
    ax.set_xticklabels(stage_cols, rotation=0)

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
