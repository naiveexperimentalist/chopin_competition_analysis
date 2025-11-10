"""
Advanced visualizations of multi-stage Chopin Competition analysis

PCA visualizations, heatmaps, participant trajectories and deep cluster analysis
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
    PCA for participants - 2D/3D space projection
    Shows participant clusters in reduced space
    """
    from matplotlib.colors import ListedColormap

    profile_df, participant_info = analyzer.build_participant_multistage_matrix()

    # Filter participants
    valid_participants = [nr for nr, info in participant_info.items()
                          if len(info['stages']) >= min_stages]
    profile_df = profile_df.loc[valid_participants]
    profile_df = profile_df.dropna(axis=1, how='all')

    # Remove participants with too few scores
    min_valid_scores = len(profile_df.columns) * 0.3
    valid_rows = profile_df.count(axis=1) >= min_valid_scores
    profile_df = profile_df[valid_rows]

    if len(profile_df) < 5:
        print("Too few participants for PCA")
        return None, None, pd.DataFrame()

    # Fill missing values
    profile_df_filled = profile_df.fillna(profile_df.mean())
    if profile_df_filled.isnull().any().any():
        profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

    # Standardization
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df_filled)

    # Check infinite values
    if not np.isfinite(profile_scaled).all():
        print("WARNING: Replacing infinite values with 0")
        profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    n_components = min(n_components, len(profile_df_filled), len(profile_df_filled.columns))
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(profile_scaled)

    # K-means for coloring
    from sklearn.cluster import KMeans
    n_clusters = min(5, len(profile_df_filled) // 2)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
    else:
        clusters = np.zeros(len(profile_df_filled), dtype=int)

    # Prepare discrete color palette
    colors = sns.color_palette("Set1", n_clusters)
    cmap = ListedColormap(colors)

    # Visualization with cluster table below charts
    if n_components >= 2:
        import matplotlib.gridspec as gridspec
        # Prepare GridSpec: 2 rows, 3 columns (top row for charts, bottom for table)
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # two rows: top for charts, bottom for table
        # divide first row into 3 columns
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
            cbar1 = plt.colorbar(scatter, ax=ax1, label='Cluster')
            cbar1.set_ticks(range(n_clusters))
        # Finalist labels
        finalists = [nr for nr in valid_participants if 'final' in participant_info[nr]['stages']]
        for i, nr in enumerate(profile_df_filled.index):
            if nr in finalists:
                ax1.annotate(f"{nr}", (pca_components[i, 0]+0.16, pca_components[i, 1]+0.16), fontsize=8, alpha=0.7)

        # Second and third chart in top row
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
                cbar2 = plt.colorbar(scatter2, ax=ax2, label='Cluster')
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
                cbar3 = plt.colorbar(scatter3, ax=ax3, label='Cluster')
                cbar3.set_ticks(range(n_clusters))
        else:
            # If we only have 2 components, expand chart to remaining columns
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

        # Build cluster table: for each cluster list of participants "firstname lastname (nr)"
        cluster_members = {cid: [] for cid in range(n_clusters)}
        for i, nr in enumerate(profile_df_filled.index):
            cid = int(clusters[i])
            info = participant_info.get(nr, {})
            first_name = info.get('firstname') or info.get('firstname') or ''
            last_name = info.get('lastname') or ''
            cluster_members[cid].append(f"{first_name} {last_name} ({nr})".strip())
        # Prepare table rows
        table_rows = []
        for cid in sorted(cluster_members.keys()):
            table_rows.append([f"Cluster {cid}", ', '.join(cluster_members[cid])])
        # Add table in bottom row spanning all columns
        ax_table = fig.add_subplot(gs_bottom[0, 0])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=table_rows,
            colLabels=["Cluster", "Contestants"],
            cellLoc="left",
            loc="center",
            colWidths=[0.13, 0.87]
        )
        # Reduce cell margins (affects all columns)
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
        # rect param ensures that title does not overlap charts
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    variance_df = pd.DataFrame({
        'component': [f'PC{i + 1}' for i in range(n_components)],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })

    return pca, pca_components, variance_df


def visualize_pca_judges(analyzer, save_path='judge_pca.png'):
    """
    PCA for judges - shows similarities in scoring styles
    """
    from matplotlib.colors import ListedColormap

    profile_df = analyzer.build_judge_multistage_matrix()
    profile_df = profile_df.dropna(axis=1, how='all')

    profile_df_filled = profile_df.fillna(profile_df.mean(axis=1), axis=0)
    if profile_df_filled.isnull().any().any():
        profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df_filled)

    if not np.isfinite(profile_scaled).all():
        print("WARNING: Replacing infinite values with 0")
        profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    n_components = min(3, len(profile_df_filled))
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(profile_scaled)

    # K-means for coloring
    from sklearn.cluster import KMeans
    n_clusters = min(3, len(profile_df_filled) // 2)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
    else:
        clusters = np.zeros(len(profile_df_filled), dtype=int)

    colors = sns.color_palette("Set1", n_clusters)
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                         c=clusters, cmap=cmap, s=300, alpha=0.7, edgecolors='black',
                         vmin=0, vmax=n_clusters - 1)

    for i, judge in enumerate(profile_df_filled.index):
        x_offset = 0
        if str(judge) == 'K. Popowa-Zydron':
            x_offset = 1.3
        elif str(judge) == 'Krzysztof Jablonski':
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
    print(f"Judges PCA saved: {save_path}")

def calculate_trimmed_mean_from_processor(row: pd.Series, margin: float) -> float:
    scores = row.dropna()

    initial_mean = round(scores.mean(), 2)
    corrected_scores = scores.copy()
    for judge in scores.index:
        score = scores[judge]

        if abs(score - initial_mean) > margin:
            if score > initial_mean:
                corrected_scores[judge] = initial_mean + margin
            else:
                corrected_scores[judge] = initial_mean - margin

    final_avg = round(corrected_scores.mean(), 2)

    return final_avg

def visualize_multistage_heatmap(analyzer, save_path='multistage_heatmap.png'):
    profile_df, participant_info = analyzer.build_participant_multistage_matrix()
    # Limit to participants who went through >= 2 stages
    valid_participants = [nr for nr, info in participant_info.items()
                         if len(info['stages']) >= 2]
    profile_df = profile_df.loc[valid_participants]

    self_weights = {
        'stage1_cumulative': {'stage1': 1.0},
        'stage2_cumulative': {'stage1': 0.30, 'stage2': 0.70},
        'stage3_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.70},
        'final_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.35, 'final': 0.35}
    }

    # Column range definitions
    stage_cols = {
        'stage1': profile_df.columns[0:17],
        'stage2': profile_df.columns[17:34],
        'stage3': profile_df.columns[34:51],
        'final': profile_df.columns[51:68]
    }

    stage_margin = {
        'stage1': 3,
        'stage2': 2,
        'stage3': 2,
        'final': 2
    }

    stage_means = {}
    for stage_name, cols in stage_cols.items():
        stage_df = profile_df[cols]
        margin = stage_margin[stage_name]

        stage_means[stage_name] = stage_df.apply(
            calculate_trimmed_mean_from_processor,
            axis=1,
            margin=margin
        ).round(2)

    # 2. Calculate cumulative, weighted scores
    for cumulative_key, weights in self_weights.items():
        weighted_score = pd.Series(0.0, index=profile_df.index)
        for stage_name, weight in weights.items():
            if stage_name in stage_means:
                weighted_score += stage_means[stage_name] * weight

        # Cumulative results are added to profile_df
        profile_df[cumulative_key] = round(weighted_score, 2)

    final_ranking_logic = {
        'final_cumulative': 125,
        'stage3_cumulative': 25,
        'stage2_cumulative': 5,
        'stage1_cumulative': 1,
    }

    profile_df['total_weighted_score'] = pd.Series(0.0, index=profile_df.index)
    cumulative_keys = list(final_ranking_logic.keys())

    for index in profile_df.index:
        for key, multiplier in final_ranking_logic.items():
            cumulative_score = profile_df.loc[index, key]

            # Check if result IS available (not NaN)
            if pd.notna(cumulative_score):
                # Assigning result from highest achieved stage, multiplied by stage weight
                final_score = cumulative_score * multiplier
                profile_df.loc[index, 'total_weighted_score'] = round(final_score, 2)

                break

    sorted_profile_df = profile_df.sort_values(
        by='total_weighted_score',
        ascending=False
    )

    columns_to_drop = cumulative_keys + ['total_weighted_score']

    final_profile_df = sorted_profile_df.drop(
        columns=columns_to_drop,
        axis=1
    )

    row_labels = [
        f"{nr}: {participant_info[nr]['firstname']} {participant_info[nr]['lastname']}"
        for nr in final_profile_df.index
    ]
    fig, ax = plt.subplots(figsize=(20, max(12, len(final_profile_df) * 0.3)))

    sns.heatmap(final_profile_df, cmap='RdYlGn', center=15,
               cbar_kws={'label': 'score'},
               yticklabels=row_labels,
               xticklabels=[col.split('_')[0] for col in final_profile_df.columns],
               ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('Heatmap of all scores across all stages\n' +
                '(contestants sorted by final placement)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    # Add lines separating stages
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
    print(f"Saved multi-stage heatmap: {save_path}")

def visualize_multistage_heatmap_with_changed_margins(analyzer, save_path='multistage_heatmap_with_changed_margins.png'):
    profile_df, participant_info = analyzer.build_participant_multistage_matrix()
    # Limit to participants who went through >= 2 stages
    valid_participants = [nr for nr, info in participant_info.items()
                         if len(info['stages']) >= 2]
    profile_df = profile_df.loc[valid_participants]

    self_weights = {
        'stage1_cumulative': {'stage1': 1.0},
        'stage2_cumulative': {'stage1': 0.30, 'stage2': 0.70},
        'stage3_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.70},
        'final_cumulative': {'stage1': 0.10, 'stage2': 0.20, 'stage3': 0.35, 'final': 0.35}
    }

    # Column range definitions
    stage_cols = {
        'stage1': profile_df.columns[0:17],
        'stage2': profile_df.columns[17:34],
        'stage3': profile_df.columns[34:51],
        'final': profile_df.columns[51:68]
    }

    stage_margin = {
        'stage1': 5,
        'stage2': 4,
        'stage3': 3,
        'final': 2
    }

    stage_means = {}
    for stage_name, cols in stage_cols.items():
        stage_df = profile_df[cols]
        margin = stage_margin[stage_name]

        stage_means[stage_name] = stage_df.apply(
            calculate_trimmed_mean_from_processor,
            axis=1,
            margin=margin
        ).round(2)

    # 2. Calculate cumulative, weighted scores
    for cumulative_key, weights in self_weights.items():
        weighted_score = pd.Series(0.0, index=profile_df.index)
        for stage_name, weight in weights.items():
            if stage_name in stage_means:
                weighted_score += stage_means[stage_name] * weight

        # Cumulative results are added to profile_df
        profile_df[cumulative_key] = round(weighted_score, 2)

    final_ranking_logic = {
        'final_cumulative': 125,
        'stage3_cumulative': 25,
        'stage2_cumulative': 5,
        'stage1_cumulative': 1,
    }

    profile_df['total_weighted_score'] = pd.Series(0.0, index=profile_df.index)
    profile_df['score_after_correction'] = pd.Series(0.0, index=profile_df.index)
    cumulative_keys = list(final_ranking_logic.keys())

    for index in profile_df.index:
        for key, multiplier in final_ranking_logic.items():
            cumulative_score = profile_df.loc[index, key]

            # Check if result IS available (not NaN)
            if pd.notna(cumulative_score):
                # Assigning result from highest achieved stage, multiplied by stage weight
                final_score = cumulative_score * multiplier
                profile_df.loc[index, 'total_weighted_score'] = round(final_score, 2)
                profile_df.loc[index, 'score_after_correction'] = round(cumulative_score, 2)
                break

    sorted_profile_df = profile_df.sort_values(
        by='total_weighted_score',
        ascending=False
    )

    columns_to_drop = cumulative_keys + ['total_weighted_score', 'score_after_correction']

    final_profile_df = sorted_profile_df.drop(
        columns=columns_to_drop,
        axis=1
    )

    print(profile_df.head())
    row_labels = [
        f"{nr}: {participant_info[nr]['firstname']} {participant_info[nr]['lastname']} {profile_df.loc[nr, 'score_after_correction']}"
        for nr in final_profile_df.index
    ]
    fig, ax = plt.subplots(figsize=(20, max(12, len(final_profile_df) * 0.3)))

    sns.heatmap(final_profile_df, cmap='RdYlGn', center=15,
               cbar_kws={'label': 'score'},
               yticklabels=row_labels,
               xticklabels=[col.split('_')[0] for col in final_profile_df.columns],
               ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('Heatmap of scores\n' +
                '(contestants sorted by final placement)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    # Add lines separating stages
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
    print(f"Saved multi-stage heatmap with changed margins: {save_path}")

def visualize_judge_consistency(analyzer, save_path='judge_consistency.png'):
    """
    Judge consistency analysis between stages
    Shows how stable judge scores are for the same participants
    """
    profile_df = analyzer.build_judge_multistage_matrix()

    stage_order = ['stage1', 'stage2', 'stage3', 'final']

    judge_consistency = []

    for judge in analyzer.judge_columns:
        judge_row = profile_df.loc[judge]

        # Collect scores from each stage
        stage_scores = {}
        for stage in stage_order:
            stage_cols = [col for col in judge_row.index if col.endswith(f'_{stage}')]
            if len(stage_cols) > 0:
                stage_scores[stage] = judge_row[stage_cols].dropna()

        # Calculate mean correlation between all stage pairs
        correlations = []
        for i, stage1 in enumerate(stage_order):
            for stage2 in stage_order[i+1:]:
                if stage1 in stage_scores and stage2 in stage_scores:
                    # Find common participants
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

    consistency_df = pd.DataFrame(judge_consistency)
    consistency_df = consistency_df.sort_values('mean_correlation', ascending=False)

    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Ranking consistency
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(consistency_df)))
    bars = ax1.barh(range(len(consistency_df)), consistency_df['mean_correlation'], color=colors)
    ax1.set_yticks(range(len(consistency_df)))
    ax1.set_yticklabels(consistency_df['judge'])
    ax1.set_xlabel('Mean inter–stage correlation', fontsize=12)
    ax1.set_title('Judges’ consistency\n(higher correlation = more consistent scoring)',
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. Konsystencja vs mean score
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

    # 3. Consistency vs variability
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

    # 4. Consistency distribution
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
    print(f"Saved consistency analysis: {save_path}")

    return consistency_df


def visualize_participant_trajectories(analyzer, top_n=20, save_path='participant_trajectories.png'):
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    stage_labels = {'stage1': 'I', 'stage2': 'II', 'stage3': 'III', 'final': 'F'}

    progression_df = analyzer.analyze_participant_progression()
    participant_counts = progression_df.groupby('number').size()
    multi_stage = participant_counts[participant_counts >= 3].index

    if len(multi_stage) < 3:
        print("Too few participants with >= 3 stages")
        return

    # Limit to top N participants (by final mean)
    final_scores = progression_df[progression_df['stage'] == 'final']
    if len(final_scores) > 0:
        top_participants = final_scores.nlargest(top_n, 'mean_score')['number'].values
        selected = [nr for nr in top_participants if nr in multi_stage]
    else:
        selected = multi_stage[:top_n].tolist()

    if len(selected) < 3:
        selected = multi_stage[:min(top_n, len(multi_stage))].tolist()

    # For each stage, build score matrix and perform PCA
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(selected)))

    for i, participant_nr in enumerate(selected):
        trajectory_points = []
        trajectory_stages = []

        for stage in stage_order:
            if stage in analyzer.stages_data:
                df = analyzer.stages_data[stage]
                participant_row = df[df['number'] == participant_nr]

                if len(participant_row) > 0:
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

                    # Use first 2 scores as "coordinates" (simplified visualization)
                    # In real implementation we would use PCA on full profile
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if len(valid_scores) >= 2:
                        # Use mean and SD as "coordinates"
                        x = np.mean(valid_scores)
                        y = np.std(valid_scores)
                        trajectory_points.append((x, y))
                        trajectory_stages.append(stage_labels[stage])

        if len(trajectory_points) >= 2:
            # Draw trajectory
            xs = [p[0] for p in trajectory_points]
            ys = [p[1] for p in trajectory_points]

            participant_info = progression_df[progression_df['number'] == participant_nr].iloc[0]
            label = f"{participant_info['firstname']} {participant_info['lastname']}"

            ax.plot(xs, ys, marker='o', linewidth=2, markersize=8,
                   color=colors[i], label=label, alpha=0.7)

            # Add stage labels
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
    print(f"Saved in: {save_path}")


def visualize_cluster_evolution(analyzer, save_path='cluster_evolution.png'):
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap

    stage_order = ['stage1', 'stage2', 'stage3', 'final']

    participant_info = {}
    for stage in stage_order:
        if stage in analyzer.stages_data:
            df = analyzer.stages_data[stage]
            for idx, row in df.iterrows():
                nr = int(row['number'])
                if nr not in participant_info:
                    participant_info[nr] = {
                        'firstname': row['firstname'],
                        'lastname': row['lastname']
                    }
    # For each stage, perform clustering
    stage_clusters = {}

    for stage in stage_order:
        if stage not in analyzer.stages_data:
            continue

        df = analyzer.stages_data[stage]

        participant_scores = {}
        for idx, row in df.iterrows():
            nr = int(row['number'])
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

        # Standardization i clustering
        if len(participant_scores) >= 5:
            score_matrix = []
            participant_nrs = []

            for nr, scores in participant_scores.items():
                # Pad to constant length (17 judges)
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

    # Find participants who are in >= 3 stages
    participant_stages = {}
    for stage, clusters_dict in stage_clusters.items():
        for nr in clusters_dict.keys():
            if nr not in participant_stages:
                participant_stages[nr] = []
            participant_stages[nr].append(stage)

    multi_stage_participants = {nr: stages for nr, stages in participant_stages.items()
                                if len(stages) >= 3}

    evolution_data = []
    for nr, stages in multi_stage_participants.items():
        row = {'number': nr}
        for stage in stage_order:
            if stage in stage_clusters and nr in stage_clusters[stage]:
                row[stage] = stage_clusters[stage][nr]
            else:
                row[stage] = np.nan
        evolution_data.append(row)

    evolution_df = pd.DataFrame(evolution_data)
    evolution_df = evolution_df.dropna(thresh=3)  # Remove if < 3 stages

    # Sort by cluster in final
    if 'final' in evolution_df.columns:
        evolution_df = evolution_df.sort_values('final')

    stage_cols = [s for s in stage_order if s in evolution_df.columns]
    heatmap_data = evolution_df[stage_cols].copy()
    heatmap_data.index = [
        f"{participant_info[nr]['firstname']} {participant_info[nr]['lastname']}"
        if nr in participant_info
        else f"Participant {nr}"
        for nr in evolution_df['number']
    ]

    colors = sns.color_palette("Set1", 5)
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(14, max(10, len(evolution_df) * 0.4)))

    # Use seaborn heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,
        fmt='.0f',
        cbar_kws={'label': 'Cluster number', 'ticks': [0, 1, 2, 3, 4]},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=4,
        ax=ax
    )

    ax.set_title('Evolution of cluster assignments across stages\n' +
                 '(for contestants who advanced through 3+ stages)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    ax.set_xticklabels(stage_cols, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster evolution: {save_path}")


def run_advanced_visualizations(analyzer, output_dir='multistage_advanced'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("ADVANCED MULTISTAGE VISUALIZATIONS")
    print("=" * 80)

    # 1. PCA participants
    print("\n1. PCA participants...")
    try:
        pca, pca_components, variance_df = visualize_pca_participants(
            analyzer, n_components=3, save_path=f'{output_dir}/10_participant_pca.png')
        variance_df.to_csv(f'{output_dir}/pca_variance_explained.csv',
                          index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Error: {e}")

    # 2. PCA judges
    print("\n2. PCA judges...")
    try:
        visualize_pca_judges(analyzer, save_path=f'{output_dir}/11_judge_pca.png')
    except Exception as e:
        print(f"  Error: {e}")

    # 3a. Multi-stage heatmap
    print("\n3. Heatmap...")
    try:
        visualize_multistage_heatmap(analyzer,
                                     save_path=f'{output_dir}/12_multistage_heatmap.png')
    except Exception as e:
        raise e
        print(f"  Error: {e}")

    # 3b. Multi-stage heatmap (changed margis)
    print("\n3b. Heatmap (changed margins)...")
    try:
        visualize_multistage_heatmap_with_changed_margins(analyzer,
                                    save_path=f'{output_dir}/12b_multistage_heatmap_with_changed_margins.png')
    except Exception as e:
        raise e
        print(f"  Error: {e}")

    # 4. Judge consistency
    print("\n4. Judge consistency analysis...")
    try:
        consistency_df = visualize_judge_consistency(
            analyzer, save_path=f'{output_dir}/13_judge_consistency.png')
        if consistency_df is not None:
            consistency_df.to_csv(f'{output_dir}/judge_consistency.csv',
                                 index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Error: {e}")

    print("\n5. Participant trajectories...")
    try:
        visualize_participant_trajectories(analyzer, top_n=20,
                                           save_path=f'{output_dir}/14_participant_trajectories.png')
    except Exception as e:
        print(f"  Error: {e}")

    print("\n6. Cluster evolution...")
    try:
        visualize_cluster_evolution(analyzer,
                                    save_path=f'{output_dir}/15_cluster_evolution.png')
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print(f"Saved {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    print("Import this module and use run_advanced_visualizations(analyzer)")