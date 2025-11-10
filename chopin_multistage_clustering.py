"""
Multistage clustering and PCA of contestants/judges.

Provides clustering across stages (I–Final), PCA projections, and
cluster-evolution summaries to understand structure in the score space.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MultiStageClusteringAnalyzer:

    def __init__(self, data_files):
        self.data_files = data_files
        self.stages_data = {}
        self.judge_columns = []
        self.load_data()

    def load_data(self):
        print("Loading data from all stages…")

        for stage, filepath in self.data_files.items():
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip()
                self.stages_data[stage] = df
                print(f"  {stage}: {len(df)} contestants")

                if not self.judge_columns:
                    self.judge_columns = [col for col in df.columns
                                         if col not in ['number', 'firstname', 'lastname']]
            except Exception as e:
                print(f"  CSV reading error for {stage}: {e}")

        print(f"{len(self.judge_columns)} judges found")

    def build_participant_multistage_matrix(self):
        participant_profiles = {}
        participant_info = {}

        stage_order = ['stage1', 'stage2', 'stage3', 'final']

        for stage in stage_order:
            if stage not in self.stages_data:
                continue

            df = self.stages_data[stage]

            for idx, row in df.iterrows():
                nr = int(row['number'])

                # Initialize contestant profile
                if nr not in participant_profiles:
                    participant_profiles[nr] = {}
                    participant_info[nr] = {
                        'firstname': row['firstname'],
                        'lastname': row['lastname'],
                        'stages': []
                    }

                participant_info[nr]['stages'].append(stage)

                for judge in self.judge_columns:
                    col_name = f"{judge}_{stage}"
                    score_val = row[judge]

                    if pd.isna(score_val) or str(score_val).strip().lower() == 's':
                        participant_profiles[nr][col_name] = np.nan
                    else:
                        try:
                            participant_profiles[nr][col_name] = float(score_val)
                        except:
                            participant_profiles[nr][col_name] = np.nan

        profile_df = pd.DataFrame.from_dict(participant_profiles, orient='index')
        profile_df = profile_df.sort_index()

        for no, info in sorted(participant_info.items())[:5]:
            print(f"  Contestant {no} ({info['firstname']} {info['lastname']}): {info['stages']}")

        return profile_df, participant_info

    def build_judge_multistage_matrix(self):
        judge_profiles = {judge: {} for judge in self.judge_columns}

        stage_order = ['stage1', 'stage2', 'stage3', 'final']

        for stage in stage_order:
            if stage not in self.stages_data:
                continue

            df = self.stages_data[stage]

            for idx, row in df.iterrows():
                nr = int(row['number'])
                participant_label = f"P{nr}_{stage}"

                for judge in self.judge_columns:
                    score_val = row[judge]

                    if pd.isna(score_val) or str(score_val).strip().lower() == 's':
                        judge_profiles[judge][participant_label] = np.nan
                    else:
                        try:
                            judge_profiles[judge][participant_label] = float(score_val)
                        except:
                            judge_profiles[judge][participant_label] = np.nan

        return pd.DataFrame.from_dict(judge_profiles, orient='index')

    def cluster_participants_hierarchical(self, min_stages=2):
        profile_df, participant_info = self.build_participant_multistage_matrix()

        valid_participants = [nr for nr, info in participant_info.items()
                             if len(info['stages']) >= min_stages]
        profile_df = profile_df.loc[valid_participants]

        profile_df = profile_df.dropna(axis=1, how='all')

        min_valid_scores = len(profile_df.columns) * 0.3
        valid_rows = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_rows]

        profile_df_filled = profile_df.fillna(profile_df.mean())

        if profile_df_filled.isnull().any().any():
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

        # Standardization
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)

        if not np.isfinite(profile_scaled).all():
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            linkage_matrix = linkage(profile_scaled, method='ward')
        except ValueError as e:
            print(f"Using 'average' instead of 'ward', because of the error: {e})")
            linkage_matrix = linkage(profile_scaled, method='average')

        labels = [f"{nr}: {participant_info[nr]['firstname']} {participant_info[nr]['lastname']}"
                 for nr in profile_df_filled.index]

        return linkage_matrix, labels, profile_df_filled, participant_info

    def cluster_judges_hierarchical(self):
        profile_df = self.build_judge_multistage_matrix()

        profile_df = profile_df.dropna(axis=1, how='all')

        min_valid_scores = len(profile_df.columns) * 0.3
        valid_judges = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_judges]

        profile_df_filled = profile_df.fillna(profile_df.mean(axis=1), axis=0)

        if profile_df_filled.isnull().any().any():
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)

        if not np.isfinite(profile_scaled).all():
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            linkage_matrix = linkage(profile_scaled, method='ward')
        except ValueError as e:
            print(f"Using 'average' instead of 'ward', because of the error: {e})")
            linkage_matrix = linkage(profile_scaled, method='average')

        return linkage_matrix, profile_df_filled.index.tolist(), profile_df_filled

    def kmeans_cluster_participants(self, n_clusters=5, min_stages=2):
        profile_df, participant_info = self.build_participant_multistage_matrix()

        valid_participants = [nr for nr, info in participant_info.items()
                             if len(info['stages']) >= min_stages]
        profile_df = profile_df.loc[valid_participants]
        profile_df = profile_df.dropna(axis=1, how='all')

        min_valid_scores = len(profile_df.columns) * 0.3
        valid_rows = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_rows]

        profile_df_filled = profile_df.fillna(profile_df.mean())

        if profile_df_filled.isnull().any().any():
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)

        if not np.isfinite(profile_scaled).all():
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(profile_scaled)

        results = []
        for i, nr in enumerate(profile_df_filled.index):
            results.append({
                'number': nr,
                'firstname': participant_info[nr]['firstname'],
                'lastname': participant_info[nr]['lastname'],
                'stages_count': len(participant_info[nr]['stages']),
                'stages': ', '.join(participant_info[nr]['stages']),
                'cluster': cluster_labels[i]
            })

        results_df = pd.DataFrame(results)

        stage_cols = {}
        for col in profile_df_filled.columns:
            stage = col.split('_')[-1]
            if stage not in stage_cols:
                stage_cols[stage] = []
            stage_cols[stage].append(col)

        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]

            if not cluster_indices:
                continue

            cluster_stats.append({
                'cluster': cluster_id,
                'count': len(cluster_indices),
                'avg_stages': np.mean([len(participant_info[nr]['stages'])
                                      for nr in profile_df_filled.index[cluster_indices]]),
                **{f'avg_score_{stage}': profile_df_filled.iloc[cluster_indices][cols].mean().mean()
                   for stage, cols in stage_cols.items() if cols}
            })

        cluster_stats_df = pd.DataFrame(cluster_stats)

        return results_df, cluster_stats_df, kmeans, scaler, profile_df_filled

    def kmeans_cluster_judges(self, n_clusters=3):
        profile_df = self.build_judge_multistage_matrix()

        profile_df = profile_df.dropna(axis=1, how='all')

        min_valid_scores = len(profile_df.columns) * 0.3
        valid_judges = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_judges]

        profile_df_filled = profile_df.fillna(profile_df.mean(axis=1), axis=0)

        if profile_df_filled.isnull().any().any():
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())

        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)

        if not np.isfinite(profile_scaled).all():
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(profile_scaled)

        results = []
        for i, judge in enumerate(profile_df_filled.index):
            judge_scores = profile_df_filled.loc[judge].dropna()
            results.append({
                'judge': judge,
                'cluster': cluster_labels[i],
                'avg_score': judge_scores.mean(),
                'std_score': judge_scores.std(),
                'num_scores': len(judge_scores)
            })

        results_df = pd.DataFrame(results)

        return results_df, kmeans, profile_df_filled

    def analyze_participant_progression(self):
        all_progressions = []

        for stage in ['stage1', 'stage2', 'stage3', 'final']:
            if stage not in self.stages_data:
                continue

            df = self.stages_data[stage]

            for idx, row in df.iterrows():
                nr = int(row['number'])

                scores = []
                for judge in self.judge_columns:
                    score_val = row[judge]
                    if not pd.isna(score_val) and str(score_val).strip().lower() != 's':
                        try:
                            scores.append(float(score_val))
                        except:
                            pass

                if scores:
                    all_progressions.append({
                        'number': nr,
                        'firstname': row['firstname'],
                        'lastname': row['lastname'],
                        'stage': stage,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'min_score': np.min(scores),
                        'max_score': np.max(scores),
                        'num_judges': len(scores)
                    })

        return pd.DataFrame(all_progressions)

    def cluster_evolution_analysis(self, n_clusters=4):
        stage_order = ['stage1', 'stage2', 'stage3', 'final']
        evolution_data = []

        for i, stage1 in enumerate(stage_order[:-1]):
            stage2 = stage_order[i+1]

            if stage1 not in self.stages_data or stage2 not in self.stages_data:
                continue

            df1 = self.stages_data[stage1]
            df2 = self.stages_data[stage2]

            common_participants = set(df1['number'].astype(int)) & set(df2['number'].astype(int))

            if len(common_participants) < 3:
                continue

            for p_cluster in range(n_clusters):
                transition_counts = {c: 0 for c in range(n_clusters)}

                for nr in common_participants:
                    pass

                evolution_data.append({
                    'from_stage': stage1,
                    'to_stage': stage2,
                    'transition': f"{stage1} -> {stage2}",
                    'num_participants': len(common_participants)
                })

        return pd.DataFrame(evolution_data)


def visualize_participant_dendrogram(linkage_matrix, labels, save_path='participant_dendrogram.png'):
    plt.figure(figsize=(16, 10))

    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8)

    plt.title('Hierarchical Clustering of Participants\n' +
             'Based on scores across all stages',
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Participants', fontsize=12)
    plt.ylabel('Distance (Ward method)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram: {save_path}")


def visualize_judge_dendrogram(linkage_matrix, labels, save_path='judge_dendrogram.png'):
    plt.figure(figsize=(14, 10))

    dendrogram(linkage_matrix, labels=labels, orientation='left', leaf_font_size=10)

    plt.title('Hierarchical Clustering of Judges\n' +
             'Based on scoring patterns across stages',
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Distance (Ward method)', fontsize=12)
    plt.ylabel('Judges', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram: {save_path}")


def visualize_kmeans_clusters(results_df, cluster_stats_df,
                              save_path='participant_clusters.png'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Participants per cluster
    ax = axes[0, 0]
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    ax.bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Number of participants', fontsize=11)
    ax.set_title('Distribution of participants across clusters', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Stages per cluster
    ax = axes[0, 1]
    avg_stages = results_df.groupby('cluster')['stages_count'].mean()
    ax.bar(avg_stages.index, avg_stages.values, color='darkgreen', alpha=0.7)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Average number of stages', fontsize=11)
    ax.set_title('Competition advancement by cluster', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Score profile by stage
    ax = axes[1, 0]
    stage_cols = [col for col in cluster_stats_df.columns if col.startswith('avg_score_')]
    if stage_cols:
        stages = [col.replace('avg_score_', '') for col in stage_cols]
        x_pos = np.arange(len(stages))

        for cluster_id in cluster_stats_df['cluster']:
            scores = cluster_stats_df[cluster_stats_df['cluster'] == cluster_id][stage_cols].values[0]
            ax.plot(x_pos, scores, marker='o', label=f'Cluster {cluster_id}', linewidth=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45)
        ax.set_xlabel('Stage', fontsize=11)
        ax.set_ylabel('Average score', fontsize=11)
        ax.set_title('Score profiles by stage', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # 4. Cluster characteristics
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for _, row in cluster_stats_df.iterrows():
        table_row = [f"Cluster {int(row['cluster'])}",
                    f"{int(row['count'])}",
                    f"{row['avg_stages']:.1f}"]

        for stage_col in stage_cols:
            if not pd.isna(row[stage_col]):
                table_row.append(f"{row[stage_col]:.1f}")
            else:
                table_row.append("-")

        table_data.append(table_row)

    headers = ['Cluster', 'Count', 'Avg Stages'] + [s.replace('avg_score_', '').upper()
                                                   for s in stage_cols]

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Cluster summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'K-Means Clustering Analysis (k={len(cluster_stats_df)})',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization: {save_path}")


def visualize_judge_clusters(results_df, save_path='judge_clusters.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Judges per cluster
    ax = axes[0, 0]
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    ax.bar(cluster_counts.index, cluster_counts.values, color='coral', alpha=0.7)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Number of judges', fontsize=11)
    ax.set_title('Distribution of judges', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Average scores
    ax = axes[0, 1]
    cluster_avg_scores = results_df.groupby('cluster')['avg_score'].mean()
    cluster_std_scores = results_df.groupby('cluster')['avg_score'].std()

    ax.bar(cluster_avg_scores.index, cluster_avg_scores.values,
           yerr=cluster_std_scores.values, color='purple', alpha=0.6, capsize=5)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Average score', fontsize=11)
    ax.set_title('Scoring tendencies by cluster', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Score variation
    ax = axes[1, 0]
    cluster_std = results_df.groupby('cluster')['std_score'].mean()
    ax.bar(cluster_std.index, cluster_std.values, color='teal', alpha=0.7)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Average standard deviation', fontsize=11)
    ax.set_title('Scoring consistency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Table with judges
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for cluster_id in sorted(results_df['cluster'].unique()):
        cluster_judges = results_df[results_df['cluster'] == cluster_id]
        judge_names = ', '.join(cluster_judges.nlargest(5, 'avg_score')['judge'].tolist()[:3])
        if len(cluster_judges) > 3:
            judge_names += f" (+{len(cluster_judges)-3} others)"

        table_data.append([
            f"Cluster {cluster_id}",
            f"{len(cluster_judges)}",
            f"{cluster_judges['avg_score'].mean():.1f}",
            f"{cluster_judges['std_score'].mean():.2f}",
            judge_names[:40]
        ])

    headers = ['Cluster', 'Count', 'Avg Score', 'Avg Std', 'Judges (sample)']
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title('Cluster members', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'Judge Clustering Analysis (k={len(results_df["cluster"].unique())})',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved judge cluster visualization: {save_path}")


def visualize_participant_progression(progression_df, save_path='participant_progression.png'):
    fig, ax = plt.subplots(figsize=(14, 8))

    filtered_df = progression_df.copy()
    stage_counts = filtered_df.groupby('number').size()
    participants_3plus = stage_counts[stage_counts >= 3].index

    if len(participants_3plus) == 0:
        print("No participants with 3+ stages to visualize")
        return

    np.random.seed(42)
    selected_participants = np.random.choice(participants_3plus,
                                           size=min(12, len(participants_3plus)),
                                           replace=False)

    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    stage_labels = {'stage1': "Stage 1", 'stage2': "Stage 2", 'stage3': 'Stage 3', 'final': 'Final'}

    for nr in selected_participants:
        participant_data = filtered_df[filtered_df['number'] == nr].sort_values('stage',
                                      key=lambda x: x.map({s: i for i, s in enumerate(stage_order)}))

        if len(participant_data) >= 2:
            stages = [stage_labels.get(s, s) for s in participant_data['stage']]
            scores = participant_data['mean_score'].values

            label = f"{nr}: {participant_data.iloc[0]['firstname']} {participant_data.iloc[0]['lastname']}"
            ax.plot(stages, scores, marker='o', linewidth=2, label=label, alpha=0.7)

    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Average score', fontsize=12)
    ax.set_title('Participant progression through competition stages\n' +
                '(participants who advanced through 3+ stages)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved progression visualization: {save_path}")


def visualize_stage_correlation_heatmap(analyzer, save_path='stage_correlation_heatmap.png'):
    profile_df = analyzer.build_judge_multistage_matrix()

    stage_order = ['stage1', 'stage2', 'stage3', 'final']

    judge_stage_correlations = []

    for judge in analyzer.judge_columns:
        judge_row = profile_df.loc[judge]

        stage_scores = {}
        for stage in stage_order:
            stage_cols = [col for col in judge_row.index if col.endswith(f'_{stage}')]
            stage_scores[stage] = judge_row[stage_cols].dropna()

        corr_row = {'judge': judge}
        for i, stage1 in enumerate(stage_order):
            for stage2 in stage_order[i+1:]:
                if len(stage_scores[stage1]) > 0 and len(stage_scores[stage2]) > 0:
                    common_participants = set([col.split('_')[0] for col in stage_scores[stage1].index]) &\
                                        set([col.split('_')[0] for col in stage_scores[stage2].index])

                    if len(common_participants) >= 3:
                        scores1 = []
                        scores2 = []
                        for p in common_participants:
                            col1 = f"{p}_{stage1}"
                            col2 = f"{p}_{stage2}"
                            if col1 in stage_scores[stage1].index and col2 in stage_scores[stage2].index:
                                scores1.append(stage_scores[stage1][col1])
                                scores2.append(stage_scores[stage2][col2])

                        if len(scores1) >= 3:
                            corr = np.corrcoef(scores1, scores2)[0, 1]
                            corr_row[f'{stage1}_vs_{stage2}'] = corr

        if len(corr_row) > 1:
            judge_stage_correlations.append(corr_row)

    if not judge_stage_correlations:
        print("No data to calculate correlations between stages")
        return

    corr_df = pd.DataFrame(judge_stage_correlations)
    corr_df = corr_df.set_index('judge')

    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
               vmin=0, vmax=1, ax=ax, cbar_kws={'label': "Correlation"})

    ax.set_title('Judge scoring correlation between stages\n' +
                '(for the same participants across different stages)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Stage comparison', fontsize=12)
    ax.set_ylabel('Judge', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap: {save_path}")


def run_multistage_analysis(data_files, output_dir='multistage_results'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("MULTISTAGE CHOPIN COMPETITION ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = MultiStageClusteringAnalyzer(data_files)

    print("\n1. Hierarchical clustering of participants…")
    try:
        linkage_matrix, labels, profile_df, participant_info =\
            analyzer.cluster_participants_hierarchical(min_stages=2)
        if linkage_matrix is not None:
            visualize_participant_dendrogram(linkage_matrix, labels,
                                            f'{output_dir}/participant_dendrogram.png')
        else:
            print("  Skipped - insufficient data")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. Hierarchical clustering of judges…")
    try:
        linkage_matrix_judges, judge_labels, judge_profile_df =\
            analyzer.cluster_judges_hierarchical()
        if linkage_matrix_judges is not None:
            visualize_judge_dendrogram(linkage_matrix_judges, judge_labels,
                                       f'{output_dir}/judge_dendrogram.png')
        else:
            print("  Skipped - insufficient data")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n3. K-means clustering of participants…")
    for n_clusters in [3, 4, 5]:
        try:
            results_df, cluster_stats_df, kmeans, scaler, profile_filled =\
                analyzer.kmeans_cluster_participants(n_clusters=n_clusters, min_stages=2)

            # Save result
            results_df.to_csv(f'{output_dir}/participant_clusters_k{n_clusters}.csv',
                            index=False, encoding='utf-8-sig')
            cluster_stats_df.to_csv(f'{output_dir}/participant_cluster_stats_k{n_clusters}.csv',
                                   index=False, encoding='utf-8-sig')

            # Visualisation
            visualize_kmeans_clusters(results_df, cluster_stats_df,
                                    f'{output_dir}/participant_clusters_k{n_clusters}.png')
        except Exception as e:
            print(f"  Error for k={n_clusters}: {e}")

    print("\n4. K-means clustering of judges…")
    for n_clusters in [2, 3, 4]:
        try:
            judge_results_df, judge_kmeans, judge_profile_filled =\
                analyzer.kmeans_cluster_judges(n_clusters=n_clusters)

            judge_results_df.to_csv(f'{output_dir}/judge_clusters_k{n_clusters}.csv',
                                   index=False, encoding='utf-8-sig')

            visualize_judge_clusters(judge_results_df,
                                    f'{output_dir}/judge_clusters_k{n_clusters}.png')
        except Exception as e:
            print(f"  Error for k={n_clusters}: {e}")

    print("\n5. Participant progression analysis…")
    progression_df = analyzer.analyze_participant_progression()
    progression_df.to_csv(f'{output_dir}/participant_progression.csv',
                         index=False, encoding='utf-8-sig')
    visualize_participant_progression(progression_df,
                                     f'{output_dir}/participant_progression.png')

    print("\n6. Judge scoring correlation between stages…")
    try:
        visualize_stage_correlation_heatmap(analyzer,
                                          f'{output_dir}/stage_correlation_heatmap.png')
    except Exception as e:
        print(f"  Correlation heatmap error: {e}")

    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved in: {output_dir}")
    print("=" * 80)

    return analyzer


if __name__ == "__main__":
    data_files = {
        'stage1': "Chopin_2025_stage1_by_judge.csv",
        'stage2': "Chopin_2025_stage2_by_judge.csv",
        'stage3': "Chopin_2025_stage3_by_judge.csv",
        'final': "Chopin_2025_final_by_judge.csv"
    }

    analyzer = run_multistage_analysis(data_files, output_dir='multistage_results')