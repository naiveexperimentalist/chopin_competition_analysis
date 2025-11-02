"""
Moduł do clusteringu uczestników i analizy PCA sędziów
- K-means i hierarchical clustering uczestników
- PCA dla profili sędziów
- Identyfikacja "stylów" wykonania
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ChopinClusteringAnalyzer:
    """Analiza clusteringu uczestników i PCA sędziów"""
    
    def __init__(self, processor):
        self.processor = processor
        self.judge_columns = processor.judge_columns
        self.stages_data = processor.stages_data
        self.corrected_data = processor.corrected_data
    
    def prepare_participant_score_matrix(self, stage: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Przygotowuje macierz ocen: uczestnicy (wiersze) vs sędziowie (kolumny)
        """
        if stage and stage in self.stages_data:
            df = self.stages_data[stage]
            participant_labels = []
            score_matrix = []
            
            for idx, row in df.iterrows():
                label = f"{row['Nr']}: {row['imię']} {row['nazwisko']}"
                participant_labels.append(label)
                
                scores = []
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    scores.append(score if pd.notna(score) else np.nan)
                
                score_matrix.append(scores)
            
            score_df = pd.DataFrame(score_matrix, 
                                   index=participant_labels,
                                   columns=self.judge_columns)
            
            return score_df, participant_labels
        
        else:
            # Zbierz dane ze wszystkich etapów
            participant_data = {}
            
            for stage_name, df in self.stages_data.items():
                for idx, row in df.iterrows():
                    nr = row['Nr']
                    label = f"{row['Nr']}: {row['imię']} {row['nazwisko']}"
                    
                    if nr not in participant_data:
                        participant_data[nr] = {
                            'label': label,
                            'scores': {judge: [] for judge in self.judge_columns}
                        }
                    
                    for judge in self.judge_columns:
                        score = pd.to_numeric(row[judge], errors='coerce')
                        if pd.notna(score):
                            participant_data[nr]['scores'][judge].append(score)
            
            # Uśrednij oceny z różnych etapów
            participant_labels = []
            score_matrix = []
            
            for nr, data in participant_data.items():
                participant_labels.append(data['label'])
                
                avg_scores = []
                for judge in self.judge_columns:
                    scores = data['scores'][judge]
                    avg_score = np.mean(scores) if len(scores) > 0 else np.nan
                    avg_scores.append(avg_score)
                
                score_matrix.append(avg_scores)
            
            score_df = pd.DataFrame(score_matrix,
                                   index=participant_labels,
                                   columns=self.judge_columns)
            
            return score_df, participant_labels
    
    def kmeans_clustering_participants(self, stage: str = None, n_clusters: int = 5) -> pd.DataFrame:
        """
        K-means clustering uczestników na podstawie otrzymanych ocen
        Identyfikuje grupy uczestników z podobnymi profilami ocen
        """
        score_df, participant_labels = self.prepare_participant_score_matrix(stage)
        
        # Usuń wiersze z zbyt wieloma brakami
        min_scores = len(self.judge_columns) * 0.5  # Minimum 50% ocen
        valid_participants = score_df.count(axis=1) >= min_scores
        score_df_clean = score_df[valid_participants].fillna(score_df.mean())
        
        if len(score_df_clean) < n_clusters:
            print(f"Za mało uczestników dla {n_clusters} klastrów")
            return pd.DataFrame()
        
        # Standaryzacja
        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(score_df_clean)
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(scores_scaled)
        
        # Przygotuj wyniki
        results = []
        for i, (label, cluster) in enumerate(zip(score_df_clean.index, clusters)):
            # Pobierz oryginalne wyniki
            participant_scores = score_df_clean.iloc[i]
            
            results.append({
                'participant': label,
                'cluster': int(cluster),
                'mean_score': participant_scores.mean(),
                'std_score': participant_scores.std(),
                'distance_to_center': np.linalg.norm(scores_scaled[i] - kmeans.cluster_centers_[cluster])
            })
        
        results_df = pd.DataFrame(results)
        
        # Dodaj statystyki klastrów
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_members = results_df[results_df['cluster'] == cluster_id]
            
            cluster_stats.append({
                'cluster': cluster_id,
                'n_members': len(cluster_members),
                'avg_mean_score': cluster_members['mean_score'].mean(),
                'avg_std_score': cluster_members['std_score'].mean()
            })
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        
        return results_df, cluster_stats_df, kmeans.cluster_centers_, scaler
    
    def hierarchical_clustering_participants(self, stage: str = None, 
                                            method: str = 'ward') -> Tuple[np.ndarray, List[str]]:
        """
        Hierarchical clustering uczestników
        Zwraca linkage matrix i etykiety do dendrogramu
        """
        score_df, participant_labels = self.prepare_participant_score_matrix(stage)
        
        # Usuń wiersze z brakami
        score_df_clean = score_df.dropna()
        
        if len(score_df_clean) < 2:
            print("Za mało uczestników bez braków")
            return None, []
        
        # Standaryzacja
        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(score_df_clean)
        
        # Hierarchical clustering
        linkage_matrix = linkage(scores_scaled, method=method)
        
        return linkage_matrix, score_df_clean.index.tolist()
    
    def identify_participant_styles(self, stage: str = None, n_clusters: int = 5) -> pd.DataFrame:
        """
        Identyfikuje "style" wykonań - grupy uczestników podobnie ocenianych
        Opisuje każdy styl przez charakterystyki sędziowskie
        """
        score_df, participant_labels = self.prepare_participant_score_matrix(stage)
        
        # Clustering
        results_df, cluster_stats_df, cluster_centers, scaler = \
            self.kmeans_clustering_participants(stage, n_clusters)
        
        if results_df.empty:
            return pd.DataFrame()
        
        # Analizuj charakterystyki każdego klastra
        style_descriptions = []
        
        score_df_clean = score_df.dropna()
        
        for cluster_id in range(n_clusters):
            cluster_members = results_df[results_df['cluster'] == cluster_id]['participant'].tolist()
            
            # Pobierz oceny członków klastra
            cluster_scores = score_df_clean.loc[
                [p for p in cluster_members if p in score_df_clean.index]
            ]
            
            if len(cluster_scores) == 0:
                continue
            
            # Średnie oceny od każdego sędziego dla tego klastra
            judge_means = cluster_scores.mean()
            judge_stds = cluster_scores.std()
            
            # Zidentyfikuj sędziów którzy faworyzują ten klaster
            overall_means = score_df_clean.mean()
            favorable_judges = []
            unfavorable_judges = []
            
            for judge in self.judge_columns:
                if judge in judge_means.index and judge in overall_means.index:
                    diff = judge_means[judge] - overall_means[judge]
                    if diff > 1.0:
                        favorable_judges.append((judge, diff))
                    elif diff < -1.0:
                        unfavorable_judges.append((judge, diff))
            
            # Sortuj
            favorable_judges.sort(key=lambda x: x[1], reverse=True)
            unfavorable_judges.sort(key=lambda x: x[1])
            
            style_descriptions.append({
                'cluster': cluster_id,
                'n_participants': len(cluster_members),
                'avg_score': judge_means.mean(),
                'favorable_judges': ', '.join([f"{j[0].split()[-1]} (+{j[1]:.1f})" 
                                              for j in favorable_judges[:3]]),
                'unfavorable_judges': ', '.join([f"{j[0].split()[-1]} ({j[1]:.1f})" 
                                                for j in unfavorable_judges[:3]]),
                'score_variability': judge_stds.mean(),
                'top_member': cluster_members[0] if cluster_members else ''
            })
        
        return pd.DataFrame(style_descriptions)
    
    def pca_judge_profiles(self) -> Tuple[PCA, pd.DataFrame, np.ndarray]:
        """
        PCA na profilach sędziów
        Każdy sędzia = punkt w przestrzeni uczestników
        Identyfikuje główne wymiary różnic między sędziami
        """
        # Przygotuj macierz: sędziowie (wiersze) vs uczestnicy (kolumny)
        # Zbierz wszystkie oceny
        participant_judge_matrix = {}
        
        for stage_name, df in self.stages_data.items():
            for idx, row in df.iterrows():
                participant_key = f"{row['Nr']}_{stage_name}"
                
                for judge in self.judge_columns:
                    score = pd.to_numeric(row[judge], errors='coerce')
                    
                    if judge not in participant_judge_matrix:
                        participant_judge_matrix[judge] = {}
                    
                    if pd.notna(score):
                        participant_judge_matrix[judge][participant_key] = score
        
        # Konwertuj do DataFrame
        judge_profiles = []
        judge_names = []
        
        for judge, scores in participant_judge_matrix.items():
            judge_names.append(judge)
            judge_profiles.append(list(scores.values()))
        
        # Uzupełnij braki średnią
        max_len = max(len(p) for p in judge_profiles)
        
        for i in range(len(judge_profiles)):
            while len(judge_profiles[i]) < max_len:
                judge_profiles[i].append(np.nan)
        
        judge_df = pd.DataFrame(judge_profiles, index=judge_names)
        judge_df_filled = judge_df.fillna(judge_df.mean())
        
        # Standaryzacja
        scaler = StandardScaler()
        judge_scaled = scaler.fit_transform(judge_df_filled)
        
        # PCA
        pca = PCA(n_components=min(5, len(self.judge_columns)))
        judge_pca = pca.fit_transform(judge_scaled)
        
        # Przygotuj wyniki
        pca_results = []
        for i, judge in enumerate(judge_names):
            result = {'judge': judge}
            for j in range(pca.n_components_):
                result[f'PC{j+1}'] = judge_pca[i, j]
            pca_results.append(result)
        
        pca_df = pd.DataFrame(pca_results)
        
        return pca, pca_df, pca.explained_variance_ratio_
    
    def interpret_pca_components(self, pca: PCA, n_components: int = 3) -> pd.DataFrame:
        """
        Interpretuje główne komponenty PCA
        Pokazuje które wymiary oceniania są najważniejsze
        """
        component_interpretations = []
        
        for i in range(min(n_components, pca.n_components_)):
            component = pca.components_[i]
            variance_explained = pca.explained_variance_ratio_[i]
            
            # Znajdź najbardziej wpływowe cechy
            abs_component = np.abs(component)
            top_indices = abs_component.argsort()[-5:][::-1]
            
            top_features = []
            for idx in top_indices:
                weight = component[idx]
                top_features.append(f"Feature {idx}: {weight:.3f}")
            
            component_interpretations.append({
                'component': f'PC{i+1}',
                'variance_explained': variance_explained,
                'cumulative_variance': pca.explained_variance_ratio_[:i+1].sum(),
                'top_features': ' | '.join(top_features)
            })
        
        return pd.DataFrame(component_interpretations)
    
    def find_similar_participants(self, participant_nr: int, stage: str = None, 
                                  top_n: int = 5) -> pd.DataFrame:
        """
        Znajduje uczestników najbardziej podobnych do danego uczestnika
        (na podstawie profilu ocen od sędziów)
        """
        score_df, participant_labels = self.prepare_participant_score_matrix(stage)
        
        # Znajdź wiersz dla danego uczestnika
        target_row = None
        target_label = None
        
        for label in participant_labels:
            if label.startswith(f"{participant_nr}:"):
                target_label = label
                target_row = score_df.loc[label]
                break
        
        if target_row is None:
            print(f"Nie znaleziono uczestnika nr {participant_nr}")
            return pd.DataFrame()
        
        # Oblicz podobieństwo do wszystkich innych
        similarities = []
        
        for label in participant_labels:
            if label == target_label:
                continue
            
            other_row = score_df.loc[label]
            
            # Usuń brakujące wartości
            valid_mask = ~(target_row.isna() | other_row.isna())
            
            if valid_mask.sum() >= 3:
                # Korelacja Pearsona
                corr, _ = stats.pearsonr(target_row[valid_mask], other_row[valid_mask])
                
                # Odległość euklidesowa (znormalizowana)
                euclidean_dist = np.sqrt(((target_row[valid_mask] - other_row[valid_mask])**2).sum())
                
                similarities.append({
                    'participant': label,
                    'correlation': corr,
                    'euclidean_distance': euclidean_dist,
                    'n_common_judges': valid_mask.sum()
                })
        
        similarities_df = pd.DataFrame(similarities)
        
        if not similarities_df.empty:
            # Sortuj po korelacji (malejąco)
            similarities_df = similarities_df.sort_values('correlation', ascending=False)
            return similarities_df.head(top_n)
        
        return pd.DataFrame()
    
    def cluster_stability_analysis(self, stage: str = None, n_clusters_range: range = range(2, 8)) -> pd.DataFrame:
        """
        Analizuje stabilność clusteringu dla różnej liczby klastrów
        Pomaga wybrać optymalną liczbę klastrów (elbow method, silhouette)
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        score_df, participant_labels = self.prepare_participant_score_matrix(stage)
        
        # Przygotuj dane
        min_scores = len(self.judge_columns) * 0.5
        valid_participants = score_df.count(axis=1) >= min_scores
        score_df_clean = score_df[valid_participants].fillna(score_df.mean())
        
        if len(score_df_clean) < max(n_clusters_range):
            print(f"Za mało uczestników dla analizy {max(n_clusters_range)} klastrów")
            return pd.DataFrame()
        
        # Standaryzacja
        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(score_df_clean)
        
        # Testuj różne liczby klastrów
        stability_results = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            cluster_labels = kmeans.fit_predict(scores_scaled)
            
            # Inertia (suma kwadratów odległości do centrów)
            inertia = kmeans.inertia_
            
            # Silhouette score
            silhouette = silhouette_score(scores_scaled, cluster_labels)
            
            # Calinski-Harabasz score
            calinski = calinski_harabasz_score(scores_scaled, cluster_labels)
            
            stability_results.append({
                'n_clusters': n_clusters,
                'inertia': inertia,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            })
        
        return pd.DataFrame(stability_results)


def run_clustering_analysis(processor, output_dir: str = 'clustering_results'):
    """Uruchamia wszystkie analizy clusteringowe"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ChopinClusteringAnalyzer(processor)
    
    # 1. K-means clustering uczestników
    print("Clustering uczestników (K-means)...")
    for stage in [None, 'stage1', 'stage2', 'stage3', 'final']:
        stage_name = stage if stage else 'all_stages'
        try:
            clusters_df, cluster_stats, _, _ = analyzer.kmeans_clustering_participants(stage, n_clusters=5)
            if not clusters_df.empty:
                clusters_df.to_csv(f'{output_dir}/kmeans_clusters_{stage_name}.csv', index=False)
                cluster_stats.to_csv(f'{output_dir}/kmeans_stats_{stage_name}.csv', index=False)
        except Exception as e:
            print(f"  Błąd dla {stage_name}: {e}")
    
    # 2. Identyfikacja stylów
    print("Identyfikacja stylów wykonania...")
    for stage in [None, 'final']:
        stage_name = stage if stage else 'all_stages'
        try:
            styles = analyzer.identify_participant_styles(stage, n_clusters=5)
            if not styles.empty:
                styles.to_csv(f'{output_dir}/participant_styles_{stage_name}.csv', index=False)
        except Exception as e:
            print(f"  Błąd dla {stage_name}: {e}")
    
    # 3. PCA sędziów
    print("Analiza PCA profili sędziów...")
    try:
        pca, pca_df, variance_ratio = analyzer.pca_judge_profiles()
        pca_df.to_csv(f'{output_dir}/judge_pca_scores.csv', index=False)
        
        # Variance explained
        variance_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(variance_ratio))],
            'variance_explained': variance_ratio,
            'cumulative_variance': np.cumsum(variance_ratio)
        })
        variance_df.to_csv(f'{output_dir}/pca_variance_explained.csv', index=False)
        
        # Interpretacja komponentów
        interpretation = analyzer.interpret_pca_components(pca, n_components=3)
        interpretation.to_csv(f'{output_dir}/pca_interpretation.csv', index=False)
    except Exception as e:
        print(f"  Błąd w PCA: {e}")
    
    # 4. Stabilność clusteringu
    print("Analiza stabilności clusteringu...")
    try:
        stability = analyzer.cluster_stability_analysis(stage='final', n_clusters_range=range(2, 10))
        if not stability.empty:
            stability.to_csv(f'{output_dir}/clustering_stability.csv', index=False)
    except Exception as e:
        print(f"  Błąd w analizie stabilności: {e}")
    
    # 5. Podobni uczestnicy (przykład dla pierwszych 5)
    print("Szukam podobnych uczestników...")
    try:
        if 'final' in processor.corrected_data:
            final_df = processor.corrected_data['final']
            for nr in final_df['Nr'].head(5):
                similar = analyzer.find_similar_participants(nr, stage='final', top_n=5)
                if not similar.empty:
                    similar.to_csv(f'{output_dir}/similar_to_{nr}.csv', index=False)
    except Exception as e:
        print(f"  Błąd w szukaniu podobnych: {e}")
    
    print(f"\nAnalizy clusteringowe zapisane w: {output_dir}")
    
    return analyzer


if __name__ == "__main__":
    print("Uruchom najpierw chopin_data_processor.py, a następnie ten skrypt z obiektem processor")
