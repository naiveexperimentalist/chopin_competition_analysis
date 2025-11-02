"""
Analiza klastrowa uczestników i sędziów przez wszystkie etapy konkursu Chopinowskiego

Śledzi uczestników przez wszystkie rundy i identyfikuje:
- Klastry sędziów na podstawie ich wzorców oceniania przez cały konkurs
- Klastry uczestników na podstawie profili ocen ze wszystkich etapów
- Dynamikę ocen i wzorce progresji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MultiStageClusteringAnalyzer:
    """Analiza klastrowa przez wszystkie etapy konkursu"""
    
    def __init__(self, data_files):
        """
        Args:
            data_files: dict z kluczami 'stage1', 'stage2', 'stage3', 'final'
                       i wartościami - ścieżkami do plików CSV
        """
        self.data_files = data_files
        self.stages_data = {}
        self.judge_columns = []
        self.load_data()
        
    def load_data(self):
        """Wczytuje dane ze wszystkich etapów"""
        print("Wczytuję dane ze wszystkich etapów...")
        
        for stage, filepath in self.data_files.items():
            try:
                df = pd.read_csv(filepath)
                # Usuń spacje z nazw kolumn
                df.columns = df.columns.str.strip()
                self.stages_data[stage] = df
                print(f"  {stage}: {len(df)} uczestników")
                
                # Pobierz kolumny sędziów (pomijając Nr, imię, nazwisko)
                if not self.judge_columns:
                    self.judge_columns = [col for col in df.columns 
                                         if col not in ['Nr', 'imię', 'nazwisko']]
            except Exception as e:
                print(f"  Błąd wczytywania {stage}: {e}")
        
        print(f"Znaleziono {len(self.judge_columns)} sędziów")
    
    def build_participant_multistage_matrix(self):
        """
        Buduje macierz cech dla uczestników przez wszystkie etapy.
        Każdy uczestnik ma wektor: [Judge1_Stage1, Judge2_Stage1, ..., Judge1_Stage2, ...]
        
        Returns:
            pd.DataFrame: wiersze=uczestnicy, kolumny=sędzia_etap
            dict: mapowanie nr uczestnika -> info (imię, nazwisko, etapy)
        """
        participant_profiles = {}
        participant_info = {}
        
        stage_order = ['stage1', 'stage2', 'stage3', 'final']
        
        # Zbierz wszystkich uczestników i ich oceny z każdego etapu
        for stage in stage_order:
            if stage not in self.stages_data:
                continue
                
            df = self.stages_data[stage]
            
            for idx, row in df.iterrows():
                nr = int(row['Nr'])
                
                # Inicjalizuj profil uczestnika
                if nr not in participant_profiles:
                    participant_profiles[nr] = {}
                    participant_info[nr] = {
                        'imię': row['imię'],
                        'nazwisko': row['nazwisko'],
                        'stages': []
                    }
                
                participant_info[nr]['stages'].append(stage)
                
                # Zbierz oceny od wszystkich sędziów w tym etapie
                for judge in self.judge_columns:
                    col_name = f"{judge}_{stage}"
                    score_val = row[judge]
                    
                    # Konwersja do liczby, obsługa 's' i braków
                    if pd.isna(score_val) or str(score_val).strip().lower() == 's':
                        participant_profiles[nr][col_name] = np.nan
                    else:
                        try:
                            participant_profiles[nr][col_name] = float(score_val)
                        except:
                            participant_profiles[nr][col_name] = np.nan
        
        # Konwersja do DataFrame
        profile_df = pd.DataFrame.from_dict(participant_profiles, orient='index')
        profile_df = profile_df.sort_index()  # Sortuj po numerze uczestnika
        
        print(f"\nMacierz wieloetapowa: {len(profile_df)} uczestników x {len(profile_df.columns)} cech")
        print(f"Zakres etapów uczestników:")
        for nr, info in sorted(participant_info.items())[:5]:
            print(f"  Uczestnik {nr} ({info['imię']} {info['nazwisko']}): {info['stages']}")
        
        return profile_df, participant_info
    
    def build_judge_multistage_matrix(self):
        """
        Buduje macierz cech dla sędziów przez wszystkie etapy.
        Każdy sędzia ma wektor ocen ze wszystkich etapów dla wszystkich uczestników.
        
        Returns:
            pd.DataFrame: wiersze=sędziowie, kolumny=uczestnik_etap
        """
        judge_profiles = {judge: {} for judge in self.judge_columns}
        
        stage_order = ['stage1', 'stage2', 'stage3', 'final']
        
        # Zbierz oceny każdego sędziego dla każdego uczestnika w każdym etapie
        for stage in stage_order:
            if stage not in self.stages_data:
                continue
                
            df = self.stages_data[stage]
            
            for idx, row in df.iterrows():
                nr = int(row['Nr'])
                participant_label = f"P{nr}_{stage}"
                
                for judge in self.judge_columns:
                    score_val = row[judge]
                    
                    # Konwersja do liczby
                    if pd.isna(score_val) or str(score_val).strip().lower() == 's':
                        judge_profiles[judge][participant_label] = np.nan
                    else:
                        try:
                            judge_profiles[judge][participant_label] = float(score_val)
                        except:
                            judge_profiles[judge][participant_label] = np.nan
        
        # Konwersja do DataFrame
        profile_df = pd.DataFrame.from_dict(judge_profiles, orient='index')
        
        print(f"\nMacierz sędziów: {len(profile_df)} sędziów x {len(profile_df.columns)} ocen")
        
        return profile_df
    
    def cluster_participants_hierarchical(self, min_stages=2):
        """
        Hierarchiczne klasterowanie uczestników na podstawie pełnych profili wieloetapowych
        
        Args:
            min_stages: minimalna liczba etapów w których uczestnik musiał wystąpić
        """
        profile_df, participant_info = self.build_participant_multistage_matrix()
        
        # Filtruj uczestników którzy wystąpili w wystarczającej liczbie etapów
        valid_participants = [nr for nr, info in participant_info.items() 
                             if len(info['stages']) >= min_stages]
        profile_df = profile_df.loc[valid_participants]
        
        # Usuń kolumny z samymi NaN
        profile_df = profile_df.dropna(axis=1, how='all')
        
        # Usuń uczestników którzy mają zbyt mało danych (< 30% ocen)
        min_valid_scores = len(profile_df.columns) * 0.3
        valid_rows = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_rows]
        
        if len(profile_df) < 2:
            print("Za mało uczestników z wystarczającą liczbą ocen")
            return None, [], pd.DataFrame(), {}
        
        # Wypełnij braki średnią (dla każdej kolumny osobno)
        profile_df_filled = profile_df.fillna(profile_df.mean())
        
        # Sprawdź czy są jeszcze NaN
        if profile_df_filled.isnull().any().any():
            # Wypełnij pozostałe NaN średnią globalną
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())
        
        print(f"\nKlasterowanie {len(profile_df_filled)} uczestników " +
              f"(min {min_stages} etapów)")
        
        # Standaryzacja
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)
        
        # Sprawdź czy są wartości inf/nan po standaryzacji
        if not np.isfinite(profile_scaled).all():
            print("UWAGA: Wykryto wartości nieskończone po standaryzacji")
            # Zamień inf/nan na 0
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Hierarchiczne klasterowanie
        try:
            linkage_matrix = linkage(profile_scaled, method='ward')
        except ValueError as e:
            print(f"UWAGA: Użycie metody 'average' zamiast 'ward' (błąd: {e})")
            linkage_matrix = linkage(profile_scaled, method='average')
        
        # Etykiety dla dendrogramu
        labels = [f"{nr}: {participant_info[nr]['imię']} {participant_info[nr]['nazwisko']}" 
                 for nr in profile_df_filled.index]
        
        return linkage_matrix, labels, profile_df_filled, participant_info
    
    def cluster_judges_hierarchical(self):
        """
        Hierarchiczne klasterowanie sędziów na podstawie ich wzorców oceniania
        przez wszystkie etapy
        """
        profile_df = self.build_judge_multistage_matrix()
        
        # Usuń kolumny z samymi NaN
        profile_df = profile_df.dropna(axis=1, how='all')
        
        # Usuń sędziów którzy mają zbyt mało danych (< 30% ocen)
        min_valid_scores = len(profile_df.columns) * 0.3
        valid_judges = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_judges]
        
        if len(profile_df) < 2:
            print("Za mało sędziów z wystarczającą liczbą ocen")
            return None, [], pd.DataFrame()
        
        # Wypełnij braki średnią sędziego
        profile_df_filled = profile_df.fillna(profile_df.mean(axis=1), axis=0)
        
        # Sprawdź czy są jeszcze NaN (może być jeśli sędzia ma same NaN)
        if profile_df_filled.isnull().any().any():
            # Wypełnij pozostałe NaN średnią globalną
            profile_df_filled = profile_df_filled.fillna(profile_df_filled.mean().mean())
        
        print(f"\nKlasterowanie {len(profile_df_filled)} sędziów")
        
        # Standaryzacja
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_df_filled)
        
        # Sprawdź czy są wartości inf/nan po standaryzacji
        if not np.isfinite(profile_scaled).all():
            print("UWAGA: Wykryto wartości nieskończone po standaryzacji, używam metody 'average'")
            # Zamień inf/nan na 0
            profile_scaled = np.nan_to_num(profile_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Hierarchiczne klasterowanie - użyj 'average' jeśli są problemy
        try:
            linkage_matrix = linkage(profile_scaled, method='ward')
        except ValueError:
            print("UWAGA: Użycie metody 'average' zamiast 'ward' (problemy z danymi)")
            linkage_matrix = linkage(profile_scaled, method='average')
        
        return linkage_matrix, profile_df_filled.index.tolist(), profile_df_filled
    
    def kmeans_cluster_participants(self, n_clusters=5, min_stages=2):
        """
        K-means klasterowanie uczestników
        
        Returns:
            DataFrame z przypisaniami do klastrów i statystykami
        """
        profile_df, participant_info = self.build_participant_multistage_matrix()
        
        # Filtruj uczestników
        valid_participants = [nr for nr, info in participant_info.items() 
                             if len(info['stages']) >= min_stages]
        profile_df = profile_df.loc[valid_participants]
        profile_df = profile_df.dropna(axis=1, how='all')
        
        # Usuń uczestników z zbyt małą liczbą ocen
        min_valid_scores = len(profile_df.columns) * 0.3
        valid_rows = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_rows]
        
        if len(profile_df) < n_clusters:
            print(f"Za mało uczestników ({len(profile_df)}) dla {n_clusters} klastrów")
            return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()
        
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
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
        
        # Przygotuj wyniki
        results = []
        for i, nr in enumerate(profile_df_filled.index):
            info = participant_info[nr]
            
            results.append({
                'Nr': nr,
                'imię': info['imię'],
                'nazwisko': info['nazwisko'],
                'stages': ', '.join(info['stages']),
                'n_stages': len(info['stages']),
                'cluster': int(clusters[i]),
                'mean_score': profile_df_filled.iloc[i].mean(),
                'std_score': profile_df_filled.iloc[i].std()
            })
        
        results_df = pd.DataFrame(results)
        
        # Statystyki klastrów
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_members = results_df[results_df['cluster'] == cluster_id]
            
            cluster_stats.append({
                'cluster': cluster_id,
                'n_members': len(cluster_members),
                'avg_mean_score': cluster_members['mean_score'].mean(),
                'avg_std_score': cluster_members['std_score'].mean(),
                'avg_n_stages': cluster_members['n_stages'].mean()
            })
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        
        return results_df, cluster_stats_df, kmeans, scaler, profile_df_filled
    
    def kmeans_cluster_judges(self, n_clusters=3):
        """
        K-means klasterowanie sędziów
        """
        profile_df = self.build_judge_multistage_matrix()
        profile_df = profile_df.dropna(axis=1, how='all')
        
        # Usuń sędziów z zbyt małą liczbą ocen
        min_valid_scores = len(profile_df.columns) * 0.3
        valid_judges = profile_df.count(axis=1) >= min_valid_scores
        profile_df = profile_df[valid_judges]
        
        if len(profile_df) < n_clusters:
            print(f"Za mało sędziów ({len(profile_df)}) dla {n_clusters} klastrów")
            return pd.DataFrame(), None, pd.DataFrame()
        
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
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(profile_scaled)
        
        # Wyniki
        results = []
        for i, judge in enumerate(profile_df_filled.index):
            results.append({
                'judge': judge,
                'cluster': int(clusters[i]),
                'mean_score': profile_df_filled.iloc[i].mean(),
                'std_score': profile_df_filled.iloc[i].std(),
                'n_scores': profile_df_filled.iloc[i].notna().sum()
            })
        
        results_df = pd.DataFrame(results)
        
        return results_df, kmeans, profile_df_filled
    
    def analyze_participant_progression(self):
        """
        Analizuje jak uczestnicy progresowali przez etapy
        Zwraca DataFrame z średnimi ocenami w każdym etapie dla każdego uczestnika
        """
        stage_order = ['stage1', 'stage2', 'stage3', 'final']
        progression_data = []
        
        for stage in stage_order:
            if stage not in self.stages_data:
                continue
                
            df = self.stages_data[stage]
            
            for idx, row in df.iterrows():
                nr = int(row['Nr'])
                
                # Zbierz wszystkie oceny (pomijając 's' i braki)
                scores = []
                for judge in self.judge_columns:
                    score_val = row[judge]
                    if pd.notna(score_val) and str(score_val).strip().lower() != 's':
                        try:
                            scores.append(float(score_val))
                        except:
                            pass
                
                if scores:
                    progression_data.append({
                        'Nr': nr,
                        'imię': row['imię'],
                        'nazwisko': row['nazwisko'],
                        'stage': stage,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'n_scores': len(scores)
                    })
        
        return pd.DataFrame(progression_data)


def visualize_participant_dendrogram(linkage_matrix, labels, save_path='participant_dendrogram.png'):
    """Wizualizacja dendrogramu dla uczestników"""
    fig, ax = plt.subplots(figsize=(20, max(12, len(labels) * 0.15)))
    
    dendrogram(linkage_matrix, 
              labels=labels,
              orientation='right',
              leaf_font_size=8,
              ax=ax)
    
    ax.set_title('Hierarchiczne klasterowanie uczestników\n' +
                'na podstawie pełnych profili ocen ze wszystkich etapów',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Odległość', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano dendrogram uczestników: {save_path}")


def visualize_judge_dendrogram(linkage_matrix, labels, save_path='judge_dendrogram.png'):
    """Wizualizacja dendrogramu dla sędziów"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    dendrogram(linkage_matrix, 
              labels=labels,
              orientation='top',
              leaf_font_size=10,
              ax=ax)
    
    ax.set_title('Hierarchiczne klasterowanie sędziów\n' +
                'na podstawie wzorców oceniania przez wszystkie etapy',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Odległość', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano dendrogram sędziów: {save_path}")


def visualize_kmeans_clusters(results_df, cluster_stats_df, save_path='participant_clusters.png'):
    """Wizualizacja klastrów K-means dla uczestników"""
    n_clusters = len(cluster_stats_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Rozkład klastrów
    ax1 = axes[0, 0]
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    colors = sns.color_palette("husl", n_clusters)
    ax1.bar(cluster_counts.index, cluster_counts.values, color=colors)
    ax1.set_xlabel('Numer klastra', fontsize=12)
    ax1.set_ylabel('Liczba uczestników', fontsize=12)
    ax1.set_title('Rozkład uczestników w klastrach', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Średnie oceny w klastrach
    ax2 = axes[0, 1]
    ax2.bar(cluster_stats_df['cluster'], cluster_stats_df['avg_mean_score'], color=colors)
    ax2.set_xlabel('Numer klastra', fontsize=12)
    ax2.set_ylabel('Średnia ocena', fontsize=12)
    ax2.set_title('Średnia ocena w klastrach', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Liczba etapów w klastrach
    ax3 = axes[1, 0]
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        ax3.scatter(cluster_data['n_stages'], 
                   cluster_data['mean_score'],
                   label=f'Klaster {cluster_id}',
                   alpha=0.6,
                   s=100,
                   color=colors[cluster_id])
    ax3.set_xlabel('Liczba etapów', fontsize=12)
    ax3.set_ylabel('Średnia ocena', fontsize=12)
    ax3.set_title('Liczba etapów vs średnia ocena', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Zmienność ocen w klastrach
    ax4 = axes[1, 1]
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        ax4.scatter(cluster_data['mean_score'], 
                   cluster_data['std_score'],
                   label=f'Klaster {cluster_id}',
                   alpha=0.6,
                   s=100,
                   color=colors[cluster_id])
    ax4.set_xlabel('Średnia ocena', fontsize=12)
    ax4.set_ylabel('Odchylenie standardowe', fontsize=12)
    ax4.set_title('Średnia vs zmienność ocen', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Analiza klastrów uczestników (K-means)\n' +
                'na podstawie profili ze wszystkich etapów',
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wizualizację klastrów: {save_path}")


def visualize_judge_clusters(results_df, save_path='judge_clusters.png'):
    """Wizualizacja klastrów sędziów"""
    n_clusters = results_df['cluster'].nunique()
    colors = sns.color_palette("husl", n_clusters)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Średnia ocena vs zmienność
    ax1 = axes[0]
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        ax1.scatter(cluster_data['mean_score'], 
                   cluster_data['std_score'],
                   label=f'Klaster {cluster_id}',
                   alpha=0.7,
                   s=200,
                   color=colors[cluster_id])
        
        # Dodaj etykiety sędziów
        for idx, row in cluster_data.iterrows():
            ax1.annotate(row['judge'], 
                        (row['mean_score'], row['std_score']),
                        fontsize=8,
                        alpha=0.7)
    
    ax1.set_xlabel('Średnia ocena', fontsize=12)
    ax1.set_ylabel('Odchylenie standardowe', fontsize=12)
    ax1.set_title('Klastry sędziów: średnia vs zmienność', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rozkład klastrów
    ax2 = axes[1]
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color=colors)
    ax2.set_xlabel('Numer klastra', fontsize=12)
    ax2.set_ylabel('Liczba sędziów', fontsize=12)
    ax2.set_title('Rozkład sędziów w klastrach', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Analiza klastrów sędziów (K-means)\n' +
                'na podstawie wzorców oceniania przez wszystkie etapy',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wizualizację klastrów sędziów: {save_path}")


def visualize_participant_progression(progression_df, save_path='participant_progression.png'):
    """
    Wizualizacja progresji uczestników przez etapy
    Pokazuje uczestników którzy przeszli przez wiele etapów
    """
    # Znajdź uczestników którzy przeszli przez >= 3 etapy
    participant_stage_counts = progression_df.groupby('Nr').size()
    multi_stage_participants = participant_stage_counts[participant_stage_counts >= 3].index
    
    if len(multi_stage_participants) == 0:
        print("Brak uczestników którzy przeszli przez >= 3 etapy")
        return
    
    # Ogranicz do top 15 (dla czytelności)
    # Wybierz tych z najwyższą średnią oceną w finale
    final_scores = progression_df[progression_df['stage'] == 'final']
    top_participants = final_scores.nlargest(15, 'mean_score')['Nr'].values
    
    # Filtruj do uczestników którzy są zarówno w top 15 jak i przeszli przez wiele etapów
    selected_participants = [nr for nr in top_participants if nr in multi_stage_participants]
    
    if len(selected_participants) == 0:
        # Jeśli żaden z top 15 nie przeszedł przez wiele etapów, weź po prostu pierwszych 15 z multi-stage
        selected_participants = multi_stage_participants[:15].tolist()
    
    filtered_df = progression_df[progression_df['Nr'].isin(selected_participants)]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    stage_labels = {'stage1': 'Etap I', 'stage2': 'Etap II', 
                   'stage3': 'Etap III', 'final': 'Finał'}
    
    # Rysuj linie dla każdego uczestnika
    for nr in selected_participants:
        participant_data = filtered_df[filtered_df['Nr'] == nr].sort_values('stage', 
                                      key=lambda x: x.map({s: i for i, s in enumerate(stage_order)}))
        
        if len(participant_data) >= 2:
            stages = [stage_labels.get(s, s) for s in participant_data['stage']]
            scores = participant_data['mean_score'].values
            
            label = f"{nr}: {participant_data.iloc[0]['imię']} {participant_data.iloc[0]['nazwisko']}"
            ax.plot(stages, scores, marker='o', linewidth=2, label=label, alpha=0.7)
    
    ax.set_xlabel('Etap', fontsize=12)
    ax.set_ylabel('Średnia ocena', fontsize=12)
    ax.set_title('Progresja uczestników przez etapy konkursu\n' +
                '(uczestników którzy przeszli przez 3+ etapów)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wizualizację progresji: {save_path}")


def visualize_stage_correlation_heatmap(analyzer, save_path='stage_correlation_heatmap.png'):
    """
    Heatmapa korelacji ocen sędziów między różnymi etapami
    Pokazuje czy sędziowie konsekwentnie oceniają uczestników w różnych etapach
    """
    profile_df = analyzer.build_judge_multistage_matrix()
    
    # Dla każdego sędziego, oblicz korelacje między etapami
    stage_order = ['stage1', 'stage2', 'stage3', 'final']
    
    # Przygotuj macierz korelacji dla każdego sędziego
    judge_stage_correlations = []
    
    for judge in analyzer.judge_columns:
        judge_row = profile_df.loc[judge]
        
        stage_scores = {}
        for stage in stage_order:
            # Znajdź kolumny dla tego sędziego i etapu
            stage_cols = [col for col in judge_row.index if col.endswith(f'_{stage}')]
            stage_scores[stage] = judge_row[stage_cols].dropna()
        
        # Oblicz korelacje między etapami (dla uczestników którzy byli w obu)
        corr_row = {'judge': judge}
        for i, stage1 in enumerate(stage_order):
            for stage2 in stage_order[i+1:]:
                if len(stage_scores[stage1]) > 0 and len(stage_scores[stage2]) > 0:
                    # Znajdź wspólnych uczestników
                    common_participants = set([col.split('_')[0] for col in stage_scores[stage1].index]) & \
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
        print("Brak danych do obliczenia korelacji między etapami")
        return
    
    corr_df = pd.DataFrame(judge_stage_correlations)
    corr_df = corr_df.set_index('judge')
    
    # Wizualizacja
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
               vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Korelacja'})
    
    ax.set_title('Korelacja ocen sędziów między etapami\n' +
                '(dla tych samych uczestników w różnych etapach)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Porównanie etapów', fontsize=12)
    ax.set_ylabel('Sędzia', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano heatmapę korelacji: {save_path}")


def run_multistage_analysis(data_files, output_dir='multistage_results'):
    """
    Główna funkcja uruchamiająca pełną analizę wieloetapową
    
    Args:
        data_files: dict z kluczami 'stage1', 'stage2', 'stage3', 'final'
        output_dir: katalog do zapisu wyników
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ANALIZA WIELOETAPOWA KONKURSU CHOPINOWSKIEGO")
    print("=" * 80)
    
    # Inicjalizacja analizatora
    analyzer = MultiStageClusteringAnalyzer(data_files)
    
    # 1. Hierarchiczne klasterowanie uczestników
    print("\n1. Hierarchiczne klasterowanie uczestników...")
    try:
        linkage_matrix, labels, profile_df, participant_info = \
            analyzer.cluster_participants_hierarchical(min_stages=2)
        if linkage_matrix is not None:
            visualize_participant_dendrogram(linkage_matrix, labels, 
                                            f'{output_dir}/participant_dendrogram.png')
        else:
            print("  Pominięto - za mało danych")
    except Exception as e:
        print(f"  Błąd: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Hierarchiczne klasterowanie sędziów
    print("\n2. Hierarchiczne klasterowanie sędziów...")
    try:
        linkage_matrix_judges, judge_labels, judge_profile_df = \
            analyzer.cluster_judges_hierarchical()
        if linkage_matrix_judges is not None:
            visualize_judge_dendrogram(linkage_matrix_judges, judge_labels,
                                       f'{output_dir}/judge_dendrogram.png')
        else:
            print("  Pominięto - za mało danych")
    except Exception as e:
        print(f"  Błąd: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. K-means klasterowanie uczestników
    print("\n3. K-means klasterowanie uczestników...")
    for n_clusters in [3, 4, 5]:
        try:
            results_df, cluster_stats_df, kmeans, scaler, profile_filled = \
                analyzer.kmeans_cluster_participants(n_clusters=n_clusters, min_stages=2)
            
            # Zapisz wyniki
            results_df.to_csv(f'{output_dir}/participant_clusters_k{n_clusters}.csv', 
                            index=False, encoding='utf-8-sig')
            cluster_stats_df.to_csv(f'{output_dir}/participant_cluster_stats_k{n_clusters}.csv',
                                   index=False, encoding='utf-8-sig')
            
            # Wizualizacja
            visualize_kmeans_clusters(results_df, cluster_stats_df,
                                    f'{output_dir}/participant_clusters_k{n_clusters}.png')
        except Exception as e:
            print(f"  Błąd dla k={n_clusters}: {e}")
    
    # 4. K-means klasterowanie sędziów
    print("\n4. K-means klasterowanie sędziów...")
    for n_clusters in [2, 3, 4]:
        try:
            judge_results_df, judge_kmeans, judge_profile_filled = \
                analyzer.kmeans_cluster_judges(n_clusters=n_clusters)
            
            judge_results_df.to_csv(f'{output_dir}/judge_clusters_k{n_clusters}.csv',
                                   index=False, encoding='utf-8-sig')
            
            visualize_judge_clusters(judge_results_df,
                                    f'{output_dir}/judge_clusters_k{n_clusters}.png')
        except Exception as e:
            print(f"  Błąd dla k={n_clusters}: {e}")
    
    # 5. Analiza progresji uczestników
    print("\n5. Analiza progresji uczestników...")
    progression_df = analyzer.analyze_participant_progression()
    progression_df.to_csv(f'{output_dir}/participant_progression.csv', 
                         index=False, encoding='utf-8-sig')
    visualize_participant_progression(progression_df,
                                     f'{output_dir}/participant_progression.png')
    
    # 6. Heatmapa korelacji między etapami
    print("\n6. Korelacja ocen sędziów między etapami...")
    try:
        visualize_stage_correlation_heatmap(analyzer,
                                          f'{output_dir}/stage_correlation_heatmap.png')
    except Exception as e:
        print(f"  Błąd w heatmapie korelacji: {e}")
    
    print("\n" + "=" * 80)
    print(f"Analiza zakończona! Wyniki zapisane w: {output_dir}")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    # Przykładowe użycie
    data_files = {
        'stage1': 'chopin_2025_stage1_by_judge.csv',
        'stage2': 'chopin_2025_stage2_by_judge.csv',
        'stage3': 'chopin_2025_stage3_by_judge.csv',
        'final': 'chopin_2025_final_by_judge.csv'
    }
    
    analyzer = run_multistage_analysis(data_files, output_dir='multistage_results')
