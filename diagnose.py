#!/usr/bin/env python3
"""
Skrypt diagnostyczny do testowania analizy konkursu Chopinowskiego
Pomaga zidentyfikować problemy z danymi
"""

import pandas as pd
import numpy as np
import sys
import os

def diagnose_data():
    """Diagnozuje problemy z plikami danych"""
    
    print("=" * 60)
    print("DIAGNOSTYKA DANYCH KONKURSU CHOPINOWSKIEGO")
    print("=" * 60)
    
    # Sprawdź pliki
    files = [
        'chopin_2025_stage1_by_judge.csv',
        'chopin_2025_stage2_by_judge.csv',
        'chopin_2025_stage3_by_judge.csv',
        'chopin_2025_final_by_judge.csv'
    ]
    
    print("\n1. Sprawdzanie plików:")
    for f in files:
        if os.path.exists(f):
            print(f"   ✓ {f} - znaleziony")
            
            # Wczytaj i sprawdź podstawowe info
            try:
                df = pd.read_csv(f)
                print(f"     - Liczba wierszy: {len(df)}")
                print(f"     - Liczba kolumn: {len(df.columns)}")
                
                # Sprawdź podstawowe kolumny
                required = ['Nr', 'imię', 'nazwisko']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    print(f"     ⚠️ Brakujące kolumny: {missing}")
                
                # Sprawdź sędziów
                judge_cols = [col for col in df.columns if col not in ['Nr', 'imię', 'nazwisko']]
                print(f"     - Liczba sędziów: {len(judge_cols)}")
                
                # Sprawdź wartości
                for judge in judge_cols[:3]:  # Sprawdź pierwsze 3 kolumny sędziów
                    values = df[judge].dropna()
                    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                    non_numeric = values[~values.isin(numeric_values.astype(str))]
                    
                    if len(non_numeric) > 0:
                        unique_non_numeric = non_numeric.unique()
                        print(f"     - {judge}: {len(numeric_values)} ocen numerycznych, "
                              f"{len(non_numeric)} innych wartości {list(unique_non_numeric)[:5]}")
                
            except Exception as e:
                print(f"     ❌ Błąd przy wczytywaniu: {e}")
        else:
            print(f"   ❌ {f} - NIE ZNALEZIONY")
    
    print("\n2. Test podstawowego przetwarzania:")
    try:
        from chopin_data_processor import ChopinCompetitionProcessor
        
        processor = ChopinCompetitionProcessor()
        processor.load_data(files[0], files[1], files[2], files[3])
        
        print("   ✓ Dane wczytane pomyślnie")
        print(f"   - Zidentyfikowano {len(processor.judge_columns)} sędziów")
        
        # Test obliczeń
        processor.calculate_all_stages()
        print("   ✓ Obliczenia podstawowe zakończone")
        
        # Sprawdź korekty
        total_corrections = 0
        for stage, df in processor.corrected_data.items():
            corrections = df['corrections_made'].sum()
            total_corrections += corrections
            print(f"   - {stage}: {corrections} korekt ocen")
        
        if total_corrections == 0:
            print("   ⚠️ Brak korekt - sprawdź czy algorytm działa poprawnie")
        
        # Test wyników skumulowanych
        processor.calculate_cumulative_scores()
        print("   ✓ Wyniki skumulowane obliczone")
        
        for cum_name, df in processor.cumulative_scores.items():
            print(f"   - {cum_name}: {len(df)} uczestników")
            
    except ImportError:
        print("   ❌ Nie można zaimportować modułu chopin_data_processor")
    except Exception as e:
        print(f"   ❌ Błąd: {e}")
    
    print("\n3. Test zaawansowanych analiz:")
    try:
        from chopin_advanced_analyzer import ChopinAdvancedAnalyzer
        
        analyzer = ChopinAdvancedAnalyzer(processor)
        
        # Test wykorzystania skali
        scale_usage = analyzer.analyze_scale_usage()
        print(f"   ✓ Analiza skali: średnie wykorzystanie {scale_usage['scale_coverage'].mean():.1f}%")
        
        # Test tendencji
        tendencies = analyzer.analyze_judge_tendencies()
        harsh_count = len(tendencies[tendencies['overall_harshness'] < -1])
        lenient_count = len(tendencies[tendencies['overall_harshness'] > 1])
        print(f"   ✓ Tendencje: {harsh_count} surowych, {lenient_count} łagodnych sędziów")
        
        # Test sojuszy
        correlation_matrix, alliances = analyzer.analyze_judge_alliances(threshold=0.5)
        print(f"   ✓ Sojusze: znaleziono {len(alliances)} par z korelacją > 0.5")
        
        if len(alliances) == 0:
            print("   ⚠️ Brak silnych korelacji między sędziami - może to być normalne")
        
    except Exception as e:
        print(f"   ❌ Błąd w analizach: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTYKA ZAKOŃCZONA")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_data()
