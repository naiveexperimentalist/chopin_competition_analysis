# 19th International Fryderyk Chopin Piano Competition Analysis

## How to Run

### Basic Usage
```bash
python run_full_analysis.py
```

### Select Specific Analyses
```bash
python run_full_analysis.py --analyses AMN
```

Available analysis codes:
- `A` - Advanced analysis and general visualisation
- `N` - Normalisation impact analysis  
- `M` - Multistage clustering and advanced visualisation
- `C` - Controversy and statistical analysis
- `S` - Statistical visualisation
- `B` - Bootstrap-based final score stability analysis
- `P` - Score-perturbation-based final score stability analysis

Default: `ALL` (runs everything)

### Examples
```bash
# Run only bootstrap and perturbation analyses
python run_full_analysis.py --analyses BP

# Run advanced analysis, normalisation, and multistage
python run_full_analysis.py --analyses ANM

# Run everything (same as default)
python run_full_analysis.py --analyses ALL
```

### Custom Input Files
```bash
python run_full_analysis.py --stage1 my_stage1.csv --stage2 my_stage2.csv --stage3 my_stage3.csv --final my_final.csv
```

### Custom Output Directory
```bash
python run_full_analysis.py --output my_results_folder
```
