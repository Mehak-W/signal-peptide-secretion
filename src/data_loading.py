"""
Data loading utilities for signal peptide efficiency prediction.

Loads physicochemical features from the original xlsx file and
PLM embeddings from precomputed parquet files.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ---------------------------------------------------------------------------
# Grasso et al. validated physicochemical features (156 features)
# ---------------------------------------------------------------------------
GRASSO_FEATURES = [
    # N-region (amino-terminal region)
    'Turn_N', 'A_N', 'C_N', 'D_N', 'E_N', 'F_N', 'G_N', 'H_N', 'I_N',
    'L_N', 'N_N', 'P_N', 'Q_N', 'S_N', 'T_N', 'V_N', 'W_N', 'Y_N',
    'Length_N', 'InstabilityInd_N', 'Aromaticity_N', 'flexibility_N',
    'kytedoolittle_N', 'mfe_N',
    # H-region (hydrophobic core)
    'Turn_H', 'G_H', 'M_H', 'N_H', 'P_H', 'Q_H', 'S_H', 'T_H', 'W_H', 'Y_H',
    'Length_H', 'InstabilityInd_H', 'BomanInd_H', 'mfe_H',
    # C-region (carboxy-terminal)
    'Helix_C', 'Turn_C', 'Sheet_C', 'A_C', 'C_C', 'D_C', 'E_C', 'G_C',
    'I_C', 'L_C', 'M_C', 'N_C', 'P_C', 'Q_C', 'R_C', 'S_C', 'T_C',
    'V_C', 'W_C', 'Y_C', 'Length_C', 'pI_C', 'InstabilityInd_C',
    'AliphaticInd_C', 'ez_C', 'gravy_C', 'mfe_C', 'CAI_RSCU_C',
    # Ac-region (post-cleavage amino acids)
    'Turn_Ac', 'Sheet_Ac', 'A_Ac', 'D_Ac', 'E_Ac', 'F_Ac', 'G_Ac',
    'H_Ac', 'I_Ac', 'L_Ac', 'M_Ac', 'N_Ac', 'P_Ac', 'Q_Ac', 'R_Ac',
    'S_Ac', 'T_Ac', 'V_Ac', 'MW_Ac', 'pI_Ac', 'InstabilityInd_Ac',
    'BomanInd_Ac', 'ez_Ac', 'mfe_Ac', 'CAI_RSCU_Ac',
    # SP-region global features
    'Helix_SP', 'Turn_SP', 'D_SP', 'E_SP', 'F_SP', 'G_SP', 'H_SP',
    'L_SP', 'M_SP', 'N_SP', 'P_SP', 'Q_SP', 'S_SP', 'T_SP', 'W_SP', 'Y_SP',
    'Length_SP', 'Charge_SP', 'InstabilityInd_SP', 'flexibility_SP',
    'gravy_SP', 'mfe_SP', '-35_mfe_SP', 'amyQ_mfe_SP', 'CAI_RSCU_SP',
    # Cleavage site specificity (-3 and -1 positions)
    '-3_A', '-3_C', '-3_D', '-3_E', '-3_F', '-3_G', '-3_H', '-3_I', '-3_K',
    '-3_L', '-3_M', '-3_N', '-3_P', '-3_Q', '-3_R', '-3_S', '-3_T', '-3_V',
    '-3_W', '-3_Y', '-1_A', '-1_C', '-1_D', '-1_E', '-1_F', '-1_G', '-1_H',
    '-1_I', '-1_K', '-1_L', '-1_M', '-1_N', '-1_P', '-1_Q', '-1_R', '-1_S',
    '-1_T', '-1_V', '-1_W', '-1_Y',
]

assert len(GRASSO_FEATURES) == 156, f"Expected 156 features, got {len(GRASSO_FEATURES)}"

# Quality control thresholds from Grasso et al.
MIN_SP_LENGTH = 10
MAX_SP_LENGTH = 40
MIN_WA = 1.0
MAX_WA = 10.0
SKEWNESS_THRESHOLD = 1.0


def _default_data_dir():
    return Path(__file__).resolve().parent.parent / 'data'


def load_physicochemical(data_dir=None):
    """
    Load physicochemical features from the xlsx file (single source of truth).

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
        Plus a dict with metadata: feature_names, n_train, n_test
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    xlsx_path = data_dir / 'sb2c00328_si_011.xlsx'

    df = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')

    # Quality filters
    mask = (
        df['WA'].notna() &
        df['SP_aa'].notna() &
        (df['SP_aa'].str.len() >= MIN_SP_LENGTH) &
        (df['SP_aa'].str.len() <= MAX_SP_LENGTH) &
        (df['WA'] >= MIN_WA) &
        (df['WA'] <= MAX_WA) &
        df['Set'].notna()
    )
    df = df[mask].copy()

    # Verify all 156 features exist
    available = [f for f in GRASSO_FEATURES if f in df.columns]
    missing = [f for f in GRASSO_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in xlsx: {missing}")

    # Split by original Grasso assignments
    train_mask = df['Set'] == 'Train'
    test_mask = df['Set'] == 'Test'

    X_train = df.loc[train_mask, available].fillna(0).values.astype(np.float64)
    X_test = df.loc[test_mask, available].fillna(0).values.astype(np.float64)
    y_train = df.loc[train_mask, 'WA'].values.astype(np.float64)
    y_test = df.loc[test_mask, 'WA'].values.astype(np.float64)

    meta = {
        'feature_names': available,
        'n_features': len(available),
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
    return X_train, X_test, y_train, y_test, meta


def load_plm_embeddings(model_name, data_dir=None):
    """
    Load PLM embeddings from parquet files.

    Args:
        model_name: one of 'esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M'
        data_dir: path to data directory

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
        Plus a dict with metadata
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()

    train_path = data_dir / f'trainAA_{model_name}.parquet'
    test_path = data_dir / f'testAA_{model_name}.parquet'

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    X_train = np.stack(df_train['embedding'].values)
    X_test = np.stack(df_test['embedding'].values)
    y_train = df_train['WA'].values.astype(np.float64)
    y_test = df_test['WA'].values.astype(np.float64)

    meta = {
        'model_name': model_name,
        'n_features': X_train.shape[1],
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
    return X_train, X_test, y_train, y_test, meta


def preprocess(X_train, X_test, y_train, y_test, scale_features=True, log_transform_target='auto'):
    """
    Standard preprocessing pipeline.

    Args:
        scale_features: whether to StandardScale features
        log_transform_target: True, False, or 'auto' (apply if |skew| > 1.0)

    Returns:
        X_train_p, X_test_p, y_train_p, y_test_p, info dict
    """
    info = {'scaled': False, 'log_transformed': False, 'scaler': None}

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        info['scaled'] = True
        info['scaler'] = scaler

    if log_transform_target == 'auto':
        skewness = stats.skew(y_train)
        do_log = abs(skewness) > SKEWNESS_THRESHOLD
        info['skewness'] = float(skewness)
    else:
        do_log = bool(log_transform_target)

    if do_log:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        info['log_transformed'] = True

    return X_train, X_test, y_train, y_test, info


def inverse_transform_target(y, log_transformed):
    """Undo log1p if it was applied."""
    if log_transformed:
        return np.expm1(y)
    return y


# Feature type registry for iteration
FEATURE_TYPES = {
    'PhysChem': {
        'loader': load_physicochemical,
        'args': {},
        'dim': 156,
    },
    'ESM2-650M': {
        'loader': load_plm_embeddings,
        'args': {'model_name': 'esm2-650M'},
        'dim': 1280,
    },
    'ESM2-3B': {
        'loader': load_plm_embeddings,
        'args': {'model_name': 'esm2-3B'},
        'dim': 2560,
    },
    'Ginkgo-AA0': {
        'loader': load_plm_embeddings,
        'args': {'model_name': 'ginkgo-AA0-650M'},
        'dim': 1280,
    },
}


def load_all_feature_types(data_dir=None):
    """Load all 4 feature types, return dict of {name: (X_train, X_test, y_train, y_test, meta)}."""
    results = {}
    for name, spec in FEATURE_TYPES.items():
        kwargs = dict(spec['args'])
        if data_dir:
            kwargs['data_dir'] = data_dir
        results[name] = spec['loader'](**kwargs)
    return results


# ---------------------------------------------------------------------------
# Bin probability loading (for vector regression)
# ---------------------------------------------------------------------------
BIN_COLUMNS = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]


def load_plm_with_bins(model_name, data_dir=None):
    """
    Load PLM embeddings with bin probability distributions.

    Args:
        model_name: one of 'esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M'
        data_dir: path to data directory

    Returns:
        X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins, meta
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()

    train_path = data_dir / f'trainAA_{model_name}.parquet'
    test_path = data_dir / f'testAA_{model_name}.parquet'

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Drop rows with NaN bin probabilities (some samples have WA but no bin data)
    train_valid = df_train[BIN_COLUMNS].notna().all(axis=1)
    test_valid = df_test[BIN_COLUMNS].notna().all(axis=1)
    n_train_dropped = (~train_valid).sum()
    n_test_dropped = (~test_valid).sum()
    df_train = df_train[train_valid].reset_index(drop=True)
    df_test = df_test[test_valid].reset_index(drop=True)

    X_train = np.stack(df_train['embedding'].values)
    X_test = np.stack(df_test['embedding'].values)
    y_train_wa = df_train['WA'].values.astype(np.float64)
    y_test_wa = df_test['WA'].values.astype(np.float64)
    y_train_bins = df_train[BIN_COLUMNS].values.astype(np.float64)
    y_test_bins = df_test[BIN_COLUMNS].values.astype(np.float64)

    meta = {
        'model_name': model_name,
        'n_features': X_train.shape[1],
        'n_train': len(y_train_wa),
        'n_test': len(y_test_wa),
        'n_bins': 10,
        'n_train_dropped_nan_bins': int(n_train_dropped),
        'n_test_dropped_nan_bins': int(n_test_dropped),
    }
    return X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins, meta


# ---------------------------------------------------------------------------
# Design library loading (for design task evaluation)
# ---------------------------------------------------------------------------

def load_design_library(data_dir=None):
    """
    Load design variants from xlsx with WA values.

    Returns design variants NOT in train/test sets (Set is NaN/blank).
    Filters out rows with NaN WA values.

    Returns:
        DataFrame with columns: SP_aa, WA, gene, Library
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    xlsx_path = data_dir / 'sb2c00328_si_011.xlsx'

    df = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')

    # Design variants: Set is NaN/blank (not Train, not Test)
    design_mask = df['Set'].isna()
    df_design = df[design_mask].copy()

    # Filter out rows with NaN WA
    df_design = df_design[df_design['WA'].notna()].copy()

    # Keep relevant columns
    keep_cols = ['SP_aa', 'WA']
    if 'gene' in df_design.columns:
        keep_cols.append('gene')
    if 'Library' in df_design.columns:
        keep_cols.append('Library')

    df_design = df_design[keep_cols].copy()
    df_design = df_design.reset_index(drop=True)

    return df_design


def load_wt_sequences(data_dir=None):
    """
    Load wild-type sequences from the 'WT sequences' sheet of the xlsx.

    Returns:
        DataFrame with WT sequence info including gene and WA if available
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    xlsx_path = data_dir / 'sb2c00328_si_011.xlsx'

    df_wt = pd.read_excel(xlsx_path, sheet_name='WT sequences')
    return df_wt


def load_design_embeddings(model_name='esm2-650M', data_dir=None):
    """
    Load precomputed PLM embeddings for design variants.

    Returns:
        DataFrame with columns: sequence, embedding, WA
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    path = data_dir / f'design_{model_name}_embeddings.parquet'
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Unified dataset loading (standardized schema from Script 16)
# ---------------------------------------------------------------------------

def load_unified(dataset_source=None, set_filter=None, data_dir=None):
    """
    Load from the unified parquet schema (produced by Script 16).

    Args:
        dataset_source: 'grasso', 'wu', 'xue', 'zhang_p43', 'zhang_pglvm', or None for all
        set_filter: 'train', 'test', 'design', 'external', or None for all
        data_dir: path to data directory (default: data/)

    Returns:
        DataFrame with standardized columns:
        sequence, embedding, WA, BIN01-BIN10, gene, set, dataset_source
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    unified_dir = data_dir / 'unified'

    if dataset_source is not None:
        # Load per-dataset file
        if dataset_source == 'grasso' and set_filter in ('train', 'test', 'design'):
            path = unified_dir / f'grasso_{set_filter}.parquet'
        elif dataset_source == 'grasso':
            # Combine all grasso splits
            parts = []
            for split in ('train', 'test', 'design'):
                p = unified_dir / f'grasso_{split}.parquet'
                if p.exists():
                    parts.append(pd.read_parquet(p))
            df = pd.concat(parts, ignore_index=True)
            if set_filter:
                df = df[df['set'] == set_filter]
            return df
        else:
            path = unified_dir / f'{dataset_source}.parquet'
        df = pd.read_parquet(path)
    else:
        path = unified_dir / 'all_datasets_esm2_650M.parquet'
        df = pd.read_parquet(path)

    if set_filter is not None:
        df = df[df['set'] == set_filter].reset_index(drop=True)

    return df
