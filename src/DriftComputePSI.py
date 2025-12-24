import numpy as np
import pandas as pd

def calculate_psi(df_expected, df_actual, buckettype='quantiles', buckets=10, drift_thresholds=(0.1, 0.25)):
    """
    Calculate PSI for both numerical and categorical variables with a descriptive summary.
    
    Args:
        df_expected: pandas DataFrame, reference/original data
        df_actual: pandas DataFrame, new data to compare
        buckettype: 'bins' or 'quantiles' for numeric variables
        buckets: number of buckets for numeric variables
        drift_thresholds: tuple (moderate, significant) PSI thresholds
    
    Returns:
        psi_summary: pandas DataFrame with PSI values and drift interpretation
    """

    def psi_numeric(expected_array, actual_array, buckets):
        """PSI for numerical variables"""
        def scale_range(input_array, min_val, max_val):
            input_array = input_array - np.min(input_array)
            input_array = input_array / np.max(input_array) * (max_val - min_val)
            input_array = input_array + min_val
            return input_array

        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        else:
            breakpoints = np.percentile(expected_array, breakpoints)

        expected_fractions = np.histogram(expected_array, bins=breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, bins=breakpoints)[0] / len(actual_array)

        def sub_psi(e, a):
            e, a = max(e, 1e-4), max(a, 1e-4)
            return (e - a) * np.log(e / a)

        return np.sum([sub_psi(e, a) for e, a in zip(expected_fractions, actual_fractions)])

    def psi_categorical(expected_array, actual_array):
        """PSI for categorical variables"""
        expected_counts = pd.Series(expected_array).value_counts(normalize=True)
        actual_counts = pd.Series(actual_array).value_counts(normalize=True)

        categories = set(expected_counts.index).union(set(actual_counts.index))
        psi_value = 0
        for cat in categories:
            e = expected_counts.get(cat, 1e-4)
            a = actual_counts.get(cat, 1e-4)
            psi_value += (e - a) * np.log(e / a)
        return psi_value

    # Séparer les colonnes numériques et catégorielles
    numeric_cols = df_expected.select_dtypes(include=np.number).columns
    categorical_cols = df_expected.select_dtypes(exclude=np.number).columns

    psi_dict = {}

    # Calcul PSI pour numérique
    for col in numeric_cols:
        psi_dict[col] = psi_numeric(df_expected[col].values, df_actual[col].values, buckets)

    # Calcul PSI pour catégoriel
    for col in categorical_cols:
        psi_dict[col] = psi_categorical(df_expected[col].values, df_actual[col].values)

    # Résumé descriptif
    summary = []
    for col, psi_val in psi_dict.items():
        if psi_val < drift_thresholds[0]:
            drift = "No drift"
        elif psi_val < drift_thresholds[1]:
            drift = "Moderate drift"
        else:
            drift = "Significant drift"
        summary.append({"Variable": col, "PSI": psi_val, "Drift": drift})

    psi_summary = pd.DataFrame(summary).sort_values(by="PSI", ascending=False).reset_index(drop=True)
    return psi_summary
