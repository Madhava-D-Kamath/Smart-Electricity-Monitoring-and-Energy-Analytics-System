"""
AI-Enabled Smart Electricity Monitoring System
Model Comparison and Evaluation Analysis

Purpose:
1. Compare existing random-split model evaluation results.
2. Compare session-based validation results.
3. Generate a combined comparison report.
4. Identify the best model using Macro F1 and Balanced Accuracy.
5. Generate comparison charts.
6. Preserve all original model files and evaluation reports.

This script DOES NOT train models.
"""

from pathlib import Path
import json
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"

RANDOM_RESULTS_FILE = RESULTS_DIR / "model_comparison.csv"

SESSION_RESULTS_FILE = (
    RESULTS_DIR
    / "session_validation"
    / "session_model_comparison.csv"
)

OUTPUT_DIR = RESULTS_DIR / "model_comparison_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. METRIC CONFIGURATION
# ============================================================

METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision Macro",
    "Recall Macro",
    "F1 Macro",
    "F1 Weighted",
]

METRIC_ALIASES = {
    "Accuracy": [
        "accuracy",
        "acc",
        "testaccuracy",
    ],

    "Balanced Accuracy": [
        "balancedaccuracy",
        "balancedacc",
    ],

    "Precision Macro": [
        "precisionmacro",
        "macroprecision",
        "precision_macro",
    ],

    "Recall Macro": [
        "recallmacro",
        "macrorecall",
        "recall_macro",
    ],

    "F1 Macro": [
        "f1macro",
        "macrof1",
        "f1_macro",
        "macro_f1",
    ],

    "F1 Weighted": [
        "f1weighted",
        "weightedf1",
        "f1_weighted",
        "weighted_f1",
    ],
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_column_name(name):
    """
    Convert column names into a consistent format.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


def find_model_column(df):
    """
    Identify the model-name column.
    """

    possible_names = [
        "model",
        "modelname",
        "classifier",
        "algorithm",
        "estimator",
    ]

    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for name in possible_names:

        if name in normalized_columns:
            return normalized_columns[name]

    raise ValueError(
        "Could not identify the model-name column. "
        f"Available columns: {list(df.columns)}"
    )


def find_metric_column(df, metric):
    """
    Find a metric column using supported aliases.
    """

    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
    }

    aliases = METRIC_ALIASES[metric]

    for alias in aliases:

        if alias in normalized_columns:
            return normalized_columns[alias]

    return None


def convert_metric_values(series):
    """
    Convert metric values into numeric decimal values.

    Example:
    99.5 becomes 0.995.
    0.995 remains 0.995.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid_values = values.dropna()

    if not valid_values.empty:

        if valid_values.max() > 1:

            values = values / 100

    return values


# ============================================================
# 4. LOAD AND STANDARDIZE RESULTS
# ============================================================

def load_comparison_file(file_path, evaluation_type):

    print("\n" + "=" * 65)

    print("READING:", file_path)

    print("=" * 65)

    if not file_path.exists():

        print("WARNING: File not found.")

        return None

    try:

        df = pd.read_csv(file_path)

    except Exception as error:

        print("ERROR READING FILE:", error)

        return None

    if df.empty:

        print("WARNING: Evaluation file is empty.")

        return None

    print("Original columns:", list(df.columns))

    # Identify model column

    model_column = find_model_column(df)

    result = pd.DataFrame()

    result["Model"] = (
        df[model_column]
        .astype(str)
        .str.strip()
    )

    result["Evaluation Type"] = evaluation_type

    # Extract available metrics

    found_metrics = []

    for metric in METRICS:

        source_column = find_metric_column(
            df,
            metric
        )

        if source_column is not None:

            result[metric] = convert_metric_values(
                df[source_column]
            )

            found_metrics.append(metric)

        else:

            result[metric] = float("nan")

    if not found_metrics:

        raise ValueError(
            f"No recognized evaluation metrics found in {file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    print("\nRecognized metrics:")

    for metric in found_metrics:

        print(" -", metric)

    print("\nLoaded models:")

    for model in result["Model"].tolist():

        print(" -", model)

    return result


# ============================================================
# 5. LOAD BOTH EVALUATION TYPES
# ============================================================

def load_all_results():

    all_results = []

    # Original random row split results

    random_results = load_comparison_file(
        RANDOM_RESULTS_FILE,
        "Random Split"
    )

    if random_results is not None:

        all_results.append(random_results)

    # Session-based validation results

    session_results = load_comparison_file(
        SESSION_RESULTS_FILE,
        "Session-Based Split"
    )

    if session_results is not None:

        all_results.append(session_results)

    if not all_results:

        print("\nERROR: No evaluation files were found.")

        print(
            "Run the appropriate evaluation scripts first."
        )

        sys.exit(1)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    return combined


# ============================================================
# 6. SAVE COMBINED COMPARISON
# ============================================================

def save_combined_results(combined):

    output_file = (
        OUTPUT_DIR
        / "combined_model_comparison.csv"
    )

    combined.to_csv(
        output_file,
        index=False
    )

    print("\nCombined comparison saved:")
    print(output_file)

    return output_file


# ============================================================
# 7. IDENTIFY BEST MODELS
# ============================================================

def identify_best_models(combined):

    print("\n")
    print("=" * 65)

    print("BEST MODEL ANALYSIS")

    print("=" * 65)

    best_results = []

    for evaluation_type in combined[
        "Evaluation Type"
    ].dropna().unique():

        subset = combined[
            combined["Evaluation Type"] == evaluation_type
        ].copy()

        print("\nEvaluation:", evaluation_type)

        for metric in [
            "F1 Macro",
            "Balanced Accuracy",
            "Accuracy",
        ]:

            valid = subset.dropna(
                subset=[metric]
            )

            if valid.empty:

                print(
                    f"{metric}: No valid values available."
                )

                continue

            best_index = valid[metric].idxmax()

            best_row = valid.loc[best_index]

            best_model = best_row["Model"]

            best_score = best_row[metric]

            print(
                f"Best {metric}: "
                f"{best_model} "
                f"({best_score:.4f})"
            )

            best_results.append({

                "Evaluation Type": evaluation_type,

                "Metric": metric,

                "Best Model": best_model,

                "Score": best_score,

            })

    best_df = pd.DataFrame(best_results)

    output_file = (
        OUTPUT_DIR
        / "best_model_summary.csv"
    )

    best_df.to_csv(
        output_file,
        index=False
    )

    print("\nBest model summary saved:")
    print(output_file)

    return best_df


# ============================================================
# 8. GENERATE COMPARISON CHARTS
# ============================================================

def generate_metric_charts(combined):

    print("\nGenerating comparison charts...")

    for metric in METRICS:

        chart_data = combined.dropna(
            subset=[metric]
        ).copy()

        if chart_data.empty:

            print(
                f"Skipping {metric}: no valid values."
            )

            continue

        plt.figure(figsize=(12, 7))

        for evaluation_type in chart_data[
            "Evaluation Type"
        ].unique():

            subset = chart_data[
                chart_data["Evaluation Type"]
                == evaluation_type
            ]

            plt.plot(
                subset["Model"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=evaluation_type
            )

        plt.title(
            f"Model Comparison - {metric}",
            fontsize=15
        )

        plt.xlabel(
            "Machine Learning Models"
        )

        plt.ylabel(
            metric
        )

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.ylim(0, 1.05)

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            metric.lower()
            .replace(" ", "_")
            + "_comparison.png"
        )

        output_file = OUTPUT_DIR / filename

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output_file)


# ============================================================
# 9. GENERATE MACRO F1 RANKING
# ============================================================

def generate_macro_f1_ranking(combined):

    ranking_data = combined.dropna(
        subset=["F1 Macro"]
    ).copy()

    if ranking_data.empty:

        print("\nMacro F1 ranking unavailable.")

        return

    ranking_data = ranking_data.sort_values(
        by=[
            "Evaluation Type",
            "F1 Macro"
        ],
        ascending=[
            True,
            False
        ]
    )

    ranking_data["Rank"] = (
        ranking_data.groupby("Evaluation Type")[
            "F1 Macro"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    output_file = (
        OUTPUT_DIR
        / "macro_f1_model_ranking.csv"
    )

    ranking_data.to_csv(
        output_file,
        index=False
    )

    print("\nMacro F1 ranking saved:")
    print(output_file)


# ============================================================
# 10. DISPLAY FINAL COMPARISON TABLE
# ============================================================

def display_final_table(combined):

    print("\n")
    print("=" * 90)

    print("FINAL MODEL COMPARISON")

    print("=" * 90)

    display_columns = [
        "Evaluation Type",
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "F1 Weighted",
    ]

    display_columns = [
        col for col in display_columns
        if col in combined.columns
    ]

    print(
        combined[display_columns]
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# 11. SAVE EXECUTION METADATA
# ============================================================

def save_metadata(combined):

    metadata = {

        "project": (
            "AI-Enabled Smart Electricity "
            "Monitoring and Energy Optimization System"
        ),

        "comparison_script": "14_compare_models.py",

        "evaluation_types": (
            combined["Evaluation Type"]
            .dropna()
            .unique()
            .tolist()
        ),

        "total_model_evaluations": int(
            len(combined)
        ),

        "models": (
            combined["Model"]
            .dropna()
            .unique()
            .tolist()
        ),

        "metrics": METRICS,

        "note": (
            "Random row-split and session-based "
            "validation results are reported separately. "
            "They use different evaluation methodologies "
            "and should not be treated as directly "
            "equivalent test results."
        ),

    }

    output_file = (
        OUTPUT_DIR
        / "comparison_metadata.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nMetadata saved:")
    print(output_file)


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "AI SMART ELECTRICITY MONITORING SYSTEM"
    )

    print(
        "MACHINE LEARNING MODEL COMPARISON"
    )

    print("=" * 70)

    print("\nThis script only analyzes existing results.")

    print("No model training will be performed.")

    print("Existing model files will not be modified.")

    # Load existing results

    combined = load_all_results()

    # Save combined comparison

    save_combined_results(combined)

    # Identify best models

    identify_best_models(combined)

    # Generate metric charts

    generate_metric_charts(combined)

    # Generate rankings

    generate_macro_f1_ranking(combined)

    # Display comparison table

    display_final_table(combined)

    # Save metadata

    save_metadata(combined)

    print("\n")

    print("=" * 70)

    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print("\nOutput directory:")

    print(OUTPUT_DIR)


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()