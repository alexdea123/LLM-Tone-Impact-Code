import json
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st

def load_solution_data(base_results_dir='results'):
    """
    Recursively load JSON result files for each tone and return one DataFrame 
    with one row per solution (problem_id, tone, pass@1, lines_of_code, etc.).
    """
    all_rows = []
    tone_categories = ["reciprocity", "polite", "inspirational", 
                       "pressure", "neutral", "ingratiating", "insults"]

    for tone in tone_categories:
        json_path = os.path.join(base_results_dir, tone, "metrics_results.json")
        if not os.path.isfile(json_path):
            print(f"WARNING: Missing file for tone '{tone}': {json_path}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            tone_data = json.load(f)

        for problem_id, problem_info in tone_data.items():
            solutions_list = problem_info.get("solutions", [])
            for sol in solutions_list:
                row = {
                    "problem_id": sol["problem_id"],
                    "tone": sol["tone_category"],
                    "correct": 1.0 if sol["pass@1"] == 1.0 else 0.0,
                    "lines_of_code": sol.get("lines_of_code", None),
                    "comment_line_count": sol.get("comment_line_count", None),
                    "comment_char_count": sol.get("comment_char_count", None),
                    "comment_to_code_char_ratio": sol.get("comment_to_code_char_ratio", None),
                    "avg_identifier_length": sol.get("avg_identifier_length", None),
                    "avg_cyclomatic_complexity": sol.get("avg_cyclomatic_complexity", None),
                    "max_cyclomatic_complexity": sol.get("max_cyclomatic_complexity", None),
                    "pylint_score": sol.get("pylint_score", None),
                    "pylint_warnings": sol.get("pylint_warnings", None),
                    "pylint_conventions": sol.get("pylint_conventions", None),
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df["tone"] = df["tone"].astype("category")
    df["problem_id"] = df["problem_id"].astype("category")
    return df

def print_pairwise_comparisons(model, predictor="tone"):
    summary_df = model.summary2().tables[1]
    
    # Updated regex to match coefficients like:
    # "C(tone, Treatment(reference='neutral'))[T.ingratiating]"
    tone_coeffs = summary_df.loc[summary_df.index.str.contains(
        rf"C\({predictor}, Treatment\(reference='neutral'\)\)\[T\..+\]", regex=True)]
    print("Pairwise comparisons against the neutral baseline:")
    print(tone_coeffs)
    return tone_coeffs

def fit_and_test_logistic(df, response_col="correct", group_col="problem_id", predictor_col="tone"):
    """
    Fit a logistic regression model with tone (using neutral as reference) and group as fixed factors.
    Perform a likelihood ratio test and print pairwise comparisons (each tone vs neutral).
    """
    full_formula = f"{response_col} ~ C({predictor_col}, Treatment(reference='neutral')) + C({group_col})"
    full_model = smf.glm(full_formula, data=df, family=sm.families.Binomial()).fit()

    reduced_formula = f"{response_col} ~ C({group_col})"
    reduced_model = smf.glm(reduced_formula, data=df, family=sm.families.Binomial()).fit()

    lr_test_stat = 2 * (full_model.llf - reduced_model.llf)
    lr_test_df = full_model.df_model - reduced_model.df_model
    lr_test_p_value = st.chi2.sf(lr_test_stat, lr_test_df)

    print("\n============================")
    print("Logistic Regression for Correctness:")
    print(f"Likelihood Ratio Test Statistic = {lr_test_stat:.3f}, p-value = {lr_test_p_value:.5f}, df = {int(lr_test_df)}")
    print("============================\n")
    
    print_pairwise_comparisons(full_model, predictor=predictor_col)
    return full_model, reduced_model

def fit_and_test_linear(df, metric_col, group_col="problem_id", predictor_col="tone"):
    """
    Fit a linear regression model with tone (using neutral as reference) and group as fixed factors.
    Perform an ANOVA test and print pairwise comparisons (each tone vs neutral).
    """
    df_sub = df.dropna(subset=[metric_col])
    if df_sub.empty:
        print(f"[WARNING] No data for metric {metric_col}")
        return None, None

    full_formula = f"{metric_col} ~ C({predictor_col}, Treatment(reference='neutral')) + C({group_col})"
    full_model = smf.ols(full_formula, data=df_sub).fit()

    reduced_formula = f"{metric_col} ~ C({group_col})"
    reduced_model = smf.ols(reduced_formula, data=df_sub).fit()

    anova_res = sm.stats.anova_lm(reduced_model, full_model)

    print("\n============================")
    print(f"Linear Regression for Metric: {metric_col}")
    print("ANOVA Result:")
    print(anova_res)
    print("============================\n")
    
    print_pairwise_comparisons(full_model, predictor=predictor_col)
    return full_model, reduced_model

# Load Data
df_solutions = load_solution_data(base_results_dir='results')

# Run Logistic Regression for Correctness (RQ1)
fit_and_test_logistic(df_solutions)

# Define RQ2 Metrics (Stylistic Analysis)
rq2_metrics = [
    "lines_of_code",
    "comment_char_count",
    "comment_to_code_char_ratio",
    "avg_identifier_length",
]

# Run Linear Models for RQ2
for metric in rq2_metrics:
    fit_and_test_linear(df_solutions, metric_col=metric)

# Define RQ3 Metrics (Complexity & Quality)
rq3_metrics = [
    "avg_cyclomatic_complexity",
    "max_cyclomatic_complexity",
    "pylint_score",
    "pylint_warnings",
]

# Run Linear Models for RQ3
for metric in rq3_metrics:
    fit_and_test_linear(df_solutions, metric_col=metric)
