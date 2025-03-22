import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

def load_solution_data(base_results_dir='results'):
    """
    Recursively load JSON result files for each tone and return one DataFrame 
    with one row per solution.
    """
    all_rows = []
    tone_categories = ["reciprocity", "polite", "inspirational", 
                       "pressure", "neutral", "ingratiating", "insults"]

    for tone in tone_categories:
        json_path = os.path.join(base_results_dir, tone, "metrics_results.json")
        if not os.path.isfile(json_path):
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

def auto_decision(p_value, alpha=0.05):
    return "met" if p_value >= alpha else "violated"

def verify_rq1(df):
    # Fit logistic regression with tone and problem_id as fixed effects.
    formula = "correct ~ C(tone, Treatment(reference='neutral')) + C(problem_id)"
    logit_model = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
    
    # Extract problem_id coefficients (ignoring intercept) as a proxy for random intercepts.
    summary_df = logit_model.summary2().tables[1]
    prob_effects = summary_df.loc[summary_df.index.str.contains(r"C\(problem_id\)\[T\..+\]", regex=True)]
    
    if prob_effects.empty:
        print("RQ1: Not enough levels of problem_id to test the normality of random intercepts.")
        return

    coef_values = prob_effects["Coef."].values
    stat, p_value = shapiro(coef_values)
    decision = auto_decision(p_value)
    print(f"RQ1: Logistic model random intercepts assumption is {decision} (Shapiro–Wilk p-value = {p_value:.3f}).")

def verify_continuous_metric(df, metric, predictor_col="tone"):
    df_metric = df.dropna(subset=[metric])
    if df_metric.empty:
        print(f"{metric}: No data available.")
        return
    
    # Fit linear mixed-effects model with tone (neutral as reference) and problem_id as random intercept.
    formula = f"{metric} ~ C({predictor_col}, Treatment(reference='neutral'))"
    model = smf.mixedlm(formula, df_metric, groups=df_metric["problem_id"])
    result = model.fit()
    
    # Check residuals for normality.
    residuals = result.resid
    stat_res, p_value_res = shapiro(residuals)
    decision_res = auto_decision(p_value_res)
    
    # Check random effects for normality.
    random_effects = []
    for key, re in result.random_effects.items():
        if isinstance(re, dict):
            random_effects.append(list(re.values())[0])
        else:
            random_effects.append(re)
    random_effects = np.array(random_effects)
    stat_re, p_value_re = shapiro(random_effects)
    decision_re = auto_decision(p_value_re)
    
    print(f"{metric}: Residuals normality assumption is {decision_res} (p-value = {p_value_res:.3f}).")
    print(f"{metric}: Random effects normality assumption is {decision_re} (p-value = {p_value_re:.3f}).")

if __name__ == '__main__':
    df_solutions = load_solution_data(base_results_dir='results')
    
    # Verify assumptions for RQ1 (correctness with logistic model)
    verify_rq1(df_solutions)
    
    # Continuous metrics for RQ2 and RQ3.
    continuous_metrics = [
        "lines_of_code",
        "comment_char_count",
        "comment_to_code_char_ratio",
        "avg_identifier_length",
        "avg_cyclomatic_complexity",
        "max_cyclomatic_complexity",
        "pylint_score",
        "pylint_warnings"
    ]
    
    for metric in continuous_metrics:
        verify_continuous_metric(df_solutions, metric, predictor_col="tone")
