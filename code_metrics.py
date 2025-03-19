import ast
import io
import sys
import os
import re
import tempfile
import json
import statistics
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import lizard
import radon.raw as radon_raw
from pylint import lint
from pylint.reporters.text import TextReporter


# Add a function to convert NumPy types to Python native types for JSON serialization
def convert_to_python_types(obj):
    """Convert NumPy data types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_identifier_metrics(code: str) -> Dict[str, Any]:
    """Extract metrics related to identifiers (variable/function names)."""
    try:
        tree = ast.parse(code)
        
        # Extract all names from the AST
        variable_names = []
        function_names = []
        
        # Find all variable assignments and function definitions
        for node in ast.walk(tree):
            # Variables (Name nodes that are targets of assignments)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variable_names.append(target.id)
            # Function definitions
            elif isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        
        # Calculate metrics
        all_identifiers = variable_names + function_names
        
        if all_identifiers:
            avg_identifier_length = sum(len(name) for name in all_identifiers) / len(all_identifiers)
            max_identifier_length = max(len(name) for name in all_identifiers) if all_identifiers else 0
            min_identifier_length = min(len(name) for name in all_identifiers) if all_identifiers else 0
        else:
            avg_identifier_length = 0
            max_identifier_length = 0
            min_identifier_length = 0
            
        return {
            'avg_identifier_length': avg_identifier_length,
            'max_identifier_length': max_identifier_length,
            'min_identifier_length': min_identifier_length,
            'variable_count': len(variable_names),
            'function_name_count': len(function_names),
            'total_identifiers': len(all_identifiers)
        }
    except Exception as e:
        return {
            'avg_identifier_length': 0,
            'max_identifier_length': 0,
            'min_identifier_length': 0,
            'variable_count': 0,
            'function_name_count': 0,
            'total_identifiers': 0,
            'identifier_error': str(e)
        }


def get_comment_metrics(code: str) -> Dict[str, Any]:
    """Extract detailed metrics about comments and docstrings."""
    lines = code.splitlines()
    
    # Count inline and block comments
    comment_lines = [line.strip() for line in lines if line.strip().startswith('#')]
    comment_chars = sum(len(line) for line in comment_lines)
    
    # Extract docstrings using AST
    try:
        tree = ast.parse(code)
        docstrings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append(docstring)
        
        docstring_chars = sum(len(ds) for ds in docstrings)
        
        # Calculate total characters in code (excluding comments and docstrings)
        # First, remove all comment lines
        code_without_comments = '\n'.join(line for line in lines if not line.strip().startswith('#'))
        
        # Then, try to replace docstrings with empty strings (approximation)
        # This is imperfect but gives a reasonable estimate
        for ds in docstrings:
            # Escape special regex characters in docstring
            escaped_ds = re.escape(ds)
            # Replace docstring with empty string (accounting for quotes)
            code_without_comments = re.sub(f'[\'"]{{3}}{escaped_ds}[\'"]{{3}}', '', code_without_comments)
        
        # Total characters in actual code (approximation)
        code_chars = len(code_without_comments)
        
        # Total characters in all comments (including docstrings)
        total_comment_chars = comment_chars + docstring_chars
        
        return {
            'comment_line_count': len(comment_lines),
            'docstring_count': len(docstrings),
            'comment_char_count': comment_chars,
            'docstring_char_count': docstring_chars,
            'total_comment_char_count': total_comment_chars,
            'code_char_count': code_chars,
            'comment_to_code_char_ratio': total_comment_chars / code_chars if code_chars > 0 else 0
        }
    except Exception as e:
        return {
            'comment_line_count': len(comment_lines),
            'comment_char_count': comment_chars,
            'docstring_count': 0,
            'docstring_char_count': 0,
            'total_comment_char_count': comment_chars,
            'code_char_count': 0,
            'comment_to_code_char_ratio': 0,
            'comment_error': str(e)
        }


def calculate_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate code quality metrics for a given Python code using existing packages.
    
    Args:
        code: String containing Python code
        
    Returns:
        Dictionary containing various code quality metrics grouped by research question
    """
    metrics = {
        'basic': {
            'lines_of_code': len(code.splitlines()),
            'char_count': len(code)
        },
        'RQ1': {},  # Correctness metrics
        'RQ2': {},  # Stylistic attributes
        'RQ3': {}   # Code quality metrics
    }
    
    try:
        # Get identifier metrics (for RQ2)
        identifier_metrics = get_identifier_metrics(code)
        metrics['RQ2'].update(identifier_metrics)
        
        # Get comment metrics (for RQ2)
        comment_metrics = get_comment_metrics(code)
        metrics['RQ2'].update(comment_metrics)
        
        # Use lizard for complexity metrics (for RQ3)
        analysis = lizard.analyze_file.analyze_source_code("temp.py", code)
        
        # Extract function-level metrics
        if analysis.function_list:
            metrics['RQ3']['function_count'] = len(analysis.function_list)
            metrics['RQ3']['avg_cyclomatic_complexity'] = np.mean([f.cyclomatic_complexity for f in analysis.function_list])
            metrics['RQ3']['max_cyclomatic_complexity'] = max([f.cyclomatic_complexity for f in analysis.function_list])
            metrics['RQ3']['total_cyclomatic_complexity'] = sum([f.cyclomatic_complexity for f in analysis.function_list])
        else:
            metrics['RQ3']['function_count'] = 0
            metrics['RQ3']['avg_cyclomatic_complexity'] = 0
            metrics['RQ3']['max_cyclomatic_complexity'] = 0
            metrics['RQ3']['total_cyclomatic_complexity'] = 0
        
        # Parse AST for other metrics
        tree = ast.parse(code)
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        metrics['RQ3']['class_count'] = class_count
        
        # Radon - Raw metrics for code size statistics (for RQ3)
        try:
            raw_metrics = radon_raw.analyze(code)
            metrics['RQ3']['logical_loc'] = raw_metrics.lloc
            metrics['RQ3']['source_loc'] = raw_metrics.sloc
        except Exception:
            metrics['RQ3']['logical_loc'] = 0
            metrics['RQ3']['source_loc'] = 0
        
        # Run pylint for code quality score (for RQ3)
        pylint_metrics = get_pylint_metrics(code)
        metrics['RQ3'].update(pylint_metrics)
        
        # Add raw LOC for RQ3
        metrics['RQ3']['raw_loc'] = metrics['basic']['lines_of_code']
        
    except SyntaxError as e:
        metrics['error'] = {'syntax_error': True, 'error_message': str(e)}
    except Exception as e:
        metrics['error'] = {'error': str(e)}
    
    # Flatten the metrics dictionary for backward compatibility
    flat_metrics = {}
    flat_metrics.update(metrics['basic'])
    for category, category_metrics in metrics.items():
        if isinstance(category_metrics, dict):
            flat_metrics.update(category_metrics)
    
    return flat_metrics


def get_pylint_metrics(code: str) -> Dict[str, Any]:
    """Run pylint on code and extract metrics."""
    pylint_output = io.StringIO()
    reporter = TextReporter(pylint_output)
    
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        # Suppress stdout during pylint run
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Run pylint with more checks enabled to get meaningful metrics
        lint.Run([
            '--disable=all',  # First disable all
            # Re-enable categories that are most relevant for code quality assessment
            '--enable=C,E,W,R',  # Enable Convention, Error, Warning, Refactor categories
            '--reports=y',  # Enable reports to get statistics
            tmp_path
        ], reporter=reporter, exit=False)
        
        # Restore stdout
        sys.stdout = original_stdout
        pylint_text = pylint_output.getvalue()
        
        # Extract score
        score_match = re.search(r'Your code has been rated at (-?\d+\.\d+)/10', pylint_text)
        score = float(score_match.group(1)) if score_match else 0.0
        
        # Count different types of issues more accurately
        # Count by message type patterns in the output
        error_count = len(re.findall(r': E\d{4}:', pylint_text))
        warning_count = len(re.findall(r': W\d{4}:', pylint_text))
        convention_count = len(re.findall(r': C\d{4}:', pylint_text))
        refactor_count = len(re.findall(r': R\d{4}:', pylint_text))
        
        # Calculate total issues
        total_issues = error_count + warning_count + convention_count + refactor_count
        
        return {
            'pylint_score': score,
            'pylint_errors': error_count,
            'pylint_warnings': warning_count,
            'pylint_conventions': convention_count,
            'pylint_refactors': refactor_count,
            'pylint_issue_count': total_issues
        }
    except Exception as e:
        return {'pylint_error': str(e), 'pylint_score': 0.0}
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def process_eval_all_json(file_path: str) -> Dict[str, Any]:
    """Process eval_all.json file and calculate metrics for code submissions."""
    # Normalize path to use forward slashes for consistent regex matching
    normalized_path = file_path.replace('\\', '/')

    # Extract tone category and model from path
    tone_match = re.search(r'/([^/]+)/Scenario\.codegeneration', normalized_path)
    print('tone_match:', tone_match)
    model_match = re.search(r'/([^/]+)/[^/]+/Scenario\.codegeneration', normalized_path)
    
    tone_category = tone_match.group(1) if tone_match else "unknown"
    model_name = model_match.group(1) if model_match else "unknown"
    
    print(f"Processing file with tone: {tone_category}, model: {model_name}")
    
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    pass_at_1_count = 0
    pass_at_5_count = 0    # new counter for pass@5
    pass_at_10_count = 0   # new counter for pass@10
    total_problems = 0
    
    for problem_idx, problem in enumerate(data):
        problem_id = problem.get('question_id', f"problem_{problem_idx}")
        problem_metrics = []
        
        # Track pass@1 if available
        pass_at_1 = problem.get('pass@1', None)
        if pass_at_1 is not None:
            pass_at_1_count += pass_at_1
            total_problems += 1
        
        # Track pass@5 and pass@10, similar to pass@1
        pass_at_5 = problem.get('pass@5', None)
        if pass_at_5 is not None:
            pass_at_5_count += pass_at_5
        pass_at_10 = problem.get('pass@10', None)
        if pass_at_10 is not None:
            pass_at_10_count += pass_at_10

        # Process each solution
        for solution_idx, code_solution in enumerate(problem.get('code_list', [])):
            if not code_solution.strip():
                continue
                
            # Calculate metrics
            metrics = calculate_metrics(code_solution)
            
            # Add metadata including pass@ values from the problem
            metadata = {
                'problem_id': problem.get('question_id', f"problem_{problem_idx}"),
                'problem_title': problem.get('question_title', f"problem_{problem_idx}"),
                'difficulty': problem.get('difficulty', "unknown"),
                'platform': problem.get('platform', "unknown"),
                'solution_idx': solution_idx,
                'code': code_solution,
                'graded': problem.get('graded_list', [])[solution_idx] 
                         if solution_idx < len(problem.get('graded_list', [])) else None,
                'tone_category': tone_category,
                'model_name': model_name,
                'pass@1': problem.get('pass@1', None),  # added pass@1 to metadata
                'pass@5': problem.get('pass@5', None),  # added pass@5 to metadata
                'pass@10': problem.get('pass@10', None)  # added pass@10 to metadata
            }
            
            # Merge metric data with metadata
            metrics.update(metadata)
            
            # Extract error info if available
            if 'metadata' in problem and solution_idx < len(problem['metadata']):
                try:
                    error_data = problem['metadata'][solution_idx]
                    if isinstance(error_data, str):
                        error_data = json.loads(error_data)
                    
                    if isinstance(error_data, dict):
                        metrics['error_code'] = error_data.get('error_code')
                        metrics['error_message'] = error_data.get('error_message')
                        metrics['execution_time'] = error_data.get('execution time')
                except (json.JSONDecodeError, TypeError):
                    pass
            
            problem_metrics.append(metrics)
            
        if problem_metrics:
            results[problem_id] = problem_metrics
    
    return {
        "tone_category": tone_category,
        "model_name": model_name,
        "results": results,
        "pass_at_1_count": pass_at_1_count,
        "pass_at_5_count": pass_at_5_count,    # returning pass@5 count
        "pass_at_10_count": pass_at_10_count,   # returning pass@10 count
        "total_problems": total_problems
    }


def aggregate_metrics_by_rq(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics grouped by research question."""
    rq_metrics = {
        "RQ1": {  # Correctness metrics
            "metrics": ["pass@1", "pass@5", "pass@10"],
            "data": {}
        },
        "RQ2": {  # Stylistic attributes
            "metrics": [
                "comment_line_count", "docstring_count", "comment_char_count", 
                "docstring_char_count", "total_comment_char_count", "comment_to_code_char_ratio",
                "avg_identifier_length", "pylint_conventions", "char_count"
            ],
            "data": {}
        },
        "RQ3": {  # Code quality
            "metrics": [
                "avg_cyclomatic_complexity", "max_cyclomatic_complexity", "total_cyclomatic_complexity",
                "logical_loc", "source_loc", "raw_loc", "pylint_score", 
                "pylint_errors", "pylint_warnings", "pylint_issue_count"
            ],
            "data": {}
        }
    }
    
    # Get a count of all problems by tone
    problem_counts = df.groupby('tone_category')['problem_id'].nunique()
    
    # Calculate pass rates with improved handling of missing solutions
    pass_rates = df.groupby(['tone_category', 'problem_id']).apply(
        lambda x: pd.Series({
            'first_solution_passed': x.iloc[0]['graded'] if len(x) > 0 else False,
            'pass_at_5': any(x.iloc[:5]['graded']) if len(x) >= 5 else any(x['graded']) if len(x) > 0 else False,
            'pass_at_10': any(x.iloc[:10]['graded']) if len(x) >= 10 else any(x['graded']) if len(x) > 0 else False,
            'solution_count': len(x),
            'overall_pass_rate': x['graded'].mean() if len(x) > 0 else 0
        })
    ).reset_index()
    
    # Aggregate metrics by tone for each research question
    for rq, rq_info in rq_metrics.items():
        metrics_list = rq_info["metrics"]
        rq_data = {}
        
        # For RQ1, use the pass rates we calculated
        if rq == "RQ1":
            tone_performance = pass_rates.groupby('tone_category').agg(
                pass_at_1_rate=('first_solution_passed', 'mean'),
                pass_at_5_rate=('pass_at_5', 'mean'),
                pass_at_10_rate=('pass_at_10', 'mean'),
                overall_pass_rate=('overall_pass_rate', 'mean'),
                avg_solution_count=('solution_count', 'mean'),
                total_solution_count=('solution_count', 'sum')
            ).reset_index()
            
            for _, row in tone_performance.iterrows():
                tone = row['tone_category']
                total_problems = problem_counts.get(tone, 0)
                
                rq_data[tone] = {
                    "pass@1": row['pass_at_1_rate'],
                    "pass@5": row['pass_at_5_rate'],
                    "pass@10": row['pass_at_10_rate'],
                    "overall_pass_rate": row['overall_pass_rate'],
                    "problem_count": total_problems,  # Use the full problem count
                    "avg_solutions_per_problem": row['avg_solution_count'],
                    "total_solutions": row['total_solution_count']
                }
        else:
            # For RQ2 and RQ3, get mean/variance of each metric by tone
            for tone, tone_df in df.groupby('tone_category'):
                tone_metrics = {}
                
                for metric in metrics_list:
                    if metric in tone_df.columns:
                        tone_metrics[metric] = {
                            "mean": tone_df[metric].mean() if not tone_df[metric].empty else 0,
                            "median": tone_df[metric].median() if not tone_df[metric].empty else 0,
                            "stddev": tone_df[metric].std() if not tone_df[metric].empty else 0,
                            "count": len(tone_df[metric].dropna())  # Count all valid solutions
                        }
                
                tone_metrics["solution_count"] = len(tone_df)
                tone_metrics["problem_count"] = len(tone_df['problem_id'].unique())
                
                rq_data[tone] = tone_metrics
        
        rq_metrics[rq]["data"] = rq_data
    
    return rq_metrics


def calculate_per_question_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics for each question across tones."""
    # Get list of all metrics to analyze
    rq2_metrics = [
        "comment_line_count", "docstring_count", "comment_char_count", 
        "docstring_char_count", "total_comment_char_count", "comment_to_code_char_ratio",
        "avg_identifier_length", "pylint_conventions", "char_count"
    ]
    
    rq3_metrics = [
        "avg_cyclomatic_complexity", "max_cyclomatic_complexity", "total_cyclomatic_complexity",
        "logical_loc", "source_loc", "raw_loc", "pylint_score", 
        "pylint_errors", "pylint_warnings", "pylint_issue_count"
    ]
    
    all_metrics = rq2_metrics + rq3_metrics
    
    # Create dictionary to hold results
    per_question_metrics = {}
    
    # Process each question
    for problem_id, problem_df in df.groupby('problem_id'):
        problem_results = {
            "tones": {},
            "variances": {}
        }
        
        # Calculate metrics by tone for this question
        for tone, tone_df in problem_df.groupby('tone_category'):
            tone_metrics = {}
            
            # Use all solutions instead of just the first one
            for metric in all_metrics:
                if metric in tone_df.columns:
                    values = tone_df[metric].dropna()
                    if not values.empty:
                        tone_metrics[metric] = {
                            "mean": values.mean(),
                            "count": len(values)  # Count represents total number of solutions
                        }
                    else:
                        tone_metrics[metric] = {
                            "mean": 0,
                            "count": 0
                        }
            
            # Add pass rate considering all solutions
            if 'graded' in tone_df.columns:
                # Sort by solution_idx to ensure we're evaluating solutions in order
                sorted_df = tone_df.sort_values('solution_idx')
                graded_values = sorted_df['graded'].values
                
                # Calculate pass@k rates
                pass_at_1 = graded_values[0] if len(graded_values) > 0 else False
                pass_at_5 = any(graded_values[:5]) if len(graded_values) >= 5 else any(graded_values) if len(graded_values) > 0 else False
                pass_at_10 = any(graded_values[:10]) if len(graded_values) >= 10 else any(graded_values) if len(graded_values) > 0 else False
                
                # Overall pass rate
                all_passes = sorted_df['graded'].sum() if not sorted_df.empty else 0
                total_sols = len(sorted_df) if not sorted_df.empty else 0
                
                tone_metrics["pass@1"] = {
                    "rate": 1.0 if pass_at_1 else 0.0,
                    "count": 1  # Always 1 for pass@1
                }
                
                tone_metrics["pass@5"] = {
                    "rate": 1.0 if pass_at_5 else 0.0,
                    "count": min(5, len(graded_values))  # Count up to 5
                }
                
                tone_metrics["pass@10"] = {
                    "rate": 1.0 if pass_at_10 else 0.0,
                    "count": min(10, len(graded_values))  # Count up to 10
                }
                
                tone_metrics["overall_pass"] = {
                    "rate": all_passes / total_sols if total_sols > 0 else 0,
                    "count": total_sols
                }
            
            problem_results["tones"][tone] = tone_metrics
        
        # Calculate between-tone variance for each metric for this problem
        for metric in all_metrics + ["pass@1", "pass@5", "pass@10", "overall_pass"]:
            # Get means and counts by tone
            if metric in ["pass@1", "pass@5", "pass@10", "overall_pass"]:
                means = [t.get(metric, {}).get("rate", 0) for t in problem_results["tones"].values()]
                counts = [t.get(metric, {}).get("count", 0) for t in problem_results["tones"].values()]
            else:
                if metric in tone_df.columns:
                    means = [t.get(metric, {}).get("mean", 0) for t in problem_results["tones"].values()]
                    counts = [t.get(metric, {}).get("count", 0) for t in problem_results["tones"].values()]
            
            # Calculate weighted variance if possible
            if len(means) > 1 and sum(counts) > 0:
                # Convert to numpy arrays for calculation
                means_array = np.array(means)
                counts_array = np.array(counts)
                
                # Filter out zeros
                valid_indices = counts_array > 0
                if np.sum(valid_indices) > 1:
                    valid_means = means_array[valid_indices]
                    valid_counts = counts_array[valid_indices]
                    
                    # Calculate weighted mean
                    weighted_mean = np.average(valid_means, weights=valid_counts)
                    
                    # Calculate weighted variance
                    variance = np.average((valid_means - weighted_mean) ** 2, weights=valid_counts)
                    variance = variance * (np.sum(valid_counts) / (np.sum(valid_counts) - 1))
                    
                    problem_results["variances"][metric] = float(variance)
                else:
                    problem_results["variances"][metric] = 0.0
            else:
                problem_results["variances"][metric] = 0.0
        
        per_question_metrics[problem_id] = problem_results
    
    return per_question_metrics


def calculate_variance_components(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate variance components (between-question, between-tone, within) for key metrics."""
    # Define metrics for each research question
    rq_metrics = {
        "RQ1": ["graded"],  # For pass@1 (first solution correctness)
        "RQ2": [
            "comment_line_count", "docstring_count", "comment_char_count", 
            "docstring_char_count", "total_comment_char_count", "comment_to_code_char_ratio",
            "avg_identifier_length", "pylint_conventions", "char_count"
        ],
        "RQ3": [
            "avg_cyclomatic_complexity", "max_cyclomatic_complexity", "total_cyclomatic_complexity",
            "logical_loc", "source_loc", "raw_loc", "pylint_score", 
            "pylint_errors", "pylint_warnings", "pylint_issue_count"
        ]
    }
    
    # Keep track of first solutions for pass@1 analysis
    first_solutions = df[df['solution_idx'] == 0]
    
    # Prepare result structure
    variance_components = {}
    
    # Calculate variance components for each metric
    for rq, metrics in rq_metrics.items():
        variance_components[rq] = {}
        
        for metric in metrics:
            # For graded (RQ1), we need to handle it specially
            if metric == "graded":
                # For pass@1, use only first solutions
                pass_at_1_variance = calculate_metric_variance_components(first_solutions, metric)
                
                # For pass@5 and pass@10, we need to process by problem/tone groups
                pass_at_5_data = process_pass_at_k(df, k=5)
                pass_at_10_data = process_pass_at_k(df, k=10)
                
                # For all solutions pass rate, use all data
                all_solutions_variance = calculate_metric_variance_components(df, metric)
                
                variance_components[rq]["pass@1"] = pass_at_1_variance
                variance_components[rq]["pass@5"] = calculate_metric_variance_components(pass_at_5_data, 'pass_at_k')
                variance_components[rq]["pass@10"] = calculate_metric_variance_components(pass_at_10_data, 'pass_at_k')
                variance_components[rq]["overall_pass"] = all_solutions_variance
            else:
                # For all other metrics, use all solutions
                if metric in df.columns:
                    metric_variance = calculate_metric_variance_components(df, metric)
                    variance_components[rq][metric] = metric_variance
    
    return variance_components


def calculate_metric_variance_components(data: pd.DataFrame, metric: str) -> Dict[str, float]:
    """Helper function to calculate variance components for a specific metric."""
    if metric not in data.columns:
        return {
            "total_variance": 0.0,
            "between_question_variance": 0.0,
            "between_tone_variance": 0.0,
            "residual_variance": 0.0,
            "variance_explained_by_question": 0.0,
            "variance_explained_by_tone": 0.0
        }
    
    # Drop NaN values
    valid_data = data[~data[metric].isna()]
    if valid_data.empty:
        return {
            "total_variance": 0.0,
            "between_question_variance": 0.0,
            "between_tone_variance": 0.0,
            "residual_variance": 0.0,
            "variance_explained_by_question": 0.0,
            "variance_explained_by_tone": 0.0
        }
    
    # Calculate total variance
    total_variance = valid_data[metric].var(ddof=1) if len(valid_data) > 1 else 0.0
    
    # Calculate grand mean
    grand_mean = valid_data[metric].mean()
    
    # Calculate between-question variance
    problem_stats = valid_data.groupby('problem_id').agg({
        metric: ['mean', 'count']
    })
    problem_stats.columns = ['mean', 'count']
    
    if len(problem_stats) > 1:
        # Calculate weighted between-question variance
        weights = problem_stats['count']
        means = problem_stats['mean']
        
        # Handle empty or single item case
        if len(means) <= 1 or sum(weights) <= 1:
            between_question_variance = 0.0
        else:
            weighted_mean_of_squares = np.average((means - grand_mean) ** 2, weights=weights)
            between_question_variance = weighted_mean_of_squares * (sum(weights) / (sum(weights) - 1))
            between_question_variance = min(between_question_variance, total_variance)
    else:
        between_question_variance = 0.0
    
    # Calculate between-tone variance (within questions)
    tone_question_stats = valid_data.groupby(['tone_category', 'problem_id']).agg({
        metric: ['mean', 'count']
    })
    tone_question_stats.columns = ['mean', 'count']
    
    # Reset index to get tone and problem_id as columns
    tone_question_stats = tone_question_stats.reset_index()
    
    # Join with problem means to get problem-level deviations
    problem_means = problem_stats.reset_index()[['problem_id', 'mean']].rename(columns={'mean': 'problem_mean'})
    tone_question_stats = pd.merge(tone_question_stats, problem_means, on='problem_id')
    
    # Calculate deviations from problem means
    tone_question_stats['deviation'] = tone_question_stats['mean'] - tone_question_stats['problem_mean']
    
    # Group by tone to get variance of deviations (within-question between-tone)
    tone_variance_stats = tone_question_stats.groupby('tone_category').agg({
        'deviation': lambda x: x.var(ddof=1) if len(x) > 1 else 0.0,
        'count': 'sum'
    })
    
    # Weight by number of samples
    if not tone_variance_stats.empty and sum(tone_variance_stats['count']) > 0:
        between_tone_variance = np.average(
            tone_variance_stats['deviation'], 
            weights=tone_variance_stats['count']
        )
    else:
        between_tone_variance = 0.0
    
    # Calculate residual (within) variance
    residual_variance = max(0.0, total_variance - between_question_variance - between_tone_variance)
    
    return {
        "total_variance": float(total_variance),
        "between_question_variance": float(between_question_variance),
        "between_tone_variance": float(between_tone_variance),
        "residual_variance": float(residual_variance),
        "variance_explained_by_question": float(between_question_variance / total_variance if total_variance > 0 else 0),
        "variance_explained_by_tone": float(between_tone_variance / total_variance if total_variance > 0 else 0)
    }


def process_pass_at_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Process dataframe to calculate pass@k metrics for variance analysis."""
    # Group by tone and problem_id
    pass_at_k_results = []
    
    for (tone, problem), group in df.groupby(['tone_category', 'problem_id']):
        sorted_group = group.sort_values('solution_idx')
        solutions_to_check = min(k, len(sorted_group))
        
        # Check if any of the first k solutions pass
        passes = any(sorted_group.iloc[:solutions_to_check]['graded']) if solutions_to_check > 0 else False
        
        pass_at_k_results.append({
            'tone_category': tone,
            'problem_id': problem,
            'pass_at_k': 1.0 if passes else 0.0,
            'count': solutions_to_check
        })
    
    return pd.DataFrame(pass_at_k_results)


def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all results with variance analysis by research question."""
    # Convert to DataFrame for easier analysis
    all_solutions = []
    
    for result in all_results:
        tone = result["tone_category"]
        model = result["model_name"]
        
        for problem_id, solutions in result["results"].items():
            for solution in solutions:
                solution_data = {k: v for k, v in solution.items() if k != 'code'}
                solution_data['tone_category'] = tone
                solution_data['model_name'] = model
                all_solutions.append(solution_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_solutions)
    
    # Basic stats
    aggregation = {
        "overall": {
            "total_problems": len(df['problem_id'].unique()),
            "total_solutions": len(df),
            "tones": list(df['tone_category'].unique())
        }
    }
    
    # Aggregate metrics by research question
    aggregation["research_questions"] = aggregate_metrics_by_rq(df)
    
    # Calculate per-question metrics and variance
    aggregation["per_question"] = calculate_per_question_metrics(df)
    
    # Calculate overall variance components for each metric (between-question, between-tone, within)
    variance_components = calculate_variance_components(df)
    aggregation["variance_components"] = variance_components
    
    return aggregation


def save_metrics_results(results, output_dir):
    """Save metrics results to files."""
    tone_category = results['tone_category']
    tone_dir = os.path.join(output_dir, tone_category)
    os.makedirs(tone_dir, exist_ok=True)
    
    # Save aggregate results only
    aggregated = {}
    for problem_id, problem_metrics in results['results'].items():
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame([{k: v for k, v in m.items() if k != 'code'} for m in problem_metrics])
        
        problem_summary = {
            'problem_id': problem_id,
            'solution_count': len(problem_metrics),
            'avg_metrics': {col: df[col].mean() for col in df.select_dtypes(include=['number']).columns},
            'pass_rate': df['graded'].mean() if 'graded' in df else None,
            'solutions': df.to_dict('records')
        }
        
        aggregated[problem_id] = problem_summary
    
    # Save to file - FIX: Convert NumPy types to Python native types
    with open(os.path.join(tone_dir, 'metrics_results.json'), 'w', encoding='utf-8') as f:
        json.dump(convert_to_python_types(aggregated), f, indent=2)


def find_eval_all_json_files(base_dir: str) -> List[str]:
    """Find all eval_all.json files in the directory structure."""
    eval_all_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_eval_all.json'):
                file_path = os.path.join(root, file)
                eval_all_files.append(file_path)
    
    return eval_all_files


def main():
    """Main function to process eval_all.json files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate code metrics for eval_all.json files')
    parser.add_argument('input', help='Input directory containing model/tone folders with eval_all.json files')
    parser.add_argument('--output', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--aggregate-only', action='store_true', help='Only generate aggregated report without individual results')
    parser.add_argument('--specific-file', help='Process only a specific eval_all.json file (optional)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    all_results = []
    
    if args.specific_file and os.path.isfile(args.specific_file) and args.specific_file.endswith('.json'):
        # Process just the specified file
        print(f"Processing specific file: {args.specific_file}...")
        results = process_eval_all_json(args.specific_file)
        all_results.append(results)
        
        if not args.aggregate_only:
            save_metrics_results(results, args.output)
    
    elif os.path.isdir(args.input):
        # Find and process all eval_all.json files
        eval_all_files = find_eval_all_json_files(args.input)
        
        if not eval_all_files:
            print(f"No eval_all.json files found in {args.input}")
            return
            
        print(f"Found {len(eval_all_files)} eval_all.json files to process")
        
        for file_path in eval_all_files:
            print(f"Processing {file_path}...")
            results = process_eval_all_json(file_path)
            all_results.append(results)
            
            if not args.aggregate_only:
                save_metrics_results(results, args.output)
    
    elif args.input.endswith('.json'):
        # Process a single file
        print(f"Processing {args.input}...")
        results = process_eval_all_json(args.input)
        all_results.append(results)
        
        if not args.aggregate_only:
            save_metrics_results(results, args.output)
    
    else:
        print(f"Error: {args.input} is not a valid directory or JSON file")
        return
    
    # Generate and save aggregated metrics
    if all_results:
        print("Generating aggregated metrics report...")
        aggregated_metrics = aggregate_metrics(all_results)
        
        # FIX: Convert all NumPy types to Python native types before saving
        aggregated_metrics_native = convert_to_python_types(aggregated_metrics)
        
        with open(os.path.join(args.output, 'aggregated_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(aggregated_metrics_native, f, indent=2)
        
        print(f"Aggregated metrics saved to {args.output}/aggregated_metrics.json")


if __name__ == "__main__":
    main()