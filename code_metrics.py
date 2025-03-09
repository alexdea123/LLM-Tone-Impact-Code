import ast
import io
import sys
import os
import re
import tempfile
import json
import statistics
from collections import defaultdict, Counter
from typing import Dict, Any, List
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


def calculate_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate code quality metrics for a given Python code using existing packages.
    
    Args:
        code: String containing Python code
        
    Returns:
        Dictionary containing various code quality metrics
    """
    metrics = {
        'lines_of_code': len(code.splitlines()),
        'char_count': len(code)
    }
    
    try:
        # Use lizard for most code metrics (cyclomatic complexity, etc.)
        analysis = lizard.analyze_file.analyze_source_code("temp.py", code)
        
        # Extract function-level metrics
        if analysis.function_list:
            metrics['function_count'] = len(analysis.function_list)
            metrics['avg_cyclomatic_complexity'] = np.mean([f.cyclomatic_complexity for f in analysis.function_list])
            metrics['max_cyclomatic_complexity'] = max([f.cyclomatic_complexity for f in analysis.function_list])
            metrics['avg_method_length'] = np.mean([f.nloc for f in analysis.function_list])
            metrics['max_method_length'] = max([f.nloc for f in analysis.function_list]) if analysis.function_list else 0
            metrics['min_method_length'] = min([f.nloc for f in analysis.function_list]) if analysis.function_list else 0
        else:
            metrics['function_count'] = 0
            metrics['avg_cyclomatic_complexity'] = 0
            metrics['max_cyclomatic_complexity'] = 0
            metrics['avg_method_length'] = 0
            metrics['max_method_length'] = 0
            metrics['min_method_length'] = 0
        
        # Parse AST for other metrics
        tree = ast.parse(code)
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        import_count = sum(1 for node in ast.walk(tree) 
                           if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom))
        docstring_count = sum(1 for node in ast.walk(tree) 
                             if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) 
                                and ast.get_docstring(node)))
        
        metrics['class_count'] = class_count
        metrics['import_count'] = import_count
        metrics['docstring_count'] = docstring_count
        
        # Radon - Raw metrics for code size statistics
        try:
            raw_metrics = radon_raw.analyze(code)
            metrics['raw_lloc'] = raw_metrics.lloc
            metrics['raw_sloc'] = raw_metrics.sloc
            metrics['raw_comments'] = raw_metrics.comments
            metrics['raw_multi'] = raw_metrics.multi
            metrics['raw_blank'] = raw_metrics.blank
        except Exception:
            pass
        
        # Run pylint for code quality score
        metrics.update(get_pylint_metrics(code))
        
        # Add line length metrics
        lines = [line for line in code.splitlines() if line.strip()]
        if lines:
            line_lengths = [len(line) for line in lines]
            metrics['max_line_length'] = max(line_lengths) if line_lengths else 0
            metrics['avg_line_length'] = np.mean(line_lengths) if line_lengths else 0
        
        # Count comments
        comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith('#'))
        metrics['comment_count'] = comment_lines
        metrics['comment_ratio'] = comment_lines / len(lines) if lines else 0
        
    except SyntaxError as e:
        metrics['syntax_error'] = True
        metrics['error_message'] = str(e)
    except Exception as e:
        metrics['error'] = str(e)
        
    return metrics

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
    # Extract tone category and model from path
    tone_match = re.search(r'/([^/]+)/Scenario\.codegeneration', file_path)
    model_match = re.search(r'/([^/]+)/[^/]+/Scenario\.codegeneration', file_path)
    
    tone_category = tone_match.group(1) if tone_match else "unknown"
    model_name = model_match.group(1) if model_match else "unknown"
    
    print(f"Processing file with tone: {tone_category}, model: {model_name}")
    
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    pass_at_1_count = 0
    total_problems = 0
    
    for problem_idx, problem in enumerate(data):
        problem_id = problem.get('question_id', f"problem_{problem_idx}")
        problem_metrics = []
        
        # Track pass@1 if available
        pass_at_1 = problem.get('pass@1', None)
        if pass_at_1 is not None:
            pass_at_1_count += pass_at_1
            total_problems += 1
        
        # Process each solution
        for solution_idx, code_solution in enumerate(problem.get('code_list', [])):
            if not code_solution.strip():
                continue
                
            # Calculate metrics
            metrics = calculate_metrics(code_solution)
            
            # Add metadata
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
                'model_name': model_name
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
        "total_problems": total_problems
    }


def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all results with variance analysis."""
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
    
    # Prepare aggregation structure
    aggregation = {
        "overall": {
            "total_problems": len(df['problem_id'].unique()),
            "total_solutions": len(df),
            "metrics_summary": {}
        },
        "by_tone": {},
        "by_difficulty": {},
        "by_platform": {},
        "variance_analysis": {}
    }
    
    # Calculate pass@1 and pass@5 rates by tone
    pass_rates = df.groupby(['tone_category', 'problem_id']).agg(
        first_solution_passed=('graded', lambda x: x.iloc[0] == True),
        any_solution_passed=('graded', lambda x: any(x == True))
    ).reset_index()
    
    tone_performance = pass_rates.groupby('tone_category').agg(
        pass_at_1_rate=('first_solution_passed', 'mean'),
        pass_at_5_rate=('any_solution_passed', 'mean'),
        problem_count=('problem_id', 'count')
    ).reset_index()
    
    # Add to aggregation
    aggregation["overall"]["pass_at_1_rate"] = pass_rates['first_solution_passed'].mean()
    aggregation["overall"]["pass_at_5_rate"] = pass_rates['any_solution_passed'].mean()
    
    # Add variance analysis for pass@1 and pass@5
    aggregation["overall"]["pass_at_1_variance"] = pass_rates['first_solution_passed'].var(ddof=1)
    aggregation["overall"]["pass_at_5_variance"] = pass_rates['any_solution_passed'].var(ddof=1)
    
    # Calculate per-tone variance for pass@1 and pass@5
    tone_pass_variance = pass_rates.groupby('tone_category').agg(
        pass_at_1_variance=('first_solution_passed', lambda x: x.var(ddof=1) if len(x) > 1 else 0),
        pass_at_5_variance=('any_solution_passed', lambda x: x.var(ddof=1) if len(x) > 1 else 0)
    ).reset_index()
    
    # Calculate metrics for each tone
    for tone, tone_df in df.groupby('tone_category'):
        tone_metrics = {}
        
        # Numeric metrics only
        numeric_cols = [col for col in tone_df.columns if 
                       col not in ['problem_id', 'problem_title', 'solution_idx', 'graded', 
                                  'difficulty', 'platform', 'tone_category', 'model_name',
                                  'error_code', 'error_message'] and 
                       pd.api.types.is_numeric_dtype(tone_df[col])]
        
        for metric in numeric_cols:
            if not tone_df[metric].empty:
                tone_metrics[metric] = {
                    "mean": tone_df[metric].mean(),
                    "median": tone_df[metric].median(),
                    "min": tone_df[metric].min(),
                    "max": tone_df[metric].max(),
                    "stddev": tone_df[metric].std()
                }
        
        # Get pass rates for this tone
        tone_pass = tone_performance[tone_performance['tone_category'] == tone]
        tone_var = tone_pass_variance[tone_pass_variance['tone_category'] == tone]
        
        aggregation["by_tone"][tone] = {
            "total_problems": tone_pass['problem_count'].iloc[0] if not tone_pass.empty else 0,
            "total_solutions": len(tone_df),
            "pass_at_1_rate": tone_pass['pass_at_1_rate'].iloc[0] if not tone_pass.empty else 0,
            "pass_at_5_rate": tone_pass['pass_at_5_rate'].iloc[0] if not tone_pass.empty else 0,
            "pass_at_1_variance": tone_var['pass_at_1_variance'].iloc[0] if not tone_var.empty else 0,
            "pass_at_5_variance": tone_var['pass_at_5_variance'].iloc[0] if not tone_var.empty else 0,
            "metrics": tone_metrics
        }
    
    # Calculate between-question and within-question variance for key metrics
    key_metrics = ["avg_cyclomatic_complexity", "pylint_score", "raw_lloc"]
    variance_analysis = {}
    
    # Add variance analysis for pass@1 and pass@5
    for pass_metric, column in [("pass@1", "first_solution_passed"), ("pass@5", "any_solution_passed")]:
        # Total variance
        total_variance = pass_rates[column].var(ddof=1)
        
        # Calculate grand mean
        grand_mean = pass_rates[column].mean()
        
        # Between-tone variance
        tone_means = pass_rates.groupby('tone_category')[column].mean()
        tone_counts = pass_rates.groupby('tone_category')[column].count()
        
        if len(tone_means) > 1:
            # Weighted calculation of between-tone variance
            weights = tone_counts
            means = tone_means
            weighted_mean_of_squares = np.average((means - grand_mean) ** 2, weights=weights)
            between_tone_variance = weighted_mean_of_squares * (sum(weights) / (sum(weights) - 1))
            between_tone_variance = min(between_tone_variance, total_variance)
        else:
            between_tone_variance = 0.0
            
        # Within-tone variance (how consistent each tone is across different questions)
        within_variance_by_tone = {}
        for tone, tone_data in pass_rates.groupby('tone_category'):
            if len(tone_data) > 1:
                tone_variance = tone_data[column].var(ddof=1)
                within_variance_by_tone[tone] = float(tone_variance)
            else:
                within_variance_by_tone[tone] = 0.0
                
        variance_analysis[pass_metric] = {
            "total_variance": float(total_variance) if pd.notna(total_variance) else 0.0,
            "between_tone_variance": float(between_tone_variance),
            "within_tone_variance_by_tone": within_variance_by_tone
        }
    
    # Continue with existing variance analysis for other metrics
    for metric in key_metrics:
        if metric in df.columns:
            # Drop NaN values for consistent calculations
            valid_data = df[~df[metric].isna()]
            if valid_data.empty:
                continue
                
            # Total variance with explicit ddof=1 for sample variance
            total_variance = valid_data[metric].var(ddof=1)
            
            # Calculate grand mean
            grand_mean = valid_data[metric].mean()
            
            # Group by problem_id and calculate means and counts
            problem_stats = valid_data.groupby('problem_id').agg({
                metric: ['mean', 'count']
            })
            problem_stats.columns = ['mean', 'count']
            
            # Calculate between-question variance properly (weighted)
            if len(problem_stats) > 1:
                # Weighted calculation of between-question variance
                weights = problem_stats['count']
                means = problem_stats['mean']
                weighted_mean_of_squares = np.average((means - grand_mean) ** 2, weights=weights)
                between_question_variance = weighted_mean_of_squares * (sum(weights) / (sum(weights) - 1))
                
                # Ensure between variance is not larger than total variance due to numerical issues
                between_question_variance = min(between_question_variance, total_variance)
            else:
                between_question_variance = 0.0
            
            # Within-question variance by tone (random slopes)
            within_variance_by_tone = {}
            
            for tone, tone_df in valid_data.groupby('tone_category'):
                # Calculate how each tone's effect varies across questions
                tone_effects = []
                
                for problem, problem_df in tone_df.groupby('problem_id'):
                    if len(problem_df) > 0:
                        # Get the problem mean from our stats DataFrame
                        if problem in problem_stats.index:
                            problem_mean = problem_stats.loc[problem, 'mean']
                            tone_problem_mean = problem_df[metric].mean()
                            tone_effect = tone_problem_mean - problem_mean
                            tone_effects.append(tone_effect)
                
                # Calculate variance if we have enough data points (at least 2)
                if len(tone_effects) > 1:
                    within_variance_by_tone[tone] = float(np.var(tone_effects, ddof=1))
                elif len(tone_effects) == 1:
                    # If only one data point, variance is 0
                    within_variance_by_tone[tone] = 0.0
                else:
                    # No valid data points, set to 0
                    within_variance_by_tone[tone] = 0.0
            
            variance_analysis[metric] = {
                "total_variance": float(total_variance) if pd.notna(total_variance) else 0.0,
                "between_question_variance": float(between_question_variance),
                "within_question_variance_by_tone": within_variance_by_tone
            }
    
    aggregation["variance_analysis"]["pass_metrics"] = variance_analysis
    
    # Tone rankings based on pass@1 and pass@5
    top_tone_pass1 = tone_performance.loc[tone_performance['pass_at_1_rate'].idxmax()]
    worst_tone_pass1 = tone_performance.loc[tone_performance['pass_at_1_rate'].idxmin()]
    top_tone_pass5 = tone_performance.loc[tone_performance['pass_at_5_rate'].idxmax()]
    worst_tone_pass5 = tone_performance.loc[tone_performance['pass_at_5_rate'].idxmin()]
    
    aggregation["tone_rankings"] = {
        "by_pass_at_1": tone_performance.sort_values('pass_at_1_rate', ascending=False)[['tone_category', 'pass_at_1_rate']].values.tolist(),
        "by_pass_at_5": tone_performance.sort_values('pass_at_5_rate', ascending=False)[['tone_category', 'pass_at_5_rate']].values.tolist(),
        "top_performing_tone_pass1": top_tone_pass1['tone_category'],
        "lowest_performing_tone_pass1": worst_tone_pass1['tone_category'],
        "top_performing_tone_pass5": top_tone_pass5['tone_category'],
        "lowest_performing_tone_pass5": worst_tone_pass5['tone_category']
    }
    
    # Add summary statistics for key metrics
    aggregation["key_metrics_by_tone"] = {}
    for metric in key_metrics:
        if metric in df.columns:
            aggregation["key_metrics_by_tone"][metric] = df.groupby('tone_category')[metric].mean().to_dict()
    
    # Calculate overall metrics summaries
    for metric in numeric_cols:
        aggregation["overall"]["metrics_summary"][metric] = {
            "mean": df[metric].mean(),
            "median": df[metric].median(),
            "min": df[metric].min(),
            "max": df[metric].max(),
            "stddev": df[metric].std()
        }
    
    # Add error type analysis - FIX: create a proper copy of the DataFrame
    if 'error_message' in df.columns:
        error_df = df[df['error_message'].notna()].copy()  # Create an explicit copy
        if not error_df.empty:
            error_df['error_type'] = error_df['error_message'].apply(
                lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else 'Unknown')
            error_counts = error_df['error_type'].value_counts().to_dict()
            aggregation["overall"]["error_types"] = error_counts
    
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