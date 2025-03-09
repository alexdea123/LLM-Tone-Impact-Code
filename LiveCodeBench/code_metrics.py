import ast
import io
import sys
import os
import re
import tempfile
import subprocess
import json
import statistics
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Union, Set
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
from pylint import lint
from pylint.reporters.text import TextReporter

class NodeVisitor(ast.NodeVisitor):
    """Custom AST node visitor to collect various metrics."""
    
    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.if_count = 0
        self.loop_count = 0
        self.exception_count = 0
        self.import_count = 0
        self.docstring_count = 0
        self.method_lengths = []
        
    def visit_FunctionDef(self, node):
        self.function_count += 1
        if ast.get_docstring(node):
            self.docstring_count += 1
        self.method_lengths.append(len(node.body))
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        if ast.get_docstring(node):
            self.docstring_count += 1
        self.method_lengths.append(len(node.body))
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.class_count += 1
        if ast.get_docstring(node):
            self.docstring_count += 1
        self.generic_visit(node)
        
    def visit_If(self, node):
        self.if_count += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.loop_count += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.loop_count += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.exception_count += 1
        self.generic_visit(node)
        
    def visit_Import(self, node):
        self.import_count += len(node.names)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        self.import_count += len(node.names)
        self.generic_visit(node)


def calculate_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate a comprehensive set of code quality metrics for a given Python code.
    
    Args:
        code: String containing Python code
        
    Returns:
        Dictionary containing various code quality metrics
    """
    metrics = {}
    
    # Basic size metrics
    metrics['lines_of_code'] = len(code.splitlines())
    metrics['char_count'] = len(code)
    
    try:
        # Parse the AST for custom metrics
        tree = ast.parse(code)
        visitor = NodeVisitor()
        visitor.visit(tree)
        
        # Add custom AST metrics
        metrics['function_count'] = visitor.function_count
        metrics['class_count'] = visitor.class_count
        metrics['if_count'] = visitor.if_count
        metrics['loop_count'] = visitor.loop_count
        metrics['exception_count'] = visitor.exception_count
        metrics['import_count'] = visitor.import_count
        metrics['docstring_count'] = visitor.docstring_count
        
        # Method length statistics
        if visitor.method_lengths:
            metrics['avg_method_length'] = sum(visitor.method_lengths) / len(visitor.method_lengths)
            metrics['max_method_length'] = max(visitor.method_lengths)
            metrics['min_method_length'] = min(visitor.method_lengths)
        else:
            metrics['avg_method_length'] = 0
            metrics['max_method_length'] = 0
            metrics['min_method_length'] = 0
            
        # Radon metrics - Cyclomatic Complexity
        try:
            cc_results = radon_cc.cc_visit(code)
            if cc_results:
                complexities = [result.complexity for result in cc_results]
                metrics['avg_cyclomatic_complexity'] = sum(complexities) / len(complexities) if complexities else 0
                metrics['max_cyclomatic_complexity'] = max(complexities) if complexities else 0
            else:
                metrics['avg_cyclomatic_complexity'] = 0
                metrics['max_cyclomatic_complexity'] = 0
        except Exception as e:
            metrics['radon_cc_error'] = str(e)
            metrics['avg_cyclomatic_complexity'] = 0
            metrics['max_cyclomatic_complexity'] = 0
        
        # Radon - Raw metrics
        try:
            raw_metrics = radon_raw.analyze(code)
            metrics['raw_loc'] = raw_metrics.loc
            metrics['raw_lloc'] = raw_metrics.lloc
            metrics['raw_sloc'] = raw_metrics.sloc
            metrics['raw_comments'] = raw_metrics.comments
            metrics['raw_multi'] = raw_metrics.multi
            metrics['raw_blank'] = raw_metrics.blank
            metrics['raw_single_comments'] = raw_metrics.single_comments
        except Exception as e:
            metrics['radon_raw_error'] = str(e)
        
        # Radon - Halstead metrics
        try:
            halstead_metrics = radon_metrics.h_visit(code)
            if halstead_metrics:
                metrics['halstead_h1'] = halstead_metrics.h1
                metrics['halstead_h2'] = halstead_metrics.h2
                metrics['halstead_N1'] = halstead_metrics.N1
                metrics['halstead_N2'] = halstead_metrics.N2
                metrics['halstead_vocabulary'] = halstead_metrics.vocabulary
                metrics['halstead_length'] = halstead_metrics.length
                metrics['halstead_volume'] = halstead_metrics.volume
                metrics['halstead_difficulty'] = halstead_metrics.difficulty
                metrics['halstead_effort'] = halstead_metrics.effort
                metrics['halstead_bugs'] = halstead_metrics.bugs
                metrics['halstead_time'] = halstead_metrics.time
        except Exception as e:
            metrics['radon_halstead_error'] = str(e)
            
        # Run pylint and capture the score
        try:
            metrics.update(get_pylint_metrics(code))
        except Exception as e:
            metrics['pylint_error'] = str(e)
        
        # Additional metrics
        try:
            additional_metrics = get_additional_metrics(code)
            metrics.update(additional_metrics)
        except Exception as e:
            metrics['additional_metrics_error'] = str(e)
            
    except SyntaxError as e:
        metrics['syntax_error'] = True
        metrics['error_message'] = str(e)
    except Exception as e:
        metrics['error'] = str(e)
        
    return metrics


def get_pylint_metrics(code: str) -> Dict[str, Any]:
    """
    Run pylint on the code and extract metrics.
    
    Args:
        code: String containing Python code
        
    Returns:
        Dictionary with pylint metrics
    """
    # Create a StringIO to capture pylint output
    pylint_output = io.StringIO()
    reporter = TextReporter(pylint_output)
    
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    # Run pylint on the temporary file
    try:
        # Suppress stdout during pylint run
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Run pylint
        lint.Run([
            '--disable=all',  # Disable all checks
            '--enable=C0103,C0111,C0301,C0303,C0304,C0305,E0001,E0102,E0103,E0104,E0105,E0213,E0602,E0611,E1101,E1102,E1111,E0702,W0101,W0102,W0104,W0106,W0107,W0612,W0622',
            '--reports=n',  # No report
            tmp_path
        ], reporter=reporter, exit=False)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Get pylint output
        pylint_text = pylint_output.getvalue()
        
        # Parse the output to get the score
        score_match = re.search(r'Your code has been rated at (-?\d+\.\d+)/10', pylint_text)
        score = float(score_match.group(1)) if score_match else 0.0
        
        # Count issues by type
        error_count = pylint_text.count('[E')
        warning_count = pylint_text.count('[W')
        convention_count = pylint_text.count('[C')
        refactor_count = pylint_text.count('[R')
        
        return {
            'pylint_score': score,
            'pylint_errors': error_count,
            'pylint_warnings': warning_count,
            'pylint_conventions': convention_count,
            'pylint_refactors': refactor_count,
            'pylint_issue_count': error_count + warning_count + convention_count + refactor_count
        }
        
    except Exception as e:
        return {
            'pylint_error': str(e),
            'pylint_score': 0.0,
            'pylint_issues': -1
        }
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def calculate_cognitive_complexity(code: str) -> int:
    """
    Calculate cognitive complexity of the code using custom implementation.
    
    This is a simplified version of cognitive complexity that counts:
    - Nesting levels (if, for, while, etc.)
    - Logical operators (and, or)
    - Conditional expressions (ternary)
    - Recursion
    
    Args:
        code: String containing Python code
        
    Returns:
        Cognitive complexity score (int)
    """
    class CognitiveComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.complexity = 0
            self.nesting_level = 0
            self.function_names = set()
            self.current_function_calls = set()
            
        def visit_If(self, node):
            self.complexity += 1 + self.nesting_level
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
            
        def visit_For(self, node):
            self.complexity += 1 + self.nesting_level
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
            
        def visit_While(self, node):
            self.complexity += 1 + self.nesting_level
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
            
        def visit_Try(self, node):
            self.complexity += 1 + self.nesting_level
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
            
        def visit_ExceptHandler(self, node):
            self.complexity += 1
            self.generic_visit(node)
            
        def visit_With(self, node):
            self.complexity += self.nesting_level
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
        
        def visit_BoolOp(self, node):
            # Add complexity for and/or operators
            self.complexity += len(node.values) - 1
            self.generic_visit(node)
            
        def visit_IfExp(self, node):
            # Add complexity for ternary operators
            self.complexity += 1
            self.generic_visit(node)
        
        def visit_FunctionDef(self, node):
            # Record function name for recursion detection
            self.function_names.add(node.name)
            old_calls = self.current_function_calls.copy()
            self.current_function_calls = set()
            self.generic_visit(node)
            
            # Check for recursion
            if node.name in self.current_function_calls:
                self.complexity += 1
                
            self.current_function_calls = old_calls
        
        def visit_AsyncFunctionDef(self, node):
            # Handle async functions the same way
            self.visit_FunctionDef(node)
            
        def visit_Call(self, node):
            # Track function calls to detect recursion
            if isinstance(node.func, ast.Name):
                self.current_function_calls.add(node.func.id)
            self.generic_visit(node)
            
        def visit_Lambda(self, node):
            self.complexity += 1
            self.nesting_level += 1
            self.generic_visit(node)
            self.nesting_level -= 1
            
    try:
        tree = ast.parse(code)
        visitor = CognitiveComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    except SyntaxError:
        # If the code has syntax errors, return a high complexity
        return 100
    except Exception as e:
        # For any other errors, return -1 to indicate an error
        return -1


def get_additional_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate additional code metrics not covered by other functions.
    
    Args:
        code: String containing Python code
        
    Returns:
        Dictionary with additional metrics
    """
    metrics = {}
    
    # Calculate cognitive complexity
    cognitive_complexity = calculate_cognitive_complexity(code)
    metrics['cognitive_complexity'] = cognitive_complexity
    
    # Calculate line length metrics
    lines = [line for line in code.splitlines() if line.strip()]
    if lines:
        line_lengths = [len(line) for line in lines]
        metrics['max_line_length'] = max(line_lengths) if line_lengths else 0
        metrics['avg_line_length'] = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    else:
        metrics['max_line_length'] = 0
        metrics['avg_line_length'] = 0
    
    # Count indentation levels
    indentation_levels = set()
    for line in lines:
        if line.strip():
            # Count leading spaces or tabs
            leading_spaces = len(line) - len(line.lstrip())
            indentation_levels.add(leading_spaces)
    
    metrics['max_indentation_level'] = max(indentation_levels) if indentation_levels else 0
    metrics['unique_indentation_levels'] = len(indentation_levels)
    
    # Count comments
    comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith('#'))
    metrics['comment_count'] = comment_lines
    
    if len(lines) > 0:
        metrics['comment_ratio'] = comment_lines / len(lines)
    else:
        metrics['comment_ratio'] = 0
    
    return metrics

def process_eval_all_json(file_path: str) -> Dict[str, Any]:
    """
    Process the eval_all.json file and calculate metrics for each code submission.
    
    Args:
        file_path: Path to the eval_all.json file
        
    Returns:
        Dictionary with metrics for each problem and solution
    """
    # Extract tone category from file path - improve regex to match the actual path structure
    tone_category_match = re.search(r'/([^/]+)/Scenario\.codegeneration', file_path)
    tone_category = tone_category_match.group(1) if tone_category_match else "unknown"
    
    # Extract model info from file path if available - improve to match the actual path structure
    model_match = re.search(r'/([^/]+)/[^/]+/Scenario\.codegeneration', file_path)
    model_name = model_match.group(1) if model_match else "unknown"
    
    # Debug output to verify extracted tone and model
    print(f"Processing file with tone: {tone_category}, model: {model_name}")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    pass_at_1_count = 0
    total_problems = 0
    
    for problem_idx, problem in enumerate(data):
        problem_id = problem.get('question_id', f"problem_{problem_idx}")
        problem_title = problem.get('question_title', f"problem_{problem_idx}")
        problem_difficulty = problem.get('difficulty', "unknown")
        problem_platform = problem.get('platform', "unknown")
        problem_metrics = []
        
        # Check if the problem has a pass@1 value directly
        pass_at_1 = problem.get('pass@1', None)
        if pass_at_1 is not None:
            pass_at_1_count += pass_at_1
            total_problems += 1
        
        # Process each solution/code submission for this problem
        for solution_idx, code_solution in enumerate(problem.get('code_list', [])):
            if not code_solution.strip():
                continue
                
            # Calculate metrics for this solution
            metrics = calculate_metrics(code_solution)
            
            # Add metadata
            metrics['problem_id'] = problem_id
            metrics['problem_title'] = problem_title
            metrics['solution_idx'] = solution_idx
            metrics['code'] = code_solution
            metrics['graded'] = problem.get('graded_list', [])[solution_idx] if solution_idx < len(problem.get('graded_list', [])) else None
            metrics['metadata'] = problem.get('metadata', [])[solution_idx] if solution_idx < len(problem.get('metadata', [])) else None
            metrics['difficulty'] = problem_difficulty
            metrics['platform'] = problem_platform
            metrics['tone_category'] = tone_category
            metrics['model_name'] = model_name
            
            # Extract error info from metadata if available
            if metrics['metadata']:
                try:
                    metadata = json.loads(metrics['metadata']) if isinstance(metrics['metadata'], str) else metrics['metadata']
                    metrics['error_code'] = metadata.get('error_code')
                    metrics['error_message'] = metadata.get('error_message')
                    metrics['execution_time'] = metadata.get('execution time')
                    metrics['expected'] = metadata.get('expected')
                    metrics['output'] = metadata.get('output')
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Add to results
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
    """
    Aggregate metrics across all processed files to produce comprehensive statistics.
    
    Args:
        all_results: List of results from processing multiple files
        
    Returns:
        Dictionary with aggregated statistics
    """
    # Initialize aggregation structures with proper nested defaultdicts
    aggregation = {
        "overall": {
            "total_problems": 0,
            "total_solutions": 0,
            "pass_at_1_rate": 0,
            "pass_at_5_rate": 0,
            "avg_metrics": defaultdict(list),
            "error_types": Counter(),
        },
        "by_tone": defaultdict(lambda: {
            "total_problems": 0,
            "total_solutions": 0,
            "problems_pass_at_1": 0,
            "problems_pass_at_5": 0,
            "pass_at_1_rate": 0,
            "pass_at_5_rate": 0,
            "avg_metrics": defaultdict(list),
            "error_types": Counter(),
        }),
        "by_difficulty": {},
        "by_platform": {},
        "metrics_variance_by_tone": {},
        "metrics_rankings": {},
    }
    
    # Initialize nested dictionaries explicitly
    difficulties = set()
    platforms = set()
    tone_categories = set()
    
    # First pass to collect all tone categories across all results
    for result in all_results:
        tone_category = result["tone_category"]
        tone_categories.add(tone_category)
    
    # Initialize problem tracking for each tone
    problem_tracking = {tone: set() for tone in tone_categories}
    problem_pass_at_1 = {tone: set() for tone in tone_categories}
    problem_pass_at_5 = {tone: set() for tone in tone_categories}
    
    # First pass to collect categories
    for result in all_results:
        tone_category = result["tone_category"]
        
        for problem_id, problem_metrics in result["results"].items():
            for solution in problem_metrics:
                difficulty = solution.get("difficulty", "unknown")
                platform = solution.get("platform", "unknown")
                difficulties.add(difficulty)
                platforms.add(platform)
    
    # Create the nested structure with proper initialization
    for difficulty in difficulties:
        aggregation["by_difficulty"][difficulty] = {
            "total_problems": 0,
            "total_solutions": 0,
            "pass_at_1_rate": 0,
            "pass_at_5_rate": 0,
            "avg_metrics": defaultdict(list),
            "by_tone": {}
        }
        
        for tone in tone_categories:
            aggregation["by_difficulty"][difficulty]["by_tone"][tone] = {
                "total_problems": 0,
                "total_solutions": 0,
                "pass_at_1_rate": 0,
                "avg_metrics": defaultdict(list),
            }
    
    for platform in platforms:
        aggregation["by_platform"][platform] = {
            "total_problems": 0,
            "total_solutions": 0,
            "pass_at_1_rate": 0,
            "pass_at_5_rate": 0,
            "avg_metrics": defaultdict(list),
            "by_tone": {}
        }
        
        for tone in tone_categories:
            aggregation["by_platform"][platform]["by_tone"][tone] = {
                "total_problems": 0,
                "total_solutions": 0,
                "pass_at_1_rate": 0,
                "avg_metrics": defaultdict(list),
            }
    
    # Collect metrics for all solutions across all files
    all_metrics = []
    
    # Track unique problems to avoid double-counting
    seen_problems = set()
    
    # Process each result file
    for result in all_results:
        tone_category = result["tone_category"]
        model_name = result.get("model_name", "unknown")
        
        # Now process each problem
        for problem_id, problem_metrics in result["results"].items():
            # Create a unique identifier for this problem
            unique_problem_id = f"{problem_id}-{model_name}-{tone_category}"
            
            # Skip if we've already seen this problem
            if unique_problem_id in seen_problems:
                continue
                
            seen_problems.add(unique_problem_id)
            problem_tracking[tone_category].add(unique_problem_id)
            
            # Extract basic problem info from first solution
            first_solution = problem_metrics[0] if problem_metrics else None
            if not first_solution:
                continue
                
            difficulty = first_solution.get("difficulty", "unknown")
            platform = first_solution.get("platform", "unknown")
            
            # Increment problem counts
            aggregation["overall"]["total_problems"] += 1
            aggregation["by_tone"][tone_category]["total_problems"] += 1
            aggregation["by_difficulty"][difficulty]["total_problems"] += 1
            aggregation["by_difficulty"][difficulty]["by_tone"][tone_category]["total_problems"] += 1
            aggregation["by_platform"][platform]["total_problems"] += 1
            aggregation["by_platform"][platform]["by_tone"][tone_category]["total_problems"] += 1
            
            # Check if first solution passes (Pass@1)
            if problem_metrics and problem_metrics[0].get("graded", False):
                problem_pass_at_1[tone_category].add(unique_problem_id)
                aggregation["by_tone"][tone_category]["problems_pass_at_1"] += 1
            
            # Check if any solution passes (Pass@5)
            if any(sol.get("graded", False) for sol in problem_metrics):
                problem_pass_at_5[tone_category].add(unique_problem_id)
                aggregation["by_tone"][tone_category]["problems_pass_at_5"] += 1
            
            # Process each solution
            for solution in problem_metrics:
                aggregation["overall"]["total_solutions"] += 1
                aggregation["by_tone"][tone_category]["total_solutions"] += 1
                
                # Add solution to all metrics for variance analysis
                solution_copy = solution.copy()
                solution_copy.pop("code", None)  # Remove code to save memory
                solution_copy.pop("metadata", None)  # Remove metadata to save memory
                all_metrics.append(solution_copy)
                
                # Extract error information
                error_code = solution.get("error_code")
                if error_code and error_code != 0:
                    error_message = solution.get("error_message", "Unknown error")
                    error_type = error_message.split(":")[0] if ":" in error_message else error_message
                    aggregation["overall"]["error_types"][error_type] += 1
                    aggregation["by_tone"][tone_category]["error_types"][error_type] += 1
                
                # Collect numeric metrics
                for key, value in solution.items():
                    if key not in ["problem_id", "problem_title", "solution_idx", "code", "graded", "metadata", 
                                  "difficulty", "platform", "tone_category", "model_name", "error_code", 
                                  "error_message", "execution_time", "expected", "output"]:
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            aggregation["overall"]["avg_metrics"][key].append(value)
                            aggregation["by_tone"][tone_category]["avg_metrics"][key].append(value)
                            
                            difficulty = solution.get("difficulty", "unknown")
                            platform = solution.get("platform", "unknown")
                            
                            aggregation["by_difficulty"][difficulty]["avg_metrics"][key].append(value)
                            aggregation["by_difficulty"][difficulty]["by_tone"][tone_category]["avg_metrics"][key].append(value)
                            
                            aggregation["by_platform"][platform]["avg_metrics"][key].append(value)
                            aggregation["by_platform"][platform]["by_tone"][tone_category]["avg_metrics"][key].append(value)
    
    # Calculate Pass@1 and Pass@5 rates
    for tone, data in aggregation["by_tone"].items():
        num_problems = len(problem_tracking[tone])
        data["total_problems"] = num_problems
        
        if num_problems > 0:
            data["pass_at_1_rate"] = len(problem_pass_at_1[tone]) / num_problems
            data["pass_at_5_rate"] = len(problem_pass_at_5[tone]) / num_problems
        else:
            data["pass_at_1_rate"] = 0
            data["pass_at_5_rate"] = 0
    
    # Calculate overall Pass@1 and Pass@5 rates
    total_problems = aggregation["overall"]["total_problems"]
    if total_problems > 0:
        total_pass_at_1 = sum(len(problem_set) for problem_set in problem_pass_at_1.values())
        total_pass_at_5 = sum(len(problem_set) for problem_set in problem_pass_at_5.values())
        aggregation["overall"]["pass_at_1_rate"] = total_pass_at_1 / total_problems
        aggregation["overall"]["pass_at_5_rate"] = total_pass_at_5 / total_problems

    # Prepare tone rankings
    tone_performance = [(tone, len(problem_pass_at_1[tone]) / len(problem_tracking[tone]) if len(problem_tracking[tone]) > 0 else 0)
                         for tone in tone_categories]
    
    # Create problems_by_tone with accurate counts
    problems_by_tone = {tone: len(problem_tracking[tone]) for tone in tone_categories}
    
    aggregation["tone_rankings"] = {
        "by_pass_at_1": sorted(tone_performance, key=lambda x: x[1], reverse=True),
        "problems_by_tone": problems_by_tone,
        "top_performing_tone": max(tone_performance, key=lambda x: x[1])[0] if tone_performance else None,
        "lowest_performing_tone": min(tone_performance, key=lambda x: (x[1], x[0]))[0] if tone_performance else None
    }
    
    # Calculate average complexity metrics by tone (for RQ1)
    key_metrics = ["avg_cyclomatic_complexity", "cognitive_complexity", "pylint_score", "raw_lloc"]
    
    # Make sure we calculate the averages before trying to access them
    for section in ["overall", "by_tone", "by_difficulty", "by_platform"]:
        if section == "overall":
            for metric, values in aggregation[section]["avg_metrics"].items():
                if values:
                    aggregation[section]["avg_metrics"][metric] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0
                    }
        elif section == "by_tone":
            for category, data in aggregation[section].items():
                for metric, values in data["avg_metrics"].items():
                    if values:
                        data["avg_metrics"][metric] = {
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "min": min(values),
                            "max": max(values),
                            "stddev": statistics.stdev(values) if len(values) > 1 else 0
                        }
        else:
            for category, data in aggregation[section].items():
                for metric, values in data["avg_metrics"].items():
                    if values:
                        data["avg_metrics"][metric] = {
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "min": min(values),
                            "max": max(values),
                            "stddev": statistics.stdev(values) if len(values) > 1 else 0
                        }
                
                # For nested tone categories in difficulty/platform sections
                if "by_tone" in data:
                    for tone, tone_data in data["by_tone"].items():
                        for metric, values in tone_data["avg_metrics"].items():
                            if values:
                                tone_data["avg_metrics"][metric] = {
                                    "mean": statistics.mean(values),
                                    "median": statistics.median(values),
                                    "min": min(values),
                                    "max": max(values),
                                    "stddev": statistics.stdev(values) if len(values) > 1 else 0
                                }
    
    # Now create the key_metrics_by_tone using the calculated means
    aggregation["key_metrics_by_tone"] = {
        metric: {
            tone: aggregation["by_tone"][tone]["avg_metrics"].get(metric, {}).get("mean", 0)
            for tone in tone_categories
        }
        for metric in key_metrics
    }
    
    # Convert defaultdicts to regular dicts for JSON serialization
    for section in ["by_tone", "by_difficulty", "by_platform"]:
        aggregation[section] = dict(aggregation[section])
        if section in ["by_difficulty", "by_platform"]:
            for category, data in aggregation[section].items():
                if "by_tone" in data:
                    data["by_tone"] = dict(data["by_tone"])
        
    for section in ["overall", "by_tone", "by_difficulty", "by_platform"]:
        if section == "overall":
            aggregation[section]["avg_metrics"] = dict(aggregation[section]["avg_metrics"])
            aggregation[section]["error_types"] = dict(aggregation[section]["error_types"])
        elif section == "by_tone":
            for category, data in aggregation[section].items():
                data["avg_metrics"] = dict(data["avg_metrics"])
                data["error_types"] = dict(data["error_types"])
        else:
            for category, data in aggregation[section].items():
                data["avg_metrics"] = dict(data["avg_metrics"])
                if "by_tone" in data:
                    for tone, tone_data in data["by_tone"].items():
                        tone_data["avg_metrics"] = dict(tone_data["avg_metrics"])
    
    return aggregation

def save_metrics_results(results, output_dir):
    """
    Save the metrics results to files.
    
    Args:
        results: Dictionary with metrics results
        output_dir: Directory to save the results to
    """
    tone_category = results['tone_category']
    tone_dir = os.path.join(output_dir, tone_category)
    ensure_directory_exists(tone_dir)
    
    # Save aggregate results
    aggregated = {}
    for problem_id, problem_metrics in results['results'].items():
        problem_summary = {
            'problem_id': problem_id,
            'solution_count': len(problem_metrics),
            'avg_metrics': {},
            'solutions': []
        }
        
        # Collect metrics that can be averaged
        numeric_metrics = {}
        for solution in problem_metrics:
            for key, value in solution.items():
                if key not in ['problem_id', 'problem_title', 'solution_idx', 'code', 'graded', 'metadata',
                              'difficulty', 'platform', 'tone_category', 'model_name']:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(value)
        
        # Calculate averages
        for key, values in numeric_metrics.items():
            if values:
                problem_summary['avg_metrics'][key] = sum(values) / len(values)
        
        # Add solution summaries
        for solution in problem_metrics:
            solution_summary = {k: v for k, v in solution.items() if k != 'code'}
            problem_summary['solutions'].append(solution_summary)
        
        aggregated[problem_id] = problem_summary
    
    # Save to file
    with open(os.path.join(tone_dir, 'metrics_results.json'), 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2)
    
    # Save detailed metrics for each problem
    for problem_id, problem_metrics in results['results'].items():
        problem_dir = os.path.join(tone_dir, problem_id)
        ensure_directory_exists(problem_dir)
        
        for solution in problem_metrics:
            solution_idx = solution['solution_idx']
            solution_file = os.path.join(problem_dir, f'solution_{solution_idx}_metrics.json')
            
            with open(solution_file, 'w', encoding='utf-8') as f:
                json.dump(solution, f, indent=2)

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_eval_all_json_files(base_dir: str) -> List[str]:
    """
    Find all eval_all.json files in the LLM model/tone category directory structure.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to eval_all.json files
    """
    eval_all_files = []
    
    # Walk through the directory structure
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
    
    input_path = args.input
    output_dir = args.output
    aggregate_only = args.aggregate_only
    specific_file = args.specific_file
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    all_results = []
    
    if specific_file:
        # Process just the specified file
        if os.path.isfile(specific_file) and specific_file.endswith('.json'):
            print(f"Processing specific file: {specific_file}...")
            results = process_eval_all_json(specific_file)
            all_results.append(results)
            
            if not aggregate_only:
                save_metrics_results(results, output_dir)
                
            print(f"Done processing {specific_file}")
        else:
            print(f"Error: {specific_file} is not a valid JSON file")
            return
    elif os.path.isdir(input_path):
        # Find all eval_all.json files in the directory structure
        eval_all_files = find_eval_all_json_files(input_path)
        
        if not eval_all_files:
            print(f"No eval_all.json files found in {input_path}")
            return
            
        print(f"Found {len(eval_all_files)} eval_all.json files to process")
        
        # Process each eval_all.json file
        for file_path in eval_all_files:
            print(f"Processing {file_path}...")
            results = process_eval_all_json(file_path)
            all_results.append(results)
            
            if not aggregate_only:
                save_metrics_results(results, output_dir)
                
            print(f"Done processing {file_path}")
    else:
        # Process a single file
        if input_path.endswith('.json'):
            print(f"Processing {input_path}...")
            results = process_eval_all_json(input_path)
            all_results.append(results)
            
            if not aggregate_only:
                save_metrics_results(results, output_dir)
                
            print(f"Done processing {input_path}")
        else:
            print(f"Error: {input_path} is not a valid directory or JSON file")
            return
    
    # Generate and save the aggregated results
    if all_results:
        print("Generating aggregated metrics report...")
        aggregated_metrics = aggregate_metrics(all_results)
        
        # Save the aggregated metrics to a single file
        aggregated_file = os.path.join(output_dir, 'aggregated_metrics.json')
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_metrics, f, indent=2)
        print(f"Aggregated metrics saved to {aggregated_file}")
    else:
        print("No results to aggregate")

if __name__ == "__main__":
    main()