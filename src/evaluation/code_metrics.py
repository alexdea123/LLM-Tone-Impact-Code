import ast
import io
import sys
import os
import re
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Union
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