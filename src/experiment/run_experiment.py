import os
import json
import time
import random
import subprocess
import ast
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import importlib.util

from src.experiment.groq_api import GroqAPI
from src.experiment.prompt_categories import InfluenceCategory, get_prompt_with_prefix, INFLUENCE_PREFIXES
from src.processing.code_extractor import extract_code_from_text
from src.evaluation.code_metrics import calculate_metrics

class ExperimentRunner:
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = None,
        model_name: str = "llama3-70b-8192",
        api_key: str = None,
        seed: int = 42
    ):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_path: Path to the dataset file (JSON format)
            output_dir: Directory to save experiment results
            model_name: Name of the model to use
            api_key: Groq API key
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir or os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.model_name = model_name
        self.api_client = GroqAPI(api_key=api_key, model=model_name)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Load dataset
        self.problems = self._load_dataset()
    
    def _extract_test_cases(self, problem_data: Dict) -> List[Dict]:
        """
        Extract test cases from problem data.
        
        Args:
            problem_data: Problem data from the dataset
            
        Returns:
            List of dictionaries containing test case information
        """
        test_cases = []
        
        # Try to extract test cases from public_test_cases
        try:
            if "public_test_cases" in problem_data:
                print(f"Found public_test_cases field")
                public_tests = json.loads(problem_data["public_test_cases"])
                print(f"Loaded {len(public_tests)} public test cases")
                for test in public_tests:
                    input_data = test.get("input", "")
                    expected_output = test.get("output", "")
                    test_type = test.get("testtype", "functional")
                    test_cases.append({
                        "input": input_data,
                        "expected_output": expected_output,
                        "test_type": test_type
                    })
        except Exception as e:
            print(f"Error parsing public test cases: {e}")
        
        # Get starter code and metadata to understand the expected function signature
        starter_code = problem_data.get("starter_code", "")
        metadata = {}
        if "metadata" in problem_data:
            try:
                metadata = json.loads(problem_data["metadata"])
            except:
                print(f"Warning: Could not parse metadata")
        
        return test_cases, starter_code, metadata
    
    def _extract_function_name(self, code: str, starter_code: str = None, metadata: Dict = None) -> Tuple[str, bool]:
        """
        Extract the function name from the code.
        
        Args:
            code: The generated code
            starter_code: Optional starter code that might contain the function name
            metadata: Metadata that might contain the function name
            
        Returns:
            Tuple of (function_name, is_class_method)
        """
        # First try to get the name from metadata
        if metadata and "func_name" in metadata:
            func_name = metadata["func_name"]
            print(f"Function name from metadata: {func_name}")
            return func_name, "class Solution" in code
            
        # Try to find the function name from the code
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    print(f"Found function: {node.name}")
                    return node.name, False
                elif isinstance(node, ast.ClassDef) and node.name == "Solution":
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            print(f"Found method: {child.name}")
                            return child.name, True
        except:
            pass
        
        # If starter code is provided, try to extract from there
        if starter_code:
            try:
                tree = ast.parse(starter_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == "Solution":
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef):
                                print(f"Found method in starter code: {child.name}")
                                return child.name, True
            except:
                pass
            
        # Default to "solution" if nothing else works
        print("Using default function name 'solution'")
        return "solution", False
    
    def _prepare_test_code(self, code: str, test_input: str, expected_output: str, 
                          starter_code: str = None, metadata: Dict = None) -> str:
        """
        Prepare code for testing with the given input and expected output.
        
        Args:
            code: The generated code
            test_input: The test input
            expected_output: The expected output
            starter_code: Optional starter code
            metadata: Additional metadata
            
        Returns:
            Code prepared for testing
        """
        # Clean the test input and expected output
        test_input = test_input.replace('\\n', '\n').replace('\\"', '"')
        expected_output = expected_output.strip()
        
        # Parse the inputs
        inputs = test_input.strip().split('\n')
        inputs = [inp.strip('"') if inp.startswith('"') and inp.endswith('"') else inp for inp in inputs]
        
        # Extract function name
        func_name, is_class_method = self._extract_function_name(code, starter_code, metadata)
        
        # Prepare the test code
        test_code = code + "\n\n"
        test_code += "# Test code\n"
        test_code += "try:\n"
        
        if is_class_method or "class Solution" in code:
            test_code += "    solution = Solution()\n"
            # Format inputs properly for the function call
            formatted_inputs = []
            for inp in inputs:
                if inp.startswith('[') and inp.endswith(']'):
                    formatted_inputs.append(inp)  # Already a list
                elif inp.startswith('"') and inp.endswith('"'):
                    formatted_inputs.append(inp)  # Already a string
                else:
                    # Try to interpret as a number or keep as string
                    try:
                        float(inp)  # Check if it's a number
                        formatted_inputs.append(inp)
                    except:
                        formatted_inputs.append(f'"{inp}"')
                        
            test_code += f"    result = solution.{func_name}({', '.join(formatted_inputs)})\n"
        else:
            # For standalone functions, similar formatting
            formatted_inputs = []
            for inp in inputs:
                if inp.startswith('[') and inp.endswith(']'):
                    formatted_inputs.append(inp)
                elif inp.startswith('"') and inp.endswith('"'):
                    formatted_inputs.append(inp)
                else:
                    try:
                        float(inp)
                        formatted_inputs.append(inp)
                    except:
                        formatted_inputs.append(f'"{inp}"')
                        
            test_code += f"    result = {func_name}({', '.join(formatted_inputs)})\n"
            
        # Check the result against expected output
        test_code += f"    expected = {expected_output}\n"
        test_code += f"    print(f\"Result: {{result}}\")\n"
        test_code += f"    print(f\"Expected: {{expected}}\")\n"
        test_code += f"    passed = result == expected\n"
        test_code += f"    print(f\"Test passed: {{passed}}\")\n"
        test_code += "except Exception as e:\n"
        test_code += "    print(f\"Error: {e}\")\n"
        test_code += "    passed = False\n"
        
        return test_code

    def run_livebench_evaluation(self, problem_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run LiveCodeBench evaluation on generated outputs by using the dataset.
        
        Args:
            problem_outputs: List of dictionaries containing problem IDs and generated code
            
        Returns:
            Dictionary containing LiveCodeBench evaluation results
        """
        # Group outputs by problem ID (for multiple codes per problem)
        problem_code_map = {}
        for output in problem_outputs:
            problem_id = output["problem_id"]
            if problem_id not in problem_code_map:
                problem_code_map[problem_id] = []
            problem_code_map[problem_id].append({
                "code": output["extracted_code"],
                "influence_category": output["influence_category"],
                "prefix_index": output["prefix_index"]
            })
        
        # Load the original dataset to get test cases
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Create a mapping from problem_id to problem data
        problem_map = {problem.get("question_id", ""): problem for problem in dataset}
        
        # Create a dictionary to store the evaluation results
        evaluation_results = {}
        
        # Evaluate each problem
        for problem_id, code_entries in problem_code_map.items():
            print(f"Evaluating problem {problem_id}...")
            
            if problem_id not in problem_map:
                print(f"Warning: Problem {problem_id} not found in the dataset")
                continue
            
            problem_data = problem_map[problem_id]
            
            # Extract test cases, starter code, and metadata
            test_cases, starter_code, metadata = self._extract_test_cases(problem_data)
            print(f"Found {len(test_cases)} test cases")
            
            if not test_cases:
                print(f"Warning: No test cases found for problem {problem_id}")
                continue
            
            # Store results by category and prefix
            category_results = {}
            
            # Evaluate each generated code against test cases
            for entry in code_entries:
                code = entry["code"]
                category = entry["influence_category"]
                prefix_idx = entry["prefix_index"]
                
                print(f"Evaluating code for category {category} prefix {prefix_idx}")
                
                # Initialize results
                passes = []
                test_results = []
                
                # Run each test case
                for i, test_case in enumerate(test_cases):
                    test_input = test_case["input"]
                    expected_output = test_case["expected_output"]
                    
                    print(f"Running test case {i+1}:")
                    print(f"  Input: {test_input}")
                    print(f"  Expected output: {expected_output}")
                    
                    try:
                        # Prepare the test code
                        test_code = self._prepare_test_code(
                            code, 
                            test_input, 
                            expected_output, 
                            starter_code, 
                            metadata
                        )
                        
                        # Save the test code to a temporary file
                        test_file = os.path.join(self.output_dir, f"temp_test_{problem_id}_{category}_{prefix_idx}_{i}.py")
                        with open(test_file, 'w') as f:
                            f.write(test_code)
                        
                        # Run the test
                        result = subprocess.run(
                            ["python", test_file],
                            capture_output=True,
                            text=True,
                            timeout=10  # 10 seconds timeout
                        )
                        
                        # Check if the test passed
                        test_passed = "Test passed: True" in result.stdout
                        test_results.append(test_passed)
                        
                        # Add debug output
                        print(f"  Test {i+1}: {'PASSED' if test_passed else 'FAILED'}")
                        if not test_passed:
                            print(f"  Output: {result.stdout.strip()}")
                            if result.stderr:
                                print(f"  Error: {result.stderr.strip()}")
                        
                        # Clean up
                        os.remove(test_file)
                        
                    except subprocess.TimeoutExpired:
                        print(f"  Test {i+1}: TIMEOUT")
                        test_results.append(False)
                    except Exception as e:
                        print(f"  Test {i+1}: ERROR - {e}")
                        test_results.append(False)
                
                # Calculate pass rate
                all_pass = all(test_results)
                passes.append(1 if all_pass else 0)
                
                # Store the results
                category_key = f"{category}_{prefix_idx}"
                category_results[category_key] = {
                    "pass": 1 if all_pass else 0,
                    "tests_passed": sum(test_results),
                    "total_tests": len(test_results),
                    "test_results": test_results,
                    "influence_category": category,
                    "prefix_index": prefix_idx
                }
                
                print(f"Category {category} prefix {prefix_idx}: pass={all_pass}, " 
                      f"tests_passed={sum(test_results)}/{len(test_results)}")
                
            # Store the results for this problem
            evaluation_results[problem_id] = category_results
        
        return evaluation_results
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the dataset of coding problems."""
        with open(self.dataset_path, 'r') as f:
            print(f"Loading dataset from {self.dataset_path}")
            return json.load(f)
    
    def run(
        self,
        influence_categories: List[InfluenceCategory] = None,
        num_samples: int = None,
        temperature: float = 0.2,
        delay_between_calls: int = 1  # delay in seconds
    ):
        """
        Run the experiment.
        
        Args:
            influence_categories: List of influence categories to test
            num_samples: Number of samples to run (if None, use all problems)
            temperature: Temperature for generation
            delay_between_calls: Delay between API calls in seconds
        """
        if influence_categories is None:
            influence_categories = list(InfluenceCategory)
        
        # If num_samples is specified, take a random sample of problems
        problems = self.problems
        if num_samples and num_samples < len(problems):
            problems = random.sample(problems, num_samples)
        
        results = []
        problem_outputs = []  # Track outputs for LiveCodeBench evaluation
        
        for problem in tqdm(problems, desc="Processing problems"):
            problem_id = problem.get("question_id", f"unknown_{len(results)}")
            problem_text = problem.get("question_content", "")
            
            # Add starter code to the prompt if available
            starter_code = problem.get("starter_code", "")
            if starter_code:
                problem_text = f"{problem_text}\n\nHere is the starter code:\n```python\n{starter_code}\n```"
            
            for category in tqdm(influence_categories, desc=f"Testing influence categories for problem {problem_id}", leave=False):
                # Get a random prefix from the category
                prefix_index = random.randint(0, len(INFLUENCE_PREFIXES[category]) - 1) if category != InfluenceCategory.NEUTRAL else 0
                prefixed_prompt = get_prompt_with_prefix(problem_text, category, prefix_index)
                
                # Generate code using the API
                response = self.api_client.generate_code(prefixed_prompt, temperature=temperature)
                raw_output = self.api_client.extract_code_from_response(response)
                
                # Extract code from the text
                extracted_code = extract_code_from_text(raw_output)
                
                # Store for LiveCodeBench evaluation
                problem_outputs.append({
                    "problem_id": problem_id,
                    "extracted_code": extracted_code,
                    "influence_category": category.value,
                    "prefix_index": prefix_index
                })
                
                # Calculate our custom metrics
                metrics = calculate_metrics(extracted_code)
                
                # Save results
                result = {
                    "problem_id": problem_id,
                    "influence_category": category.value,
                    "prefix_index": prefix_index,
                    "prompt": prefixed_prompt,
                    "raw_output": raw_output,
                    "extracted_code": extracted_code,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Save raw output and extracted code to files
                self._save_result_files(problem_id, category.value, prefix_index, raw_output, extracted_code)
                
                # Add delay to prevent rate limiting
                time.sleep(delay_between_calls)
        
        # Run LiveCodeBench evaluation
        print("Running LiveCodeBench evaluation...")
        livebench_results = self.run_livebench_evaluation(problem_outputs)
        
        # Merge LiveCodeBench results into our results
        if livebench_results:
            for result in results:
                problem_id = result["problem_id"]
                influence_category = result["influence_category"]
                prefix_index = result["prefix_index"]
                
                # Look for the matching result in livebench_results
                if problem_id in livebench_results:
                    category_key = f"{influence_category}_{prefix_index}"
                    if category_key in livebench_results[problem_id]:
                        result["livebench_metrics"] = livebench_results[problem_id][category_key]
        
        # Save all results to a single JSON file
        self._save_results(results)
        
        # Create a DataFrame for analysis
        df = self._create_analysis_dataframe(results)
        
        # Print a summary of the results
        self._print_results_summary(df)
        
        return results
    
    def _save_result_files(
        self,
        problem_id: str,
        category: str,
        prefix_index: int,
        raw_output: str,
        extracted_code: str
    ):
        """Save raw output and extracted code to files."""
        base_dir = os.path.join(self.output_dir, f"problem_{problem_id}", f"category_{category}_prefix_{prefix_index}")
        os.makedirs(base_dir, exist_ok=True)
        
        # Save raw output
        with open(os.path.join(base_dir, "raw_output.txt"), 'w') as f:
            f.write(raw_output)
        
        # Save extracted code
        with open(os.path.join(base_dir, "extracted_code.py"), 'w') as f:
            f.write(extracted_code)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save all results to a JSON file."""
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    def _create_analysis_dataframe(self, results: List[Dict[str, Any]]):
        """Create a DataFrame for analysis and save to CSV."""
        df_data = []
        
        for result in results:
            metrics = result["metrics"] if "metrics" in result else {}
            livebench_metrics = result.get("livebench_metrics", {})
            
            row = {
                "problem_id": result["problem_id"],
                "influence_category": result["influence_category"],
                "prefix_index": result.get("prefix_index", 0),
            }
            
            # Add all custom metrics
            for metric_name, metric_value in metrics.items():
                row[f"metric_{metric_name}"] = metric_value
            
            # Add LiveCodeBench metrics with a prefix
            for metric_name, metric_value in livebench_metrics.items():
                row[f"livebench_{metric_name}"] = metric_value
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, "analysis_data.csv"), index=False)
        
        return df
    
    def _print_results_summary(self, df: pd.DataFrame):
        """Print a summary of the results."""
        print("\n===== EXPERIMENT RESULTS SUMMARY =====")
        print(f"Total problems tested: {df['problem_id'].nunique()}")
        print(f"Influence categories: {df['influence_category'].unique()}")
        
        # Show performance metrics by influence category if available
        if 'livebench_pass' in df.columns:
            print("\nLiveCodeBench Performance Metrics by Influence Category:")
            perf_by_category = df.groupby('influence_category')['livebench_pass'].mean().reset_index()
            for _, row in perf_by_category.iterrows():
                print(f"  {row['influence_category']}: pass rate = {row['livebench_pass']:.4f}")
                
            # Show tests passed information if available
            if 'livebench_tests_passed' in df.columns and 'livebench_total_tests' in df.columns:
                tests_by_category = df.groupby('influence_category').agg({
                    'livebench_tests_passed': 'sum',
                    'livebench_total_tests': 'sum'
                }).reset_index()
                
                for _, row in tests_by_category.iterrows():
                    tests_passed = row['livebench_tests_passed']
                    total_tests = row['livebench_total_tests']
                    print(f"  {row['influence_category']}: tests passed = {tests_passed}/{total_tests} ({tests_passed/total_tests:.2%})")
        
        # Show average cyclomatic complexity by influence category
        if 'metric_avg_cyclomatic_complexity' in df.columns:
            print("\nCode Complexity by Influence Category:")
            complexity_by_category = df.groupby('influence_category')['metric_avg_cyclomatic_complexity'].mean().reset_index()
            for _, row in complexity_by_category.iterrows():
                print(f"  {row['influence_category']}: avg complexity = {row['metric_avg_cyclomatic_complexity']:.2f}")
        
        print("\nResults saved to:", self.output_dir)
