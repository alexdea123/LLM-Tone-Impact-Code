import os
import json
import time
import random
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

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
        
        for problem in tqdm(problems, desc="Processing problems"):
            problem_id = problem.get("question_id", f"unknown_{len(results)}")
            problem_text = problem.get("question_content", "")  # Use question_content field instead of problem
            
            for category in influence_categories:
                # Get a random prefix from the category
                prefix_index = random.randint(0, len(INFLUENCE_PREFIXES[category]) - 1) if category != InfluenceCategory.NEUTRAL else 0
                prefixed_prompt = get_prompt_with_prefix(problem_text, category, prefix_index)  # Include problem text in the prompt
                
                # Generate code using the API
                response = self.api_client.generate_code(prefixed_prompt, temperature=temperature)
                raw_output = self.api_client.extract_code_from_response(response)
                
                # Extract code from the text
                extracted_code = extract_code_from_text(raw_output)
                
                # Calculate metrics for the generated code
                metrics = calculate_metrics(extracted_code)
                
                # Save results
                result = {
                    "problem_id": problem_id,
                    "influence_category": category.value,
                    "prefix_index": prefix_index,
                    "prompt": prefixed_prompt,  # Save the prompt used
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
        
        # Save all results to a single JSON file
        self._save_results(results)
        
        # Create a DataFrame for analysis
        self._create_analysis_dataframe(results)
        
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
            
            row = {
                "problem_id": result["problem_id"],
                "influence_category": result["influence_category"],
                "prefix_index": result.get("prefix_index", 0),
            }
            
            # Add all metrics to the row
            for metric_name, metric_value in metrics.items():
                row[f"metric_{metric_name}"] = metric_value
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, "analysis_data.csv"), index=False)
        
        return df


# Example usage (comment out in production)
'''
if __name__ == "__main__":
    runner = ExperimentRunner(
        dataset_path="path/to/dataset.json",
        model_name="llama3-8b-8192",  # Use 8B for testing
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    results = runner.run(
        influence_categories=[InfluenceCategory.NEUTRAL, InfluenceCategory.POLITE],
        num_samples=2,  # Small number for testing
        temperature=0.2,
        delay_between_calls=1
    )
    
    print(f"Experiment completed. Results saved to {runner.output_dir}")
'''
