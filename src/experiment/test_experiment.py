import os
import sys
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import argparse
from src.experiment.run_experiment import ExperimentRunner
from src.experiment.prompt_categories import InfluenceCategory

def parse_args():
    parser = argparse.ArgumentParser(description="Run LiveCodeBench experiment with tone influence")
    parser.add_argument("--model", type=str, default="llama3-8b-8192", help="Model name to use")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of problems to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--delay", type=int, default=1, help="Delay between API calls in seconds")
    parser.add_argument("--all_categories", action="store_true", help="Use all influence categories")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load the LiveCodeBench dataset using HuggingFace data loader
    print(f"Loading LiveCodeBench dataset (code_generation_lite)...")
    dataset_split = f"test[:{args.num_samples}]" if args.num_samples else "test"
    dataset = load_dataset("livecodebench/code_generation_lite", split=dataset_split)
    dataset_path = os.path.join(os.path.dirname(__file__), 'livebench_subset.json')

    # Convert dataset to a list of records using pandas
    data_list = dataset.to_pandas().to_dict(orient="records")
    with open(dataset_path, "w") as f:
        json.dump(data_list, f)
    print(f"Saved {len(data_list)} problems to {dataset_path}")

    # Read API key from GROQ_KEY.txt
    key_path = os.path.join(os.path.dirname(__file__), '..', '..', 'GROQ_KEY.txt')
    with open(key_path, 'r') as f:
        api_key = f.read().strip()

    # Select influence categories
    if args.all_categories:
        influence_categories = list(InfluenceCategory)
    else:
        influence_categories = [
            InfluenceCategory.NEUTRAL, 
            InfluenceCategory.POLITE
        ]
    
    category_names = ", ".join([cat.value for cat in influence_categories])
    print(f"Running experiment with influence categories: {category_names}")
    
    runner = ExperimentRunner(
        dataset_path=dataset_path,
        model_name=args.model,
        api_key=api_key
    )
    
    results = runner.run(
        influence_categories=influence_categories,
        num_samples=None,  # Already filtered in dataset selection
        temperature=args.temperature,
        delay_between_calls=args.delay
    )
    
    print(f"Experiment completed. Results saved to {runner.output_dir}")