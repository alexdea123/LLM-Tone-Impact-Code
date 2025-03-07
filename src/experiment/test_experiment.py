import os
import sys
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
from src.experiment.run_experiment import ExperimentRunner
from src.experiment.prompt_categories import InfluenceCategory

if __name__ == "__main__":
    # Load the LiveCodeBench dataset using HuggingFace data loader
    dataset = load_dataset("livecodebench/code_generation_lite", split="test[:2]", trust_remote_code=True)  # Load a small subset for testing
    dataset_path = os.path.join(os.path.dirname(__file__), 'livebench_subset.json')

    # Convert dataset to a list of records using pandas
    data_list = dataset.to_pandas().to_dict(orient="records")
    with open(dataset_path, "w") as f:
        json.dump(data_list, f)

    # Read API key from GROQ_KEY.txt
    key_path = os.path.join(os.path.dirname(__file__), '..', '..', 'GROQ_KEY.txt')
    with open(key_path, 'r') as f:
        api_key = f.read().strip()

    runner = ExperimentRunner(
        dataset_path=dataset_path,  # Use the HuggingFace dataset
        model_name="llama3-8b-8192",  # Use 8B for testing
        api_key=api_key  # API key from GROQ_KEY.txt
    )
    
    results = runner.run(
        influence_categories=[
            InfluenceCategory.NEUTRAL, 
            InfluenceCategory.POLITE
        ],
        num_samples=2,  # Small number for testing
        temperature=0.2,
        delay_between_calls=1
    )
    
    print(f"Experiment completed. Results saved to {runner.output_dir}")