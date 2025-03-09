import subprocess
import os
import time
from datetime import datetime
from lcb_runner.prompts.prompt_categories import InfluenceCategory


def setup_groq_key():
    """Read and set up the Groq API key from GROQ_KEY.txt"""
    try:
        with open("GROQ_KEY.txt", "r") as f:
            groq_key = f.read().strip()
        os.environ["GROQ_API_KEY"] = groq_key
        print("Successfully loaded Groq API key")
    except FileNotFoundError:
        raise FileNotFoundError("GROQ_KEY.txt not found. Please create this file with your Groq API key.")
    except Exception as e:
        raise Exception(f"Error setting up Groq API key: {str(e)}")

def main():
    setup_groq_key()

    # Base command without the category
    base_cmd = [
        "python", "-m", "lcb_runner.runner.main",
        "--model", "llama3-70b-8192",
        "--scenario", "codegeneration",
        "--evaluate",
        "--release_version", "release_v3",
        "--use_cache",
        "--n", "5",
        "--start_date", "2024-01-01",
        "--end_date", "2024-01-30"
        # "--end_date", "2024-12-31"
    ]
        
    # Get all categories from the enum
    categories = [category.value for category in InfluenceCategory]
    
    # Loop through each category and run the command
    for category in categories:
        print(f"\n\n{'='*50}")
        print(f"Running benchmark with tone category: {category}")
        print(f"{'='*50}\n")
        
        # Build the command for this category
        cmd = base_cmd + ["--tone_category", category]
        
        # Run the command
        try:            
            # Run the command with real-time output
            process = subprocess.run(
                cmd,
                stdout=None,  # This will show output in real-time
                stderr=None,  # This will show errors in real-time
                text=True
            )
            
            print(f"\nCompleted run for category: {category}")
            
        except Exception as e:
            print(f"Error running benchmark for category {category}: {str(e)}")
        
        # Add a small delay between runs to avoid rate limiting
        time.sleep(5)
    
if __name__ == "__main__":
    main()