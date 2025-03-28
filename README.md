# LLM-Tone-Impact-Code
RQ1: Does varying the politeness or influence tactic (e.g., polite request vs. threat vs. flattery) in prompts significantly affect the correctness and quality of code generated by large language models?
RQ2: Which of Yukl’s categories of influence (e.g., pressure tactics, inspirational appeals, reciprocity) produce the most notable changes in code performance, structure, and complexity?
RQ3: Which code-quality metrics (e.g., lines of code, cyclomatic complexity) best capture differences arising from varied prompt framing?


In a new conda environment (python=3.13), run:

```
pip install radon pylint groq pandas tqdm
```




## Run command:

python -m lcb_runner.runner.main --model llama3-8b-8192 --scenario codegeneration --n 5 --evaluate --release_version release_v2 --use_cache --start_date 2024-01-01

Make sure you also run:

export GROQ_API_KEY={GROQ API key here}