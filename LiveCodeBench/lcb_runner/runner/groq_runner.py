import os
from time import sleep
from groq import Groq  # Using the Groq library

from lcb_runner.runner.base_runner import BaseRunner


class GroqRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable must be set")
        
        # Initialize the Groq client with your API key
        self.client = Groq(api_key=self.api_key)
        self.client_kwargs = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        # Ensure prompt is a list of message dictionaries
        if not isinstance(prompt, list):
            prompt = [{"role": "user", "content": prompt}]

        def __run_single(counter):
            try:
                # Use the Groq library's chat completions API
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                # If it's a client error (HTTP 400), re-raise immediately
                if hasattr(e, "response") and e.response is not None and getattr(e.response, "status_code", None) == 400:
                    raise e
                print("Exception:", repr(e), "Sleeping for", 20 * (11 - counter), "seconds...")
                sleep(20 * (11 - counter))
                counter -= 1
                if counter == 0:
                    print(f"Failed to run model for prompt: {prompt}!")
                    print("Exception:", repr(e))
                    raise e
                return __run_single(counter)

        outputs = []
        for _ in range(self.args.n):
            outputs.append(__run_single(10))
        return outputs
