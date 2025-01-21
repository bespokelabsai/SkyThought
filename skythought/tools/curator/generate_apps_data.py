from bespokelabs import curator
from datasets import load_dataset
import json
import argparse
from prompt_util import generate_prompt
from code_execution_apps import process_dataset_parallel

# import os
# os.environ['DEEPSEEK_API_KEY'] = ''

SKY_T1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process 
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of 
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered 
thinking process.
"""

class APPSCurator(curator.LLM):
    return_completions_object = True
   
    def prompt(self, problem):

        test_case = json.loads(problem["input_output"])
        starter_code = problem["starter_code"]
        prompt_text = generate_prompt(test_case, problem["question"], starter_code)
        return [
            {"role": "system", "content": SKY_T1_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
    
    def parse(self, input, response):
        input['reasoning'] = response["choices"][0]["message"]["reasoning_content"]
        input['deepseek_solution'] = response["choices"][0]["message"]["content"]
        return input

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--push_to_hub', action='store_true')
    args.add_argument('--split', type=str, default="test")
    args.add_argument('--dataset_name', type=str, default="bespokelabs/sky-t1-apps")
    args = args.parse_args()

    curator = APPSCurator(
        model_name="deepseek-reasoner",
        backend_params={
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30*60,
        },
    )

    apps = load_dataset("codeparrot/apps", trust_remote_code=True)[args.split]
    curated_dataset = curator(apps)

    if args.push_to_hub:
        curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-unfiltered", private=True)

    curated_dataset = process_dataset_parallel(curated_dataset)

    if args.push_to_hub:
        curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-rejection-sampled", private=True)

    print("Done")