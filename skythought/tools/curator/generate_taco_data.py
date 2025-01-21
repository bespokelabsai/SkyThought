from bespokelabs import curator
from datasets import load_dataset
import json
import argparse
import resource

from prompt_util import generate_prompt
from code_execution_taco import process_dataset_parallel

SKY_T1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process 
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of 
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered 
thinking process.
"""

# import os
# os.environ['DEEPSEEK_API_KEY'] = ''

class TACOCurator(curator.LLM):
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
    args.add_argument('--split', type=str, default="train")
    args.add_argument('--dataset_name', type=str, default="bespokelabs/sky-t1-taco")
    args = args.parse_args()

    #push to huggingface hub
    curator = TACOCurator(
        model_name="deepseek-reasoner",
        backend_params={
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30*60,
        },
        generation_params={
            "temp": 0.0,
            "max_tokens": 8192,
        },
    )

    if args.split == "train":
        taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    else:
        taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)[args.split]

    curated_dataset = curator(taco_dataset)

    if args.push_to_hub:
        curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-unfiltered", private=True)
    
    # rejection sampling
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    rejection_sampled_dataset = process_dataset_parallel(curated_dataset)

    if args.push_to_hub:
        rejection_sampled_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-rejection-sampled", private=True)