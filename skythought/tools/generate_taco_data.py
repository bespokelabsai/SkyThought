from bespokelabs import curator
from datasets import load_dataset
import json
from util.taco.testing_util import run_test as taco_run_test
import copy
import multiprocessing
import numpy as np
from multiprocessing import Manager
import re


import os
os.environ['DEEPSEEK_API_KEY'] = 'sk-df93165a277a4722934b444d2111015b'


SKY_T1_FIXED = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process 
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of 
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered 
thinking process.
"""

# Global counter for tracking "done" occurrences
done_counter = 0

class TACOCurator(curator.LLM):
    return_completions_object = True

    @staticmethod
    def generate_prompt(test_case, prompt, starter_code=None):
        _input = ""
        data = test_case
        if not data.get("fn_name"):
            _input += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."# "\nUse Standard Input format"#\n"
        else:
            _input += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution." #"\nUse Call-Based format"#\n"
        data = prompt
        _input += data
        if starter_code != None:
            data = starter_code
            data = "\n" + data #+ "\n"
            _input += data
        else:
            #_input += "\n\n"
            pass
        
        return _input
    
    @staticmethod
    def has_code(response):
        pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
        # Use re.DOTALL to match multiline content inside backticks
        matches = re.findall(pattern, response, re.DOTALL)
        # print(matches)
        return matches

    def check_correctness(self, problem, generation):
        TIME_OUT = 10
        def _temp_run(problem, generation, debug, result):
            try:
                result.append(taco_run_test(problem, test=generation, debug=debug))
            except Exception as e:
                print(f"Error in _temp_run: {e}")

        manager = Manager()
        result = manager.list()
        p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result))
        p.start()
        p.join(timeout=TIME_OUT + 1)
        if p.is_alive():
            p.kill()
        return bool(result and np.all(result[0]))
    
    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        code_filter_result = self.has_code(response)
        if len(code_filter_result) == 0:
            response_entry["correctness"] = False
            response_entry["reason"] = "Does not contain code component."
        else:	
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(problem)
            curr_res = self.check_correctness(problem_to_check, generation=last_code)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is incorrect."
        
        return response_entry


    def prompt(self, problem):

        test_case = json.loads(problem["input_output"])
        starter_code = problem["starter_code"]
        prompt_text = self.generate_prompt(test_case, problem["question"], starter_code)
        return [
            {"role": "system", "content": SKY_T1_FIXED},
            {"role": "user", "content": prompt_text}
        ]
    
    def parse(self, input, response):
        global done_counter
        input['reasoning'] = response["choices"][0]["message"]["reasoning_content"]
        input['deepseek_solution'] = response["choices"][0]["message"]["content"]
        response_entry = self.update_results(input, input['deepseek_solution'])
        input['correctness'] = response_entry["correctness"]
        input['reason'] = response_entry["reason"]
        done_counter += 1
        print(f"done ({done_counter})")
        return input

llm = TACOCurator(
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

from datasets import Dataset

if __name__ == "__main__":

    taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    # import pdb; pdb.set_trace()
    print("LENGTH", len(taco_dataset))    
    # taco_dataset = taco_dataset.select(range(50))
    response = llm(taco_dataset)

    #push to huggingface hub
    response.push_to_hub("bespokelabs/sky-t1-taco-train-rejection-sampled-v1", private=True)
    import pdb; pdb.set_trace()