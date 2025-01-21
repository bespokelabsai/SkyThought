from bespokelabs import curator
from datasets import load_dataset
import re

from skythought.tools.util.math.testing_util import strip_answer_string, extract_answer, math_equal


SKY_T1_FIXED = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process 
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of 
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered 
thinking process.

Return your final response within \\boxed{{}}.
"""


def extract_boxed_answer(text):
    text = strip_answer_string(text)
    return extract_answer(text)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        return [
            {"role": "system", "content": SKY_T1_FIXED},
            {"role": "user", "content": input["problem"]},
        ]

    def parse(self, input, response):
        return [
            {
                "problem": input["problem"],
                "reasoning": response["choices"][0]["message"]["reasoning_content"],
                "deepseek_solution": response["choices"][0]["message"]["content"],
                "ground_truth_solution": input["solution"],
                "deepseek_final_answer": strip_answer_string(response["choices"][0]["message"]["content"]),
                "ground_truth_final_answer": strip_answer_string(input["solution"]),
            }
        ]

# amc_aime
numina_162k_amc_aime_problems = (
    load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)[
        "train"
    ]
    .filter(lambda x: x["source"] == "amc_aime")
)
llm = Reasoner(
    model_name="deepseek-reasoner",
    generation_params={"temp": 0.0},
    backend_params={
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 100_000_000,
    },
)
response = llm(numina_162k_amc_aime_problems)
response.push_to_hub("bespokelabs/sky-t1-numina-amc-aime-subset-unfiltered", private=True)

# math
numina_162k_math_problems = (
    load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)[
        "train"
    ]
    .filter(lambda x: x["source"] == "math")
)
response = llm(numina_162k_math_problems)
response.push_to_hub("bespokelabs/sky-t1-numina-math-subset-unfiltered", private=True)

# olympiads
numina_162k_math_problems = (
    load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)[
        "train"
    ]
    .filter(lambda x: x["source"] == "olympiads")
).take(20_000)
response = llm(numina_162k_math_problems)
response.push_to_hub("bespokelabs/sky-t1-numina-olympiads-subset-unfiltered", private=True)

# # # Filter correct answers and calculate accuracy
# correct_answers = response.filter(lambda x: x["ground_truth_final_answer"] == x["deepseek_final_answer"], num_proc=10)
# total_questions = len(response)
# correct_count = len(correct_answers)
# accuracy = correct_count / total_questions

# print(f"Accuracy: {correct_count}/{total_questions} ({accuracy:.2%})")

# # Optionally print all answers for inspection
# for i, item in enumerate(response.take(20)):
#     print(f"\nQuestion {i+1}:")
#     # print(f"DeepSeek Reasoning: {item['reasoning']}")
#     # print(f"DeepSeek Solution: {item['deepseek_solution']}")
#     print(f"DeepSeek Answer: {item['deepseek_final_answer']}")
#     print(f"Ground Truth: {item['ground_truth_final_answer']}")
#     print(f"Correct: {'✓' if item['deepseek_final_answer'] == item['ground_truth_final_answer'] else '✗'}")