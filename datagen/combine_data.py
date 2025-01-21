from datasets import load_dataset, concatenate_datasets
import json

SKY_T1_FIXED = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with ' '} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

def map_numina_conversations(x):
    user_message = f"Return your final response within \\boxed{{}}. {x['problem']}"
    assistant_message = f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "content": user_message},
            {"from": "assistant", "content": assistant_message},
        ],
    }

numina_rejection_correct = load_dataset(
    "bespokelabs/sky-t1-numina-rejection-sampled", trust_remote_code=True
)["train"].filter(lambda x: x["correct"]).take(10_500)
numina_conversations = numina_rejection_correct.map(
    map_numina_conversations, remove_columns=numina_rejection_correct.column_names)

def map_apps_conversations(x):
    test_case = json.loads(x["input_output"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."# "\nUse Standard Input format"#\n"
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution." #"\nUse Call-Based format"#\n"
    data = prompt
    user_message += data
    if starter_code != None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"

    return {
        "system": SKY_T1_FIXED,
        "conversations": [
            {"from": "user", "content": user_message},
            {"from": "assistant", "content": assistant_message},
        ],
    }

apps_conversations_correct = load_dataset(
    "bespokelabs/sky-t1-apps-rejection-sampled", trust_remote_code=True
)["train"].filter(lambda x: x["correctness"])
apps_conversations = apps_conversations_correct.map(
    map_apps_conversations, remove_columns=apps_conversations_correct.column_names)

still2 = load_dataset("RUC-AIBOX/long_form_thought_data_5k", trust_remote_code=True)["train"]
still2_columns = still2.column_names
still2 = still2.filter(
    lambda x: x["domain"] in ["puzzle", "physics", "biology", "chemistry"]).remove_columns(still2_columns)

final_dataset = concatenate_datasets([numina_conversations, apps_conversations, still2])

final_dataset.push_to_hub("bespokelabs/stratos-r1-incomplete", private=True)