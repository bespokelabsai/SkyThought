
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


