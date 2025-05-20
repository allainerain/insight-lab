import ast
import base64
import importlib
import io
import os
import re
import traceback
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio

from lida.datamodel import ChartExecutorResponse, Summary
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

def preprocess_code(code: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # # remove all text after chart = plot(data)
    # if "chart = plot(data)" in code:
    #     # print(code)
    #     index = code.find("chart = plot(data)")
    #     if index != -1:
    #         code = code[: index + len("chart = plot(data)")]

    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]
        # code = code.replace("```", "")
        # return code

    if "import" in code:
        # return only text after the first import statement
        index = code.find("import")
        if index != -1:
            code = code[index:]

    if "import pandas as pd" not in code:
        start = """import pandas as pd

df = data.copy()\n\n"""
        code = start + code

    code = code.replace("```", "")
    return code


def get_globals_dict(code_string, data):
    # Parse the code string into an AST
    tree = ast.parse(code_string)
    # Extract the names of the imported modules and their aliases
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = importlib.import_module(alias.name)
                imported_modules.append((alias.name, alias.asname, module))
        elif isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                imported_modules.append(
                    (f"{node.module}.{alias.name}", alias.asname, obj)
                )

    # Import the required modules into a dictionary
    globals_dict = {}
    for module_name, alias, obj in imported_modules:
        if alias:
            globals_dict[alias] = obj
        else:
            globals_dict[module_name.split(".")[-1]] = obj

    ex_dicts = {"pd": pd, "data": data, "plt": plt}
    globals_dict.update(ex_dicts)
    return globals_dict


class DataTransformer:
    """Execute code"""

    def __init__(self) -> None:
        pass

    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        return_error: bool = False,
    ) -> Any:
        """Validate and convert code"""

        # # check if user has given permission to execute code. if env variable
        # # LIDA_ALLOW_CODE_EVAL is set to '1'. Else raise exception
        # if os.environ.get("LIDA_ALLOW_CODE_EVAL") != '1':
        #     raise Exception(
        #         "Permission to execute code not granted. Please set the environment variable LIDA_ALLOW_CODE_EVAL to '1' to allow code execution.")

        if isinstance(summary, dict):
            summary = Summary(**summary)

        df = data
        feedback = None
        code_spec_copy = code_specs.copy()
        code_specs = [preprocess_code(code) for code in code_specs]

        print("executing")
        print(code_spec_copy[0])

        for code in code_specs:
            try:
                ex_locals = get_globals_dict(code, data)
                # print(ex_locals)
                loc = {}
                exec(code, ex_locals, loc)
                df = loc["df"]
                
            except Exception as exception_error:
                print(code_spec_copy[0])
                print("****\n", str(exception_error))
                print(traceback.format_exc())
                feedback = "ERROR\n\n" +  "****\n" + str(exception_error)
                # if return_error:

        hide = '''import pandas as pd\n\ndf = data.copy()'''

        return df, {"feedback": feedback, "code": code_spec_copy[0].replace(hide, '')}


system_prompt = """
You are a high skilled visualization assistant that can modify a provided dataframe based on a set of instructions. You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""

class DataAutoTransformer(object):
    """Generate visualizations from prompt"""

    def __init__(self) -> None:
        pass

    def generate(
            self, instructions: list[str], data: Any, summary: Summary,
            textgen_config: TextGenerationConfig, text_gen: TextGenerator):
        """Edit a code spec based on instructions"""

        instruction_string = ""
        for i, instruction in enumerate(instructions):
            instruction_string += f"{i+1}. {instruction} \n"

        library_template = """ALWAYS INCLUDE EVERYTHING UNLESS SPECIFIED THAT IT MUST BE EDITED.
        df = data.copy()
        df[<stub variable>] = <stub> # only modify this section
        """

        messages = [
            {
                "role": "system", "content": system_prompt}, {
                "role": "system", "content": f"The dataset summary is : \n\n {summary} \n\n"}, {
                "role": "system", "content": f"The modifications you make MUST BE CORRECT and  based on the 'Pandas DataFrames' library. The resulting code MUST use the following template \n\n {library_template} \n\n "}, {
                    "role": "user", "content": f"ALL ADDITIONAL LIBRARIES USED MUST BE IMPORTED.\n YOU MUST THINK STEP BY STEP, MEET EACH OF THE FOLLOWING INSTRUCTIONS: \n\n {instruction_string} \n\n. DO NOT EDIT CODE BEYOND WHAT IS ASKED FOR. \n The completed modified code THAT FOLLOWS THE TEMPLATE above is. \n DO NOT REDEFINE THE VARIABLE 'data'. \n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
