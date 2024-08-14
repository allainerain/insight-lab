import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.datamodel import Goal, Prompt, Insight

SYSTEM_PROMPT = """
You are a an experienced data analyst who can generate a given number of meaningful AND creative insights that people may miss at a first glance about a chart, given the goal of the data visualization and a series of questions answered by a user. 
\nBased on these questions, I want you to generate insights that make connections between the answers that the user gave. I want you to go beyond just describing the data and try to make connections and create hypothesis for why the data appears to be that certain way.
\nThen, I want you to list down the specific prompts and answers that you used that led to those insights. Do it in order of ascending index. 
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "insight": "The x could indicate ...", "prompts": ["What is the...?", "How does the...?", ...], "answers": ["It looks like the...", "There is a peak...", ...] }
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")

class InsightExplorer(object):
    """Generate insights given some answers to questions"""

    def __init__(self) -> None:
        pass

    def generate(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, n=5):
        """Generate questions to prompt the user to interpret the chart given some code and goal"""

        user_prompt = f"""
        Here are the questions and the answers to those questions:
        """

        # ADD THE USER ANSWERS AND QUESTIONS
        for i in range(len(prompts)):
            user_prompt += f"""
            \n\n Question {prompts[i].index + 1}: {prompts[i].question}
            \n Answer: {answers[i]}
            """

        user_prompt += f"""
        \nThis is the goal of the user:
        \nQuestion: {goal.question}
        \nVisualization: {goal.visualization}
        \nRationale: {goal.rationale}
        \nCan you generate insights from the user's answers that draws connections between them?
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} questions are:\n"}
        ]

        result: list[Insight] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            result = clean_code_snippet(result.text[0]["content"])
            result = json.loads(result)
            
            # cast each item in the list to a Prompt object
            if isinstance(result, dict):
                result = [result]
            result = [Insight(**x) for x in result]
        except Exception as e:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        return result