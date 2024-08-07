import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.datamodel import Goal, Prompt, Insight

SYSTEM_PROMPT = """
You are a helpful and highly skilled data analyst who is trained to provide helpful, and creative insights and connections based on the answers the user gave to specific questions about a visualization given their goal. 
The INSIGHTS YOU GENERATE MUST BE INSIGHTFUL AND BE MEANINGFUL (e.g., related to the goal). 
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "insight": "The x could indicate ..."} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")

class InsightGenerator(object):
    """Generate insights given some answers to questions"""

    def __init__(self) -> None:
        pass

    def generate(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, n=5):
        """Generate questions to prompt the user to interpret the chart given some code and goal"""

        user_prompt = f"""
        The visualization is: {goal.visualization}. The question the visualization wants to answer is: {goal.question}. The rationale for choosing the visualization is: {goal.rationale}.
        \nHere are the questions about the visualization and the corresponding answers:
        """

        # ADD THE USER ANSWERS AND QUESTIONS
        for i in range(len(prompts)):
            user_prompt += f"""
            \n\n Question {prompts[i].index}: {prompts[i].question}
            \n Answer: {answers[i]}
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
