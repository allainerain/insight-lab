import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Goal, TextGenerationConfig, Persona

SYSTEM_INSTRUCTIONS = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization and why the visualization was chosen). Each goal MUST mention the exact fields from the dataset summary above
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")


class GoalExplorer():
    """Generate goals given a summary of data"""

    def __init__(self) -> None:
        pass

    def calculate_distribution(self, summary: dict, n: int):
        dist = {'category': 0, 'date': 0, 'number': 0, 'three': 0, 'two': 0}
        dtypes = set()

        # collect the data types from the summary
        for field in summary['fields']:
            dtypes.add(field['properties']['dtype'])

        # calculate for the two
        dist['two'] = n // (len(dtypes) + 2) + 1
        remaining = n - dist['two']
        
        if 'category' in dtypes or 'string' in dtypes:
            dist['category'] = n // (len(dtypes) + 1)
        if 'date' in dtypes:
            dist['date'] = n // (len(dtypes) + 1)
        if 'number' in dtypes:
            dist['number'] = n // (len(dtypes) + 1)   

        dist['three'] = remaining - dist['category'] - dist['date'] - dist['number']

        return dist
    
    def generate_goals(self, summary: dict, textgen_config: TextGenerationConfig,
                    text_gen: TextGenerator, n: int, persona: Persona, focus: str) -> list[Goal]:
        
        if n == 0:
            return []
        
        # ADD PROMPT BASED ON TYPE OR VARIABLE NUMBER
        if focus in ['category/string', 'date', 'number']:
            user_prompt = f"""Generate a TOTAL of {n} goals. All the goals MUST FOCUS on a column with a '{focus}' data type."""
        elif focus in ['two', 'three']:
            user_prompt = f"""Generate a TOTAL of {n} goals. All the goals must explore the relationship of EXACTLY {focus} variables. """
        else:
            raise ValueError(f"Unsupported focus type: {focus}. Please provide a valid focus type.")

        # ADD SUMMARY
        user_prompt += f"\nThe goals should be based on the data summary below, \n\n{summary}\n\n"

        # ADD PERSONA
        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")
            
        user_prompt += f"\nThe generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona, who is interested in complex, insightful goals about the data.\n"

        # ARRAY OF MESSAGES
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} goals are:\n"}
        ]

        result: list[Goal] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Goal(**x) for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")
        
        return result
    
    def generate(self, summary: dict, textgen_config: TextGenerationConfig,
                 text_gen: TextGenerator, n=5, persona: Persona = None) -> list[Goal]:
        """Generate goals given a summary of data"""

        dist = self.calculate_distribution(summary=summary, n=n)

        category_goals = self.generate_goals(summary=summary, textgen_config=textgen_config, text_gen=text_gen, n=dist['category'], persona=persona, focus="category/string")
        date_goals = self.generate_goals(summary=summary, textgen_config=textgen_config, text_gen=text_gen, n=dist['date'], persona=persona, focus="date")
        number_goals = self.generate_goals(summary=summary, textgen_config=textgen_config, text_gen=text_gen, n=dist['number'], persona=persona, focus="number")
        three_goals = self.generate_goals(summary=summary, textgen_config=textgen_config, text_gen=text_gen, n=dist['three'], persona=persona, focus="three")
        two_goals = self.generate_goals(summary=summary, textgen_config=textgen_config, text_gen=text_gen, n=dist['two'], persona=persona, focus="two")

        all_goals = category_goals + date_goals + number_goals + three_goals + two_goals
        
        #Fixing the indexing
        for i in range(len(all_goals)):
            all_goals[i].index = i

        return all_goals