import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Goal, TextGenerationConfig, Persona, Prompt, Insight

SYSTEM_INSTRUCTIONS_GENERAL = """
You are an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. 
The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate) AND PERFORM THE APPROPRIATE AGGREGATIONS (e.g., if the property "groupable" is true for a column, then you MUST INCLUDE GROUP BY(COLUMN)). 
They must also be relevant to the specified persona. 
Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization and why the visualization was chosen). 
Each goal MUST mention the exact fields from the dataset summary below.
"""

SYSTEM_INSTRUCTIONS_INSIGHT = """
You are an experienced data analyst who can generate a given number of insightful GOALS ABOUT INSIGHTS about the data to explore the INSIGHT deeper and make meaningful connections between the inisight and the data. 
The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate) AND PERFORM THE APPROPRIATE AGGREGATIONS (e.g., if the property "groupable" is true for a column, then you MUST INCLUDE GROUP BY(COLUMN)). 
They must also be relevant to the specified persona, always be related to the insight AND always only contain fields/columns from the summary provided. 
Each goal must include a question (THE QUESTION MUST BE RELATED TO THE INSIGHT), a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED, what we will learn from the visualization AND how exploring this goal can reveal more insights from the initial insight). 
Each goal MUST mention the exact fields from the dataset summary below.
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

    def calculate_distribution(self, summary: dict, n: int, explore: list):
        """Calculates the distribution of goals related to each category
            - Category: Goals that must explore a column with a category data type
            - Date: Goals that must explore a column with a date data type
            - Number: Goals that must explore a column with number data type
            - Three: Goals that must explore three variables
            - Two: Miscellaneous goals that just include 2 variables
        """
        
        dist = {'none': n}
        # dtypes = set()

        # """Collect the data types from the summary"""
        # for field in summary['fields']:
        #     dtypes.add(field['properties']['dtype'])
        
        # if ('category' in explore or 'string' in explore) and ('category' in dtypes or 'string' in dtypes):
        #     dist['category'] = n // (len(explore) + 1)
        # if 'date' in explore and 'date' in dtypes:
        #     dist['date'] = n // (len(explore) + 1)
        # if 'number' in explore and 'number' in dtypes:
        #     dist['number'] = n // (len(explore) + 1)   
        # if 'three' in explore:
        #     dist['three'] = n // (len(explore) + 1)   
        
        for feature in explore:
            dist[feature] = n // (len(explore) + 1)   
            dist['none'] -= dist[feature]
        
        # dist['none'] = n - dist['category'] - dist['date'] - dist['number'] - dist['three']

        return dist
    
    def generate_goals(self, summary: dict, textgen_config: TextGenerationConfig, 
                        text_gen: TextGenerator, n: int, persona: Persona, 
                        # focus: str,
                        explore: list[str],
                        insights: list[Insight] = []) -> list[Goal]:
        
        if n == 0:
            return []
        
        # Prompt: add focus for data type
        user_prompt = f"""Generate a TOTAL of {n} goals."""
        if explore != []:
            user_prompt = f"""The goals MUST FOCUS on the following variables: '{" ".join(str(x) for x in explore)}'"""
        
        # Prompt: add  summary
        user_prompt += f"\nThe goals should be based on the data summary below, \n\n{summary}\n\n"

        # Insight Goals Prompt: add insights to the prompt if there are insights
        if insights != []:
            user_prompt += f"\nThese are insights about the data: \n"

            # FOR EACH INSIGHT, ADD THE GOAL AND THE QA PAIRS
            for i in range(len(insights)):
                print(insights[i])
                user_prompt += f"""Insight {i + 1}: {insights[i].insight}
                """

        # Define persona
        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")
        
        # Prompt: add persona
        user_prompt += f"\nThe generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona, who is interested in complex, insightful goals about the data.\n"

        # Insight Goals Prompt: add specifications of a good goal for insights
        if insights != []:
            user_prompt += f"""
            These are qualities of a good goal that relates to an insight.

            Question
            - It explores a specific aspect of the insight. It tries to find reasons that cause the insight (e.g. "How does the average x of y (specific from insight) compare to others when controlling z?"). 
            - It explores how multiple variables lead to that insight. (e.g. "How does x (from insight) with type y compare to others when controlling the variable z?", "Is there a relationship between x and y, and how does it affect z?")
            - It is NOT a general question (e.g. NOT "How does the averege x vary with different types?").
            - It NEVER asks anything that cannot be answered from the columns in the given summary.
            - It MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY.
            
            Visualization
            - IT MUST REFERENCE THE EXACT COLUMN FIEDLS FROM THE SUMMARY
            - It includes and aggregations and groupings if the groupable property in the summary is true. 

            Rationale
            - It is able to explain why it is crucial. (e.g. "This is crucial because x impacts y.")
            - It is able to provide a hypothesis (e.g. "Using this visualization will show if x is typically y compared to others")
            - It references a part of the insight (e.g. "The visualization will help us see if your insight is true or valid", "This will reveal how x affects your insight y.")
            
            If there is more than one insight, I want you to generate {n} goals that connect ALL of the insights.
            """

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS_INSIGHT},
                {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} goals are:\n"}
            ]

        elif insights == []:
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS_GENERAL},
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
                text_gen: TextGenerator, n=5, persona: Persona = None,
                insights: list[Insight] = [], explore: list = [],
                ) -> list[Goal]:
        
        """Generate goals given a summary of data"""

        # dist = self.calculate_distribution(summary=summary, n=n, explore=explore)
        goals = []
        # explore.append('none')

        # for focus in explore:
        #     """Generate goals given a focus type"""
        #     print(focus, dist[focus])
        #     goals += self.generate_goals(summary=summary, insights=insights, textgen_config=textgen_config, text_gen=text_gen, n=dist[focus], persona=persona, focus=focus)
        
        goals += self.generate_goals(summary=summary, insights=insights, textgen_config=textgen_config, text_gen=text_gen, n=n, persona=persona, explore=explore)

        # Fix the indexing of the goals
        for i in range(len(goals)):
            goals[i].index = i

        return goals