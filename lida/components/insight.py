import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.datamodel import Goal, Prompt, Insight, Persona

import http.client
import json

SYSTEM_PROMPT = """
You are a an experienced data analyst who can generate a given number of meaningful AND creative insights that people may miss at a first glance about a chart, given the goal of the data visualization and a series of questions answered by a user. 

Each insight MUST have the following:
- A hypothesis or generalization about the data given the question and answer prompts
- An explanation on how you derived the hypothesis or generalization. 
- A web search phrase that will generate RELEVANT search results that will support your hypothesis or generalization. If there is a description of the dataset provided, the web search phrase MUST take into consideration the descriptions/added domain knowledge from the descriptions.
- Must be logical and correct. If the user's answers sound wrong, YOU MUST POINT IT OUT WITH CREDIBLE SOURCES FROM THE WEB.

In a separate part, there must also be a list  of specific prompts and answers that you used that led to those insights. Do it in order of ascending index. 
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "insight": "The x could indicate (rest of insight)", "search_phrase": "What are the...", "evidence": [], "prompts": ["What is the (rest of prompt)?", "How does the (rest of prompt)?", ...], "answers": ["It looks like the (rest of answer)", "There is a peak (rest of answer)", ...] }
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")

class InsightExplorer(object):
    """Generate insights given some answers to questions"""

    def __init__(self) -> None:
        pass

    def search(self, search_phrase: str, api_key: str):
        """Search the web given some search phrase"""

        conn = http.client.HTTPSConnection("google.serper.dev")

        payload = json.dumps({
            "q": search_phrase,
            "num": 5
        })

        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")

        # Parse the JSON response
        search_results = json.loads(data)
        
        # Extract the links from the 'organic' search results
        links = [result['link'] for result in search_results.get('organic', [])]
        
        return links

    def generate(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, persona:Persona = None, n=5, 
            description: dict = {}, api_key: str = "" ):
        """Generate questions to prompt the user to interpret the chart given some code and goal"""

        user_prompt = f"""
        Here are the questions and the answers to those questions:
        """

        # Prompt: Add question and answer pairs
        for i in range(len(prompts)):
            user_prompt += f"""
            \n\n Question {prompts[i].index + 1}: {prompts[i].question}
            \n Answer: {answers[i]}
            """

        # Prompt: Add goal
        user_prompt += f"""
        \nThis is the goal of the user:
        \nQuestion: {goal.question}
        \nVisualization: {goal.visualization}
        \nRationale: {goal.rationale}
        \nCan you generate A TOTAL OF {n} INSIGHTS from the answers that draws connections between them?
        """
        
        # Define persona
        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")
            
        # Prompt: Add persona
        user_prompt += f"\nThe generated insights SHOULD TRY TO BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona, who is interested in complex, insightful insights about the data.\n"

        # Prompt: Add description if applicable
        if description != {}:
            user_prompt += "These are the descriptions of the columns of the dataset. Try to make connections with the descriptions provided below with your hypothesis and the search phrase if it's applicable when generating the insights."
            user_prompt += str(description)
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} questions are:\n"}
        ]

        result: list[Insight] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            result = clean_code_snippet(result.text[0]["content"])
            result = json.loads(result)

            # cast each item in the list to an Insight object
            if isinstance(result, dict):
                result = [result]
            result = [Insight(**x) for x in result]
        except Exception as e:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        if api_key != "":
            """Search Google"""
            for insight in result:
                links = self.search(search_phrase=insight.search_phrase, api_key=api_key)
                insight.evidence = links

        return result