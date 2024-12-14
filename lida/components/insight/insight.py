import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.datamodel import Goal, Prompt, Insight, Persona, Research
from ..insight.webscraper import WebScraper
from ..insight.retrieval import EmbeddingRetriever

import http.client
import json

SYSTEM_PROMPT = """
You are a an experienced data analyst who can generate a given number of meaningful AND creative insights that people may miss at a first glance about a chart, given the goal of the data visualization, a series of questions answered by a user. MAKE AN INSIGHT BASED ON THE REFERENCES.

Each insight MUST have the following:
- A hypothesis about the data given the question and answer prompts.
- Should be creative, complex, specific and unexpected and be multi-dimensional.
- State specific points from the web search results in the insight.
- An explanation on how you derived the hypothesis or generalization from the web search and from your own knowledge. 
- Make sure to cite results using [number] notation after the quote, MUST CITE THE CORRECT EVIDENCE BASED ON THE EVIDENCE LIST.
- Must be logical and correct. If the user's answers sound wrong, YOU MUST POINT IT OUT WITH CREDIBLE SOURCES FROM THE WEB.

"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
        { 
            "index": 0,  
            "insight": "The x could indicate (rest of insight) because of some reason [1] and some other reason [2]", 
            "evidence": {
                "1": ["URL", "Quoted Sentence"], 
                "2": ["URL", "Quoted Sentence"]
            }
        }
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE. Make sure that the JSON format is free from any errors. Any quotes within strings need to be escaped with a backslash (\").
"""

SYSTEM_PROMPT_SEARCH ="""
You are a helpful and highly skilled data analyst who is trained to find the most relevant resources and references to support certain observations of a data visualization. 

You must generate search phrases that would result in the most relevant web results that would explain or support certain observations about the visualization.

The search phrases must be:
1. Relevant to an observation about the visualization
2. Specific enough to confirm the answers given to the questions
3. Diverse enough from each other such that it would not cause overlapping search results
"""

FORMAT_INSTRUCTIONS_SEARCH = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF STRINGS (the search phrases). IT MUST USE THE FOLLOWING FORMAT:

```["What are the...", "Most popular..."]
```

THE OUTPUT SHOULD ONLY USE THE LIST FORMAT ABOVE.
"""

SYSTEM_PROMPT_RA = """
You are a helpful and highly skilled data analyst who is trained to provide helpful, prompting questions to guide the user to gain insights from a data visualization given their goal, additional references, and GIVEN THEIR ANSWERS TO QUESTIONS. 

The questions you ask must be the following
- Incite insightful ideas and be meaningful.
- Be related to the goal, the visualization given, the provided references, AND THE USER ANSWERS.
- Clarify domain knowledge of the data ONLY if necessary. If you do this, add an example answer to guide the user.

"""

FORMAT_INSTRUCTIONS_RA = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
        { 
            "index": 0,  
            "question": "prompting question", 
            "evidence": {
                "1": ["URL", "Quoted Sentence"], 
                "2": ["URL", "Quoted Sentence"]
            }
        }
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE. Make sure that the JSON format is free from any errors. Any quotes within strings need to be escaped with a backslash (\").
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
            "num": 2
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
    
    def generate_search_phrases(self, goal: Goal, answers: list[str], prompts: Prompt,
                                textgen_config: TextGenerationConfig, text_gen: TextGenerator, n=5):
        user_prompt = f"""
        \nThis is the visualization:
        \nQuestion: {goal.question}
        \nVisualization: {goal.visualization}
        \nRationale: {goal.rationale}        

        \nHere are the questions and the answers regarding the visualization, which are the observations:
        """

        # Prompt: Add question and answer pairs
        for i in range(len(prompts)):
            user_prompt += f"""
            \n\n Question {prompts[i].index + 1}: {prompts[i].question}
            \n Answer: {answers[i]}
            """

        user_prompt += f"""
        Build a summary from the given answers and questions about the visualization. From the summary, generate a total of {n} search phrases.
        """

        print(user_prompt)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SEARCH},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS_SEARCH}\n\nThe generated {n} search phrases are:\n"}
        ]

        result: list[Insight] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            result = clean_code_snippet(result.text[0]["content"])
            result = json.loads(result)
        except Exception as e:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid LIST object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        return result
    
    def generate(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, persona:Persona = None, n=5, 
            description: dict = {}, api_key: str = "" ):
        
        """Generate the search phrases"""
        search_phrases = self.generate_search_phrases(goal=goal, answers=answers, prompts=prompts, textgen_config=textgen_config, text_gen=text_gen)

        """Take web search results for each search phrase"""
        search_results = []
        for search_phrase in search_phrases:
            curr_search_results = self.search(search_phrase=search_phrase, api_key=api_key)
            for result in curr_search_results:
                search_results.append(result)
        
        print(search_results)

        scraper = WebScraper(user_agent='windows')
        contents = []

        for search_result in search_results:
            content = scraper.scrape_url(search_result)
            contents.append(content)

        print(contents)

        """Retrieve the most relevant documents"""
        retriever = EmbeddingRetriever()
        references = retriever.retrieve_embeddings(contents, search_results, answers)
        print(references)

        """Building the insight given the references"""

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

        user_prompt += f"""
        \nTHESE ARE THE REFERENCES:
        \n{references}
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
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} insights are:\n"}
        ]

        result = text_gen.generate(messages=messages, config=textgen_config)

        try:            
            result = clean_code_snippet(result.text[0]['content'])
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

        return result
    
    def research(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, persona:Persona = None, n=5, 
            description: dict = {}, api_key: str = "" ):
        
        """Generate the search phrases"""
        search_phrases = self.generate_search_phrases(goal=goal, answers=answers, prompts=prompts, textgen_config=textgen_config, text_gen=text_gen)

        """Take web search results for each search phrase"""
        search_results = []
        for search_phrase in search_phrases:
            curr_search_results = self.search(search_phrase=search_phrase, api_key=api_key)
            for result in curr_search_results:
                search_results.append(result)
        
        print(search_results)

        scraper = WebScraper(user_agent='windows')
        contents = []

        for search_result in search_results:
            content = scraper.scrape_url(search_result)
            contents.append(content)

        print(contents)

        """Retrieve the most relevant documents"""
        retriever = EmbeddingRetriever()
        references = retriever.retrieve_embeddings(contents, search_results, answers)
        print(references)

        """Building the insight given the references"""

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
        \nCan you generate A TOTAL OF {n} QUESTIONS from the answers that draws connections between them?
        """

        user_prompt += f"""
        \nTHESE ARE THE REFERENCES:
        \n{references}
        """
        
        # Define persona
        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")
            
        # Prompt: Add persona
        user_prompt += f"\nThe generated questions SHOULD TRY TO BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona, who is interested in complex, insightful questions about the data.\n"

        # Prompt: Add description if applicable
        if description != {}:
            user_prompt += "These are the descriptions of the columns of the dataset. Try to make connections with the descriptions provided below with your hypothesis and the search phrase if it's applicable when generating the questions."
            user_prompt += str(description)
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RA},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS_RA}\n\nThe generated {n} questions are:\n"}
        ]

        result = text_gen.generate(messages=messages, config=textgen_config)

        try:            
            result = clean_code_snippet(result.text[0]['content'])
            result = json.loads(result)

            # cast each item in the list to an Insight object
            if isinstance(result, dict):
                result = [result]
            result = [Research(**x) for x in result]
        except Exception as e:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        return result