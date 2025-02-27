import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig
from lida.datamodel import Goal, Prompt, Insight, Persona, Research
from ..insight.webscraper import WebScraper
from ..insight.retrieval import EmbeddingRetriever
from ..insight.search import Searcher
import concurrent.futures

import json

SYSTEM_PROMPT = """
You are an experienced data analyst tasked with generating a specific number of meaningful and creative insights from a given data visualization. Your insights should go beyond surface-level observations and highlight patterns that might be overlooked.

For each insight, follow these guidelines:

1. Form a Hypothesis
- Develop a well-thought-out hypothesis based on the user’s provided goal, their answers to specific questions, and relevant evidence.
- Ensure that the hypothesis is multi-dimensional, creative, and unexpected.

2. Incorporate Supporting Evidence
- Use references from the provided web search results to strengthen your insight.
- Cite the correct reference(s) using [number] notation, ensuring that the citation aligns accurately with the evidence list.
- When citing references, reindex the references based only on those that are actually used. For each insight, restart the reference list count to 1.

3. Explain Your Thought Process
- Clearly state how you arrived at the hypothesis using both the web search results and your own analytical knowledge.
- If a user’s answer seems incorrect or misleading, point it out with credible sources from the web.

4. Ensure Logical Soundness
- The insight must be logical and factually correct.
- All claims should be substantiated with proper evidence.
- If one of the user's claims are wrong, you must point it out.

IF THERE ARE NO REFERNCES, DO NOT MAKE UP YOUR OWN AND INSTEAD STATE THAT YOU CANNOT FIND REFERENCES.
KEEP THE "evidence" DICT EMPTY, like
'evidence' : { }
AND DO NOT USE NUMBERS TO CITE IMAGINARY REFERENCES
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
        { 
            "index": 0,  
            "insight": "The (finding) could indicate (rest of insight) because of some reason [1] and some other reason [2]", 
            "evidence": {
                "1": ["URL", "Quoted Sentence"], 
                "2": ["URL", "Quoted Sentence"],
                ...
            }
        }
    ]f
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE. Make sure that the JSON format is free from any errors. Any quotes within strings need to be escaped with a backslash (\").
"""

SYSTEM_PROMPT_RA = """
You are a helpful and highly skilled data analyst who is trained to provide helpful, prompting questions to guide the user to gain insights from a data visualization given their goal, additional references, and GIVEN THEIR ANSWERS TO QUESTIONS. 

The questions you ask must be the following
- Incite insightful ideas and be meaningful.
- Be related to the goal, the visualization given, the provided references, AND THE USER ANSWERS.
- Clarify domain knowledge of the data ONLY if necessary. If you do this, add an example answer to guide the user.

IF THERE ARE NO REFERNCES, DO NOT MAKE UP YOUR OWN AND INSTEAD STATE THAT YOU CANNOT FIND REFERENCES.
KEEP THE "evidence" DICT EMPTY, like
'evidence' : { } 
AND DO NOT USE NUMBERS TO CITE IMAGINARY REFERENCES

"""

FORMAT_INSTRUCTIONS_RA = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
        { 
            "index": 0,  
            "question": "prompting question", 
            "evidence": {
                "1": ["URL", "Quoted Sentence"], 
                "2": ["URL", "Quoted Sentence"],
                ...
            }
        }
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE. Make sure that the JSON format is free from any errors. Any quotes within strings need to be escaped with a backslash (\").
"""

logger = logging.getLogger("lida")

class InsightExplorer(object):
    """Generate insights given some answers to questions"""
    def __init__(self, serper_api_key, qdrant_api_key, qdrant_url):
        self.searcher = Searcher(serper_api_key=serper_api_key)
        self.retriever = EmbeddingRetriever(qdrant_host=qdrant_url, qdrant_api_key=qdrant_api_key)

    def generate(
            self, goal: Goal, answers: list[str], prompts: Prompt, 
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, persona:Persona = None, n=5, 
            description: dict = {}):
        
        """Generate the search phrases"""
        search_phrases = self.searcher.generate_search_phrases(goal=goal, answers=answers, prompts=prompts, textgen_config=textgen_config, text_gen=text_gen)

        """Take web search results for each search phrase"""
        search_results = []
        for search_phrase in search_phrases:
            curr_search_results = self.searcher.search(search_phrase=search_phrase)
            for result in curr_search_results:
                search_results.append(result)
        
        # print(search_results)
        scraper = WebScraper(user_agent='windows')
        contents = []

        # Scrape in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(scraper.scrape_url, search_results))

        # Store the results in contents
        contents.extend(results)
        # print(contents)

        """Retrieve the most relevant documents"""
        references = self.retriever.retrieve_embeddings(contents, search_results, search_phrases)
        # print(references)

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
            description: dict = {}):
        
        """Generate the search phrases"""
        search_phrases = self.searcher.generate_search_phrases(goal=goal, answers=answers, prompts=prompts, textgen_config=textgen_config, text_gen=text_gen)

        """Take web search results for each search phrase"""
        search_results = []
        for search_phrase in search_phrases:
            curr_search_results = self.searcher.search(search_phrase=search_phrase)
            for result in curr_search_results:
                search_results.append(result)
        
        # print(search_results)
        scraper = WebScraper(user_agent='windows')
        contents = []

        # Scrape in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(scraper.scrape_url, search_results))

        # Store the results in contents
        contents.extend(results)
        # print(contents)

        """Retrieve the most relevant documents"""
        references = self.retriever.retrieve_embeddings(contents, search_results, search_phrases)
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