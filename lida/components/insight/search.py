import json
import http
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig
from lida.datamodel import Goal, Prompt, Insight

SYSTEM_PROMPT ="""
You are a highly skilled data analyst trained to generate the most relevant web search phrases to support observations about a data visualization. Your goal is to create search queries that will yield high-quality, diverse, and authoritative results.

For each search phrase, ensure the following:

1. Relevance - The search phrase must directly relate to an observation from the data visualization.
2. Specificity - It should be precise enough to confirm or challenge the user's answers to the given questions.
3. Diversity - Each search phrase should explore a different angle to minimize redundant search results.
4. Keyword Variation - Include synonyms and alternative terms to capture a wider range of relevant sources.
5. Context Awareness - Incorporate key details such as industry, region, timeframe, or demographics if applicable.
6. Comparison & Cause-Effect Focus - Use terms that explore relationships between variables (e.g., 'impact' 'effect', 'trend', 'correlation').
7. Exclusion of Irrelevant Results - Use techniques such as negative keywords (-keyword) to remove unrelated content.
8. Advanced Search Operators - When necessary, use "exact phrases", site:.gov, site:.edu, intitle:, or inurl: to improve search precision.
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF STRINGS (the search phrases). IT MUST USE THE FOLLOWING FORMAT:

```["What are the...", "Most popular..."]
```

THE OUTPUT SHOULD ONLY USE THE LIST FORMAT ABOVE.
"""

class Searcher:
    BLACKLISTED_DOMAINS = ["quora.com", "medium.com", "reddit.com"]

    def __init__(self, serper_api_key):
        self.api_key = serper_api_key
     
    def search(self, search_phrase: str):
        """Search the web given some search phrase"""

        conn = http.client.HTTPSConnection("google.serper.dev")

        payload = json.dumps({
            "q": search_phrase,
            "num": 2
        })

        headers = {
            'X-API-KEY': self.api_key,
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

        filtered_links = [
            link for link in links if not any(domain in link for domain in self.BLACKLISTED_DOMAINS)
        ]

        return filtered_links
    
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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} search phrases are:\n"}
        ]

        result: list[Insight] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            result = clean_code_snippet(result.text[0]["content"])
            result = json.loads(result)
        except Exception as e:
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid LIST object while attempting to generate goals. Consider using a larger model or a model with a higher max token length.")

        return result
    