# 📊 LIDA++: Facilitating Insight Discovery using Large Language Models

This project is build on LIDA. LIDA is a library for generating data visualizations and data-faithful infographics. LIDA is grammar agnostic (will work with any programming language and visualization libraries e.g. matplotlib, seaborn, altair, d3 etc) and works with multiple large language model providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface).

LIDA++ aims to improve on the capabilities of LIDA by introducing new modules for insight discovery.

> **Original research on LIDA**
> Details on the original components of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927) and in this tutorial [notebook](notebooks/tutorial.ipynb). See the project page [here](https://microsoft.github.io/lida/) for updates!.

> **Note on Code Execution:**
> To create visualizations, LIDA _generates_ and _executes_ code.
> Ensure that you run LIDA in a secure environment.

## Table of Contents

1. [Features](#features)
   1.1. [Updated LIDA library features](#updated-lida-library-features)
   1.2. [WebApp features](#webapp-features)
   1.2.1. [LIDA+: Partially automated insight generation for insight discovery](#lida-partially-automated-insight-generation-for-insight-discovery)
   1.2.2. [LIDA+: Assistant for insight discovery without directly generating insights](#lida-assistant-for-insight-discovery-without-directly-generating-insights)
2. [Getting Started with the WebApp](#getting-started-with-the-web-app)
3. [Getting Started with the Library](#getting-started-with-the-library)
4. [Documentation and Citation](#documentation-and-citation)

## Features

This work has two parts: a library and a web app. The library can be accessed in the `lida` folder while the web app can be accessed in the `lida-streamlit` folder.

### Updated LIDA library features

LIDA treats _**visualizations as code**_ and provides a clean api for generating, executing, editing, explaining, evaluating and repairing visualization code.

- [x] Data Summarization
- [x] Goal Generation
- [x] Visualization Generation
- [x] Visualization Editing
- [x] Visualization Explanation
- [x] Visualization Evaluation and Repair
- [x] Chart Question and Answering
- [x] Insight Generation
- [x] Insight Discovery Research

### WebApp features

For the webapp, we present two workflows built from the modified LIDA library: LIDA+ and LIDA++.

#### LIDA+: Partially automated insight generation for insight discovery

<image src="./docs/images/LIDA+.png">

#### LIDA+: Assistant for insight discovery without directly generating insights

<image src="./docs/images/LIDA++.png">

## Getting Started with the Web App

Setup and verify that your python environment is **`python 3.10`** or higher (preferably, use [Conda](https://docs.conda.io/en/main/miniconda.html#installing)).

### Clone the repository

```bash
git clone https://github.com/allainerain/lida.git
```

### Install the requirements

```bash
pip install -U requirements.txt
```

LIDA depends on `llmx` and `openai`. If you had these libraries installed previously, consider updating them.

```bash
pip install -U llmx openai
```

### Set environment variables

1. Create a .env file with the following

```python
OPENAI_APIKEY = "sk-xxxxxxx"
SERPER_APIKEY = "xxxx"
```

2. Add a `config` folder containing a `config.yaml` file in `lida/components/insight` containing the following information:

```python
openai_api_key: "sk-xxxxxxx"
serper_api_key: "xxxx"
```

### Run the web app

```bash
cd lida-streamlit
streamlit run main.py
```

## Getting Started with the Library

The fastest and recommended way to learn about LIDA++'s capabilities is through the [LIDA+ and LIDA++ handbook notebook](notebooks/tutorial.ipynb).

## Library Methods

### Data Summarization

Given a dataset, generate a compact summary of the data.

```python
from lida import Manager

lida = Manager()
summary = lida.summarize("data/cars.json") # generate data summary
```

### Goal Generation

Generate a set of visualization goals given a data summary.

```python
goals = lida.goals(summary, n=5, persona="ceo with aerodynamics background") # generate goals
```

Add a `persona` parameter to generate goals based on that persona.

### Visualization Generation

Generate, refine, execute and filter visualization code given a data summary and visualization goal. Note that LIDA represents **visualizations as code**.

```python
# generate charts (generate and execute visualization code)
charts = lida.visualize(summary=summary, goal=goals[0], library="matplotlib") # seaborn, ggplot ..
```

### Visualization Editing

Given a visualization, edit the visualization using natural language.

```python
# modify chart using natural language
instructions = ["convert this to a bar chart", "change the color to red", "change y axes label to Fuel Efficiency", "translate the title to french"]
edited_charts = lida.edit(code=code,  summary=summary, instructions=instructions, library=library, textgen_config=textgen_config)

```

### Visualization Explanation

Given a visualization, generate a natural language explanation of the visualization code (accessibility, data transformations applied, visualization code)

```python
# generate explanation for chart
explanation = lida.explain(code=charts[0].code, summary=summary)
```

### Visualization Evaluation and Repair

Given a visualization, evaluate to find repair instructions (which may be human authored, or generated), repair the visualization.

```python
evaluations = lida.evaluate(code=code,  goal=goals[i], library=library)
```

### Prompting

Given a goal, generate prompting-probing questions to allow the user to critically analyze the visualization.

```python
prompts = lida.prompt(goal=goal, textgen_config=textgen_config)

```

### Insight Generation

Given answers to prompts, search the web for relevant references and generate suggested insights.

```python
insights = lida.insights(goal=goal, answers=answers, prompts=promts,  textgen_config=textgen_config, api_key="SERPER_APIKEY")
```

### Research

Given answers to prompts, search the web for relevant references and suggest more probing questions.

```python
research = lida.research(goal=goal, answers=answers, prompts=promts,textgen_config=textgen_config, api_key="SERPER_APIKEY")
```

## Documentation and Citation

This work is build on the work on LIDA. A short paper describing LIDA (Accepted at ACL 2023 Conference) is available [here](https://arxiv.org/abs/2303.02927).

```bibtex
@inproceedings{dibia2023lida,
    title = "{LIDA}: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models",
    author = "Dibia, Victor",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.11",
    doi = "10.18653/v1/2023.acl-demo.11",
    pages = "113--126",
}
```

LIDA builds on insights in automatic generation of visualization from an earlier paper - [Data2Vis: Automatic Generation of Data Visualizations Using Sequence to Sequence Recurrent Neural Networks](https://arxiv.org/abs/1804.03126).
