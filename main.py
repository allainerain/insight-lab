import streamlit as st
import os
import pandas as pd
import copy
import sys

sys.path.append(os.path.abspath('..'))

from llmx import llm, TextGenerationConfig, TextGenerator
from lida.components import Manager
from lida.datamodel import Goal, Persona, Insight, Prompt
from lida.utils import read_dataframe
from code_editor import code_editor

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="LIDA+: Exploratory Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
)

# States
if "saved_visualizations" not in st.session_state:
    st.session_state.saved_visualizations = []

if "saved_goals" not in st.session_state:
    st.session_state.saved_goals = []

if "saved_insights" not in st.session_state:
    st.session_state.saved_insights = []


if "visualization" not in st.session_state:
    st.session_state.visualization = None

if "prompts" not in st.session_state:
    st.session_state.prompts = None

selected_dataset = None

st.write("# LIDA+: Exploratory Data Analysis Assistant 📊")
st.markdown(
    """
    LIDA is a library for generating data visualizations and data-faithful infographics.
    LIDA is grammar agnostic (will work with any programming language and visualization
    libraries e.g. matplotlib, seaborn, altair, d3 etc) and works with multiple large language
    model providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface). Details on the components
    of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927) and in this
    tutorial [notebook](notebooks/tutorial.ipynb). See the project page [here](https://microsoft.github.io/lida/) for updates!

   ----
""")

openai_key = os.getenv("OPENAI_API_KEY")

#################
# SET UP CONFIG
#################

st.sidebar.write("# Setup")

with st.sidebar.expander("LIDA Version"):
    version = st.selectbox(
        "What version of LIDA do you want to use?", 
        ("LIDA", "LIDA+", "LIDA++")
    )

with st.sidebar.expander("Generation Settings"):
    openai_tab, serper_tab, qdrant_tab = st.tabs(["OpenAI Config", "Serper Config", "Qdrant Config"])

    # Openai settings
    with openai_tab:

        # Set openai key
        st.write("#### OpenAI Key")
        openai_key = st.text_input("## Enter OpenAI API key")

        if openai_key:
            display_openai_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
            st.write(f"Current key: {display_openai_key}")

        # Set model settings
        # Set model, temperature and cache settings
        st.write("#### Text Generation Model")
        models = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        selected_model = st.selectbox(
            'Choose a model',
            options=models,
            index=0
        )

        # select temperature on a scale of 0.0 to 1.0
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0)

        # set use_cache in sidebar
        use_cache = st.checkbox("Use cache", value=True)

    # Serper settings
    with serper_tab:

        # Set serper key
        st.write("#### Serper Key")
        serper_api_key = st.text_input("Enter Serper key")
        
        if serper_api_key:
            display_serper_key = serper_api_key[:2] + "*" * (len(serper_api_key) - 5) + serper_api_key[-3:]
            st.write(f"Current key: {display_serper_key}")

    # Serper settings
    with qdrant_tab:

        # Set serper key
        st.write("#### Qdrant Keys")
        qdrant_api_key = st.text_input("Enter Qdrant key")
        qdrant_url= st.text_input("Enter Qdrant URL")
        
        if qdrant_api_key:
            display_qdrant_key = qdrant_api_key[:2] + "*" * (len(qdrant_api_key) - 5) + qdrant_api_key[-3:]
            st.write(f"Current key: {display_qdrant_key}")


if openai_key:

    #################
    # DATASET
    #################
    selected_dataset = None

    # Upload dataset
    uploaded_file = st.file_uploader("Ready? Upload a CSV or JSON file to begin", type=["csv", "json"])

    # Select dataset
    datasets = [
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
        {"label": "Cyclistic", "path": "notebooks/cyclistic_data.csv"},  # Changed to local path
    ]

    selected_dataset_label = st.pills(
        'Dont have a dataset? Try any of the datasets below:',
        options=[dataset["label"] for dataset in datasets],
    )

    # Process uploaded or selected dataset
    if uploaded_file is not None:
        file_name, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension.lower() == ".csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension.lower() == ".json":
            data = pd.read_json(uploaded_file)

        uploaded_file_path = os.path.join("data", uploaded_file.name)
        data.to_csv(uploaded_file_path, index=False)

        selected_dataset = uploaded_file_path

        datasets.append({"label": file_name, "url": uploaded_file_path})

        # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    elif selected_dataset_label:
        selected_dataset_info = datasets[[dataset["label"] 
                                    for dataset in datasets].index(selected_dataset_label)]
        
        if "url" in selected_dataset_info:
            selected_dataset = selected_dataset_info["url"]
        elif "path" in selected_dataset_info:
            selected_dataset = selected_dataset_info["path"]
            
            # Handle local file loading
            if selected_dataset.endswith('.csv'):
                data = pd.read_csv(selected_dataset)
            elif selected_dataset.endswith('.json'):
                data = pd.read_json(selected_dataset)
    else: 
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")


if openai_key and selected_dataset:

    # PERSONA
    st.write("## Motivation")
    persona = st.text_area("Describe who you are and your goal in exploring the dataset. This will be helpful in tailoring recommendations based on your goal.")
    persona = Persona(persona=persona, rationale="")

    #################
    # SUMMARIZER
    #################
    st.write("## Dataset")
    st.write(read_dataframe(selected_dataset))
    
    st.write("## Dataset Summary")
    # Initlaize LIDA
    lida = Manager(text_gen=llm("openai", api_key=openai_key), serper_api_key=serper_api_key, qdrant_api_key=qdrant_api_key, qdrant_url=qdrant_url)
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)
    
    # Select summarization method
    summarization_methods = [
        {"label": "default",
         "description": "Annotate the data summary manually by adding descriptions and semantic types to each column."},
        {"label": "enrich",
         "description":"Use the LLM to annotate the data summary by adding descriptions and semantic types to each column. Feel free to edit the annotation of the LLM!"}
    ]

    with st.container(border=True):
        selected_method_label = st.selectbox(
            'Choose a summarization method',
            options=[method["label"] for method in summarization_methods],
            index=0
        )

        # add description of selected method in very small font to sidebar
        selected_summary_method_description = summarization_methods[[
            method["label"] for method in summarization_methods].index(selected_method_label)]["description"]
        st.write(selected_summary_method_description)

    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method_label,
        textgen_config=textgen_config)
    
    # Construct the table
    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["variables"] = field["column"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)

        # Move description to the front
        cols = list(nfields_df.columns)
        cols.remove("description")
        cols.insert(1, "description")
        cols.remove("semantic_type")
        cols.insert(2, "semantic_type")
        nfields_df = nfields_df[cols]

        # Create the editable table
        disabled_columns = [col for col in nfields_df.columns if col not in ["description", "semantic_type"]]
        nfields_df = st.data_editor(nfields_df, disabled=disabled_columns) 

    else:
        st.write(str(summary))

    #################
    # GOAL EXPLORER
    #################
    st.write("## Goal Explorer")

    if "goals" not in st.session_state:
        st.session_state.goals = []

    if "goal_questions" not in st.session_state:
        st.session_state.goal_questions = []

    if summary:
        # GOAL
        with st.container(border=True):
            
            custom_goal_tab, generate_goals_tab = st.tabs(["Custom Goal", "Generate Goal"])
            
            # Generate goals settings
            with generate_goals_tab:
                st.write("Don't know where to begin? Generate suggested things to explore with the data.")
                custom_goal = False

                # Select number 
                num_goals = st.number_input("Number of goals to generate", max_value=10, min_value=1)

                # Select variables to explore
                options = list(read_dataframe(selected_dataset))
                explore = st.pills("Variables to explore", options, selection_mode="multi")

                # Add insight to explore
                insight = []
                if version != "LIDA":
                    insight_text = st.text_input("Do you have an insight you want LIDA to explore?")

                    # Set insight to None if there's no input
                    if insight_text.strip(): 
                        insight = [Insight(insight=insight_text, evidence={}, index=0)]

                # **** lida.goals *****
                if st.button("Generate"):
                    st.session_state.goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config, explore=explore, insights=insight)
                    st.session_state.goal_questions = [goal.question for goal in st.session_state.goals]

            # Custom goal settings
            with custom_goal_tab:
                user_goal = st.text_input("Describe your goal. What do you want to explore in your dataset?")
                if user_goal:
                    if user_goal not in st.session_state.goal_questions:
                        new_goal = Goal(question=user_goal, visualization=user_goal, rationale="", persona=persona, index=len(st.session_state.goal_questions))
                        st.session_state.goals.append(new_goal)
                        st.session_state.goal_questions.append(new_goal.question)
                    # st.session_state.visualization = None
        # print(st.session_state)
        if "goals" not in st.session_state or not st.session_state.goals:
            st.info("To continue, generate goals or add your own.")

        else:
            # Display Goals
            # if not user_goal:
            goal_questions = st.session_state.goal_questions
            goals = st.session_state.goals
            selected_goal_index = None
            selected_goal_object = None
            
            # Generated columns group
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            for i, goal in enumerate(goals):
                with columns[i % 3]:  

                    # Format each goal
                    with st.container(border=True):
                        st.write(goal.question)
                        st.markdown(
                            f"""
                            <code>{goal.visualization}</code>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"""
                            <p style="font-size:12px; color:gray;">
                                {goal.rationale}
                            </p>
                            """, 
                            unsafe_allow_html=True
                        )                      

                        goal_col1, goal_col2 = st.columns(2, gap="small")

                        with goal_col1:
                            if st.button("Visualize", key=f"visualize_{goal.index}", use_container_width=True):
                                selected_goal_index = goal.index
                                st.session_state.visualization = None
                                st.session_state.prompts = None
                        with goal_col2:
                            # Button for Save
                            if st.button("Save", key=f"save_{goal.index}", use_container_width=True):
                                if goals[goal.index] not in st.session_state.saved_goals:
                                    st.session_state.saved_goals.append(copy.deepcopy(goals[goal.index]))

            if selected_goal_index != None:
                st.session_state.selected_goal_object = st.session_state.goals[selected_goal_index]

            elif "visualization" not in st.session_state or not st.session_state.visualization:
                st.info("Select a goal to visualize, or input your own goal")

            # else:
            #     st.session_state.selected_goal_object = new_goal

            # GOAL SIDEBAR
            st.sidebar.write("## Notebook")

            with st.sidebar.container():
                saved_goals_tab, saved_viz_tab, saved_insights_tab= st.tabs(["Saved Goals", "Saved Viz", "Saved Insights"])

                with saved_goals_tab:
                    for saved_goal_index, saved_goal in enumerate(st.session_state.saved_goals):

                        with st.expander(saved_goal.question):
                            st.markdown(
                                f"""
                                <code>{saved_goal.visualization}</code>
                                """,
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"""
                                <p style="font-size:12px; color:gray;">
                                    {saved_goal.rationale}
                                </p>
                                """, 
                                unsafe_allow_html=True
                            )               

                            saved_goal_col1, saved_goal_col2 = st.columns(2, gap="small")

                            with saved_goal_col1:
                                if st.button("Load", use_container_width=True, key=f"load_{saved_goal_index}"):
                                    selected_goal_index = saved_goal_index
                                    st.session_state.selected_goal_object = st.session_state.saved_goals[saved_goal_index]
                                    st.session_state.visualization = None
                                    st.session_state.prompts = None

                            with saved_goal_col2:
                                if st.button("Delete", use_container_width=True, key=f"delete_{saved_goal_index}"):
                                    st.session_state.saved_goals.pop(saved_goal_index)
                                    st.rerun() 

            #################
            # VIZ GENERATOR
            #################
            if "selected_goal_object" in st.session_state and st.session_state.selected_goal_object:
                selected_goal_object = st.session_state.selected_goal_object
                st.write("## Visualization")

                # with st.container(border=True):
                #     visualization_libraries = ["seaborn", "matplotlib", "plotly"]

                #     selected_library = st.selectbox(
                #         'Choose a visualization library',
                #         options=visualization_libraries,
                #         index=0
                #     )

                # **** lida.visualize *****
                textgen_config = TextGenerationConfig(
                    n=1, temperature=temperature,
                    model=selected_model,
                    use_cache=use_cache)

                # print(st.session_state.visualization)

                if st.session_state.visualization is None:
                    # print("NEW VIZ")
                    st.session_state.visualization = lida.visualize(
                        summary=summary,
                        goal=selected_goal_object,
                        textgen_config=textgen_config,
                        library="seaborn")
                selected_vis = st.session_state.visualization

                # print(st.session_state)
                
                if "visualization" in st.session_state and st.session_state.visualization:

                    viz_col1, viz_col2 = st.columns([3,2], gap="medium")

                    # Viz ops or prompter
                    if version != "LIDA":
                        with viz_col2:
                            viz_ops, prompter = st.tabs(["Viz Ops", "Prompter"])
                    else:
                        viz_ops = viz_col2

                #################
                # VIZ OPS
                #################
                with viz_ops:
                    st.write("### Visualization Operations")

                    # EDIT VISUALIZATION
                    with st.container(border=True):
                        instruction = st.text_input("Input your edit instructions")

                        if st.button("Edit visualization"):
                            st.session_state.visualization = lida.edit(code=selected_vis[0].code,  summary=summary, instructions=[instruction], library="seaborn", textgen_config=textgen_config)
                            selected_vis = st.session_state.visualization
                            # this state doesn't carry over?

                    #FIX VISUALIZATION
                    with st.container(border=True):
                        st.write("Encountered a bug? Visualization won't render? Repair the visualization automatically.")
                        if st.button("Repair visualization"):
                            feedback = lida.evaluate(code=selected_vis[0].code,  goal=selected_goal_object, textgen_config=textgen_config, library="seaborn")[0] 
                            st.session_state.visualization = lida.repair(code=selected_vis[0].code, goal=selected_goal_object, summary=summary, feedback=feedback, textgen_config=textgen_config, library="seaborn")
                            selected_vis = st.session_state.visualization

                    # # VISUALIZATION CODEs
                    # with st.expander("Visualization Code"):
                    #     # print(st.session_state.visualization)
                    #     # selected_vis = st.session_state.visualization
                    #     # st.code(selected_vis[0].code)
                    #     code_display, code_editor = st.tabs(["Code", "Code Editor"])
                    #     with code_display:
                    #         st.code(selected_vis[0].code)
                    #     with code_editor:
                    #         code_edit = st.text_area('', selected_vis[0].code, height=500)
                    #         if st.button("Edit code"):
                    #             # selected_vis[0].code = code_edit
                    #             st.session_state.visualization = lida.execute(code_specs=[code_edit], data=read_dataframe(selected_dataset), summary=summary, library="seaborn")
                    #             selected_vis = st.session_state.visualization

                              
                # VISUALIZATION RENDER
                with viz_col1:
                    viz_title = selected_goal_object.question
                    # st.write(st.session_state.visualization[0])

                    # Rendering the visualization
                    if selected_vis[0].raster:
                        # print(selected_vis[0].code)
                        from PIL import Image
                        import io
                        import base64

                        imgdata = base64.b64decode(selected_vis[0].raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, caption=viz_title, use_container_width=True)

                    # if there is no raster, then repair automatically
                    else:
                        # print("repairing")
                        feedback = lida.evaluate(code=selected_vis[0].code,  goal=selected_goal_object, textgen_config=textgen_config, library="seaborn")[0] 
                        st.session_state.visualization = lida.repair(code=selected_vis[0].code, goal=selected_goal_object, summary=summary, feedback=feedback, textgen_config=textgen_config, library="seaborn")
                        selected_vis = st.session_state.visualization

                    # if st.button("Reload"):
                    #     print("reloading")
                    if st.button("Save Visualization to Notebook"):
                        selected_vis = st.session_state.visualization
                        # print("hi")
                        # print(len(st.session_state.saved_visualizations))
                        if selected_vis not in st.session_state.saved_visualizations:
                            # print("hello")
                            st.session_state.saved_visualizations.append(copy.deepcopy(selected_vis))
                            # print(len(st.session_state.saved_visualizations))
                    
                    with st.expander("Visualization Code"):
                        # print(st.session_state.visualization)
                        # selected_vis = st.session_state.visualization
                        # st.code(selected_vis[0].code)
                        code_display, code_edit = st.tabs(["Code", "Code Editor"])
                        with code_display:
                            st.code(selected_vis[0].code)
                        with code_edit:
                            # custom_btns = [{"name": "Edit Code","feather": "Play","primary": True,"hasText": True,"showWithIcon": True,"commands": ["submit"],"style": {"bottom": "0.44rem","right": "0.4rem"}}]
                            # code_response_dict = code_editor(selected_vis[0].code, height="500px", buttons=custom_btns)
                            code_edited = st.text_area('', selected_vis[0].code, height=500)
                            if st.button("Edit code"):
                                # selected_vis[0].code = code_edit
                                st.session_state.visualization = lida.execute(code_specs=[code_edited], data=read_dataframe(selected_dataset), summary=summary, library="seaborn")
                                selected_vis = st.session_state.visualization
                                st.rerun(scope="app")
                            # print("A", st.session_state.visualization[0].code)
                            # print("B",code_response_dict["text"], "C")
                            # print()
                            # if st.session_state.visualization[0].code != code_response_dict["text"] and code_response_dict["text"] != "  ":
                            #     selected_vis[0].code = code_response_dict["text"]
                            #     st.session_state.visualization = lida.execute(code_specs=[code_response_dict["text"]], data=read_dataframe(selected_dataset), summary=summary, library="seaborn")
                            #     selected_vis = st.session_state.visualization                    
                
                # VISUALIZATION TABs
                with saved_viz_tab:
                    st.warning("The visualizations saved are only the raster. The goal that generated it is not saved.")
                    for saved_viz_index, saved_visualization in enumerate(st.session_state.saved_visualizations):
                        with st.container(border=True):

                            if saved_visualization[0].raster:
                                from PIL import Image
                                import io
                                import base64

                                imgdata = base64.b64decode(saved_visualization[0].raster)
                                img = Image.open(io.BytesIO(imgdata))
                                st.image(img, use_container_width=True)
                            
                            saved_viz_col1, saved_viz_col2 = st.columns(2)
                            with saved_viz_col1:
                                if st.button("Load", key=f"load_saved_viz_{saved_viz_index}", use_container_width=True):
                                    st.session_state.visualization = st.session_state.saved_visualizations[saved_viz_index]
                                    selected_vis = st.session_state.visualization
                                    st.session_state.prompts = None

                            with saved_viz_col2:
                                if st.button("Delete", key=f"delete_saved_viz_{saved_viz_index}", use_container_width=True):
                                    st.session_state.saved_visualizations.pop(saved_viz_index)
                                    st.rerun() 

                with saved_insights_tab:
                    new_custom_insight = st.text_area("Add a custom insight here")
                    new_custom_insight_object = Insight(insight=new_custom_insight, evidence={}, index=0)
                    if new_custom_insight_object not in st.session_state.saved_insights and new_custom_insight != "":
                        st.session_state.saved_insights.append(new_custom_insight_object)

                    for saved_insight_index, saved_insight in enumerate(st.session_state.saved_insights):
                        with st.container(border=True):
                            st.write(saved_insight.insight)

                            for saved_evidence_index, saved_evidence in enumerate(saved_insight.evidence):
                                st.markdown(
                                    f"""
                                    <a href="{saved_insight.evidence[saved_evidence][0]}">
                                    <p style="font-size:12px; color:gray;">
                                        [{saved_evidence}]{saved_insight.evidence[saved_evidence][1]}
                                    </p>
                                    </a>
                                    """, 
                                    unsafe_allow_html=True
                                )       
                            
                                
                            saved_insights_col1, saved_insights_col2 = st.columns(2)
                            with saved_insights_col1:
                                if st.button("Load", key=f"load_saved_insights_{saved_insight_index}", use_container_width=True, disabled=True):
                                    ...
                            with saved_insights_col2:
                                if st.button("Delete", key=f"delete_saved_insights_{saved_insight_index}", use_container_width=True):
                                    st.session_state.saved_insights.pop(saved_insight_index)
                                    st.rerun() 

                #################
                # PROMPTER
                #################
                if version != "LIDA":
                    with prompter:
                        st.write("### Prompter")

                        if version == "LIDA+":
                            prompter_tab_label = "Prompter and Insight Explorer Settings"
                            insights_label = "Number of insights to generate"

                        if version == "LIDA++":
                            prompter_tab_label = "Prompter and Research Assistant Settings"
                            insights_label = "Number of research to generate"

                        with st.expander(prompter_tab_label):
                            num_questions = st.number_input("Number of questions to generate", max_value=10, min_value=1, value=5)
                            num_insights = st.number_input(insights_label, max_value=10, min_value=1, value=5)
                        
                        if st.button("Generate Questions"):
                            st.session_state.prompts = lida.prompt(goal=selected_goal_object, textgen_config=textgen_config, n=num_questions) 

                        
                        if "prompts" in st.session_state and st.session_state.prompts and st.session_state.prompts is not None:
                            st.session_state.answers = ["" for _ in st.session_state.prompts]
                            with st.container(height=300):
                                for i in range(len(st.session_state.prompts)):
                                    st.session_state.answers[i] = st.text_area(st.session_state.prompts[i].question)
                            
                            if version == "LIDA+":
                                if st.button("Generate Insights"):
                                    st.session_state.insights = lida.insights(goal=selected_goal_object, answers=st.session_state.answers, prompts=st.session_state.prompts, n=num_insights)

                            if version == "LIDA++":
                                if st.button("Generate Research"):
                                    st.session_state.researches = lida.research(goal=selected_goal_object, answers=st.session_state.answers, prompts=st.session_state.prompts, n=num_insights)
                                    # print(st.session_state.researches)

                if version == "LIDA+":
                    if "insights" in st.session_state and st.session_state.insights:
                        st.write("## Insights")
                        insights = st.session_state.insights

                        # Generated columns group
                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                        insight_columns = [insight_col1, insight_col2, insight_col3]
                        for i, insight in enumerate(insights):
                            with insight_columns[i % 3]:  

                                # Format each insight
                                with st.container(border=True):
                                    st.write(insight.insight)

                                    for evidence_index, evidence in enumerate(insight.evidence):
                                        st.markdown(
                                            f"""
                                            <a href="{insight.evidence[evidence][0]}">
                                            <p style="font-size:12px; color:gray;">
                                                [{evidence}]{insight.evidence[evidence][1]}
                                            </p>
                                            </a>
                                            """, 
                                            unsafe_allow_html=True
                                        )                      
                                        
                                    if st.button("Save", key=f"insight_{i}"):
                                        if insight not in st.session_state.saved_insights:
                                            st.session_state.saved_insights.append(insight)
                
# st.session_state.research_answers = []
# st.session_state.research_prompts = []
# st.session_state.researches = []
# st.session_state.saved_insights = []

                if version == "LIDA++":
                    if "researches" in st.session_state and st.session_state.researches:
                        st.write("## Research")
                        researches = st.session_state.researches

                        # Initialize answers and prompts as lists if not present
                        if "research_answers" not in st.session_state:
                            st.session_state.research_answers = ["" for _ in researches]
                        if "research_prompts" not in st.session_state:
                            st.session_state.research_prompts = [None for _ in researches]

                        # Adjust length of answers and prompts if researches list changes
                        while len(st.session_state.research_answers) < len(researches):
                            st.session_state.research_answers.append("")
                        while len(st.session_state.research_prompts) < len(researches):
                            st.session_state.research_prompts.append(None)

                        # Generate columns for the researches
                        research_col1, research_col2, research_col3 = st.columns(3)
                        research_columns = [research_col1, research_col2, research_col3]

                        for i, research in enumerate(researches):
                            with research_columns[i % 3]:
                                # Format each insight
                                with st.container(border=True):

                                    st.write(research.question)

                                    # Text area for the answer
                                    research_answer = st.text_area(
                                        "Put your answer here",
                                        key=f"research_answer_{i}",
                                        value=st.session_state.research_answers[i],
                                        on_change=lambda idx=i: st.session_state.research_answers.__setitem__(idx, st.session_state[f"research_answer_{idx}"]),
                                    )
                                    
                                    # Checkbox for "Research further"
                                    is_research_further = st.checkbox(
                                        "Research further", key=f"research_further_{i}", value=st.session_state.research_prompts[i] is not None
                                    )
                                    
                                    # Handle "Research further" checkbox logic
                                    if is_research_further:
                                        # Store prompt and answer
                                        if st.session_state.research_prompts[i] is None:
                                            st.session_state.research_prompts[i] = Prompt(
                                                question=research.question, rationale="", index=i
                                            )
                                        st.session_state.research_answers[i] = research_answer
                                    else:
                                        # Remove prompt and clear the answer if unchecked
                                        st.session_state.research_prompts[i] = None
                                        st.session_state.research_answers[i] = ""

                                    # Possible references
                                    with st.expander("Suggested references"):
                                        for evidence_index, evidence in enumerate(research.evidence):
                                            st.markdown(
                                                f"""
                                                <a href="{research.evidence[evidence][0]}">
                                                <p style="font-size:12px; color:gray;">
                                                    [{evidence}]{research.evidence[evidence][1]}
                                                </p>
                                                </a>
                                                """,
                                                unsafe_allow_html=True,
                                            )

                                    # Delete button
                                    if st.button("Delete", key=f"delete_research_{i}"):
                                        # Remove from researches, answers, and prompts
                                        st.session_state.researches.pop(i)
                                        st.session_state.research_answers.pop(i)
                                        st.session_state.research_prompts.pop(i)
                                        st.rerun()

                        # Button to generate more research
                        if st.button("Generate More Research"):
                            new_research = lida.research(
                                goal=selected_goal_object,
                                answers=st.session_state.research_answers,
                                prompts=[p for p in st.session_state.research_prompts if p is not None],
                                n=num_insights,
                                api_key=serper_key,
                            )
                            st.session_state.researches.extend(new_research)
                            # Extend answers and prompts lists for new research
                            st.session_state.research_answers.extend([""] * len(new_research))
                            st.session_state.research_prompts.extend([None] * len(new_research))
                            st.rerun()

                    st.write("## Insight")

                    new_insight_text = st.text_area("Add insight here")
                    new_insight = Insight(insight=new_insight_text, evidence={}, index=0)

                    if new_insight not in st.session_state.saved_insights and new_insight_text != "":
                        st.session_state.saved_insights.append(new_insight)
                        st.rerun()