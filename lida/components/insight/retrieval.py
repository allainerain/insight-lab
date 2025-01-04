import yaml
import os
import faiss
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

class EmbeddingRetriever:
    TOP_K = 3  # Number of top K documents to retrieve

    def __init__(self):
        # Load configuration from config.yaml file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )

    def retrieve_embeddings(self, contents_list: list, link_list: list, queries: list):
        # Retrieve embeddings for a given list of contents and a query
        metadatas = [{'url': link} for link in link_list]
        texts = self.text_splitter.create_documents(contents_list, metadatas=metadatas)

        # print(texts, metadatas)

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=self.config["openai_api_key"])
        document_embeddings = [embeddings.embed_query(doc.page_content) for doc in texts]
        embedding_dim = len(document_embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)

        index.add(np.array(document_embeddings).astype("float32"))

        db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(texts)}),
            index_to_docstore_id={i: str(i) for i in range(len(texts))}
        )
        
        # Create a retriever from the database to find relevant documents
        references = set()
        retriever = db.as_retriever(search_kwargs={"k": self.TOP_K})

        for query in queries:
            curr_references = retriever.get_relevant_documents(query)
            for reference in curr_references:
                references.add((reference.metadata['url'], reference.page_content))

        return references
