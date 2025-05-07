import nltk
from nltk.tokenize import sent_tokenize
import requests
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import concurrent.futures

nltk.download('punkt_tab')
class EmbeddingRetriever:
    def __init__(self, qdrant_host: str = "https://2e6f3a2c-72b1-4bef-b5e2-b04d1cebdd48.us-west-1-0.aws.cloud.qdrant.io:6333", qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ixq4UY34mswYWbVVnmu2ttJl5PjZAdIXuthvzeSDAHw"):

        # Initialize Qdrant client with API key for a remote instance
        self.client = qdrant_client.QdrantClient(
            url=qdrant_host,  # Qdrant cloud/self-hosted URL
            api_key=qdrant_api_key   # Qdrant API Key for authentication
        )

        self.collection_name = "Collection"

        print(qdrant_host, qdrant_api_key)
        
        # Check if the collection exists before creating it
        existing_collections = self.client.get_collections().collections 

        if not any(col.name == self.collection_name for col in existing_collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  
            )

    def split_by_sentences(self, contents_list):
        """Splits text into sentences dynamically."""
        processed_texts = []
        
        for content in contents_list:
            sentences = sent_tokenize(content)  # Tokenize into sentences
            processed_texts.extend(sentences)  # Store individual sentences
        
        return processed_texts

    def split_by_paragraphs(self, contents_list):
        """Splits text into paragraphs dynamically."""
        processed_texts = []
        
        for content in contents_list:
            # Split by double or single newlines, depending on the format
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            processed_texts.extend(paragraphs)
        
        return processed_texts

    def retrieve_embeddings(self, contents_list: list, link_list: list, queries: list):
        """Embeds and stores documents in Qdrant for retrieval."""
        
        # Split text into sentence-based chunks
        metadatas = [{'url': link} for link in link_list]
        processed_texts = self.split_by_paragraphs(contents_list)
        
        # Get Jina embeddings
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_699bb4adcc274e3482354dc34b53b632XMbQ4kVPcReyJB0G_9YNe6qfI0vK"
        }
        payload = {
            "input": processed_texts,
            "model": "jina-embeddings-v3",
            "late_chunking": True,
        }
        response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)

        if response.status_code == 200:
            embedding_data = response.json()
            embeddings = [item["embedding"] for item in embedding_data.get("data", [])]
        else:
            print("Error:", response.text)
            return

        # Store embeddings in Qdrant
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={"text": processed_texts[i], "url": metadatas[i % len(metadatas)]["url"]},
            )
            for i in range(len(embeddings))
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Stored {len(points)} embeddings in Qdrant.")

        results = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.search_relevant_references, queries))
        
        return results


    def search_relevant_references(self, query: str, top_k: int = 5):
        """Retrieves the most relevant references from Qdrant based on a query."""
        
        # Get Jina embedding for the query
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_9971746643d64fd591bf25981e4e4ab3LFNJs8tk6vjvm6MpEgdcjHhNNkaj"
        }
        payload = {
            "input": [query],
            "model": "jina-embeddings-v3"
        }
        response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)

        if response.status_code == 200:
            query_embedding = response.json()["data"][0]["embedding"]
        else:
            print("Error getting query embedding:", response.text)
            return []

        # Perform nearest neighbor search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        # Extract and return results
        return [
            {"text": hit.payload["text"], "url": hit.payload["url"], "score": hit.score}
            for hit in search_results
        ]