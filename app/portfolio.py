import os
import uuid
import pandas as pd
import chromadb


class Portfolio:
    def __init__(self, file_path: str = None):
        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, "resource", "my_portfolio.csv")

        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
        # Initialize persistent ChromaDB client inside vectorstore/ folder
        vectorstore_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")
        self.chroma_client = chromadb.PersistentClient(path=vectorstore_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Loads data from CSV into ChromaDB collection if collection is empty."""
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills: list) -> list:
        """Queries ChromaDB for portfolio links matching extracted skills."""
        if isinstance(skills, list):
            query_str = ", ".join(skills)
        else:
            query_str = str(skills)

        results = self.collection.query(query_texts=[query_str], n_results=2)
        links = []
        if results and "metadatas" in results and results["metadatas"]:
            for metadata_list in results["metadatas"]:
                for meta in metadata_list:
                    if "links" in meta:
                        links.append(meta["links"])
        return links
