"""
This module is a Pipeline for ingesting raw `.txt` file into Weaviate
"""
import os
from typing import List, Dict
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_community.document_loaders import WebBaseLoader, UnstructuredHTMLLoader, UnstructuredPDFLoader, UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from weaviate import Client

VECTOR_SCHEMA = {
    "class": "Document",
    "description": "This is a class to store Document chunks",
    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
            "description": "Chunk of document content",
            "indexSearchable": False,
            "indexFilterable": False
        },
        {
            # Allow filter on this metadata with Roaring Bitmaps index
            # Disable search by keyword on this metadata 
            "name": "topic",
            "dataType": ["text"],
            "description": "topic of this document",
            "indexSearchable": False,
            "indexFilterable": True
        },
        {
            "name": "doc_name",
            "dataType": ["text"],
            "description": "title of this document",
            "indexSearchable": True,
            "indexFilterable": False
        },
        {
            "name": "source",
            "dataType": ["text"],
            "description": "location of this document",
            "indexSearchable": False,
            "indexFilterable": False
        }
    ],
    "vectorIndexConfig": {
        "distance": "cosine",
    },
}

class Ingestion:
    def __init__(self):
        self.client = None
        self.embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5") # which is also default
        self.parsed_docs: List[Document] = []

    def _connect_to_database(self):
        self.client = Client(url=os.environ["WEAVIATE_URL"])
        self.client.schema.delete_all()
        self.client.schema.create_class(VECTOR_SCHEMA)

    def _prepare_document_metadata(self, path) -> List[Dict]:
        """
        Try to extract metadata from `path` directory tree as documents
        are organized by sub_folder, and the sub_folder name is used as
        `topic` metadata for filtering purpose later
        Returns:
            - doc_metadata (List[Dict]): A list of documents' `path` and `properties`
            For example: 
            [{'path': '../data/region/Fontaine.txt', 'properties': {'topic': 'region'}},
            {'path': '../data/characters/Neuvillete.txt',]
        """
        doc_metadata = []
        for dir in os.listdir(path):
            for file_name in  os.listdir(os.path.join(path, dir)):
                properties = {
                    "topic": dir
                }
                doc_metadata.append({"path": os.path.join(path, dir, file_name),
                                    "properties": properties})
        return doc_metadata

    def _prepare_document_chunks(self, path, chunk_size: int=512) -> None:
        """
        This method reads a list of documents in a path
        (default="../data/"), parse the resulted text string using Unstructured
        module and recursively creating chunks that is within the
        max_token_size of embedding model limit (default=512)
        Args:
            - path (str): the paths to the directory where the files are saved
            This directory must follow the structure for this specific pipeline
            
            ├── <path>
            │   ├── <topic_1>
            │   │   ├── **/*.txt
            │   ├── <topic_2>
            │   │   ├── **/*.txt
        """
        doc_metadata = self._prepare_document_metadata(path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
        )
        
        for item in doc_metadata:
            filtered_doc = []
            loader_unstructured = UnstructuredFileLoader(item["path"], mode="elements")
            for doc in loader_unstructured.load_and_split(text_splitter):
                if doc.metadata["category"] in ["NarrativeText", "UncategorizedText"]:
                    new_metadata = {
                        "source": doc.metadata["source"],
                        "doc_name": doc.metadata["filename"].split(".")[0],
                        **item["properties"],
                    }
                    doc.metadata = new_metadata
                    filtered_doc.append(doc)
            # update metadata in doc
            self.parsed_docs.append(filtered_doc)
            print(f"TOTAL DOCS INGESTED {len(self.parsed_docs)}")

    def invoke(self, path="../data"):
        """
        Run the ingestion pipeline
        """
        self._connect_to_database()

        # parse and chunk text
        self._prepare_document_chunks(path)

        # create embeddings and batch insert using langchain
        for chunks in self.parsed_docs:
            Weaviate.from_documents(
                client=self.client, documents=chunks, embedding=self.embedding, index_name="Document", text_key="text", by_text=False
            )
        
        print("INGESTION COMPLETE")