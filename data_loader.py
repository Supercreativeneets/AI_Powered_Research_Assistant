import requests
import xml.etree.ElementTree as ET
from arxiv2text import arxiv_to_text

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class DataLoader:
    def __init__(self, search_agent=None):
        print("DataLoader Init")
        self.search_agent = search_agent
    
    def search_arxiv(self, query):
        """Helper function to query ArXiv API."""
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            return [
                {
                    "title": entry.find("{http://www.w3.org/2005/Atom}title").text,
                    "date": entry.find("{http://www.w3.org/2005/Atom}updated"),
                    "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text,
                    "link": entry.find("{http://www.w3.org/2005/Atom}id").text,
                    "pdf_url": entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf") + ".pdf"
                }
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry")
            ]
        return []
    
    def extract_content(self, url: str) -> str:
        """Extracts text content from an arXiv PDF URL."""
        return arxiv_to_text(url)
    
    def fetch_arxiv_papers(self, query):
        """
        Fetches top 3 research papers from ArXiv based on the user query.
        Extracts content from the PDFs and returns a list of dictionaries.
        If fewer than 3 papers are found, expands the search using related topics.
        
        Returns:
            list: A list of dictionaries containing paper details and extracted content.
        """
        papers = self.search_arxiv(query)
        
        if len(papers) < 5 and self.search_agent:  # If fewer than 5 papers, expand search
            related_topics_response = self.search_agent.generate_reply(
                messages=[{"role": "user", "content": f"Suggest 3 related research topics for '{query}'"}]
            )
            related_topics = related_topics_response.get("content", "").split("\n")

            for topic in related_topics:
                topic = topic.strip()
                if topic and len(papers) < 5:
                    new_papers = self.search_arxiv(topic)
                    papers.extend(new_papers)
                    papers = papers[:5]  # Ensure max 5 papers
        
        for paper in papers:
            paper["extracted_content"] = self.extract_content(paper["pdf_url"])
        
        return papers

        
class Embed_content:
    def __init__(self):
        print("Embed content init")
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs = {'device': 'cpu'},encode_kwargs = {'normalize_embeddings': False})

    def create_vector(self,paper_content):
        split_doc = self.text_splitter.create_documents(paper_content)
        return FAISS.from_documents(split_doc,self.embedding)
