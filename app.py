import streamlit as st
import os
from dotenv import load_dotenv
from agents import ResearchAgents
from data_loader import DataLoader, Embed_content
from rag import Rag

load_dotenv()

print("ok")

# Streamlit UI Title
st.title("📚 Virtual Research Assistant")

# Retrieve the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API key is set, else stop execution
if not groq_api_key:
   
   st.error("GROQ_API_KEY is missing. Please set it in your environment variables.")
   st.stop()

# Initialize AI Agents for summarization and analysis
agents = ResearchAgents(groq_api_key)

# Initialize Rag
rag = Rag(groq_api_key)

# Initialize DataLoader for fetching research papers
data_loader = DataLoader()

# Initialize Embed_content for creating vectors
embed_content = Embed_content()

# Input field for the user to enter a research topic
query = st.text_input("Enter a research topic:")

# When the user clicks "Search"
if st.button("Search"):
    with st.spinner("Fetching research papers..."):  # Show a loading spinner
        
        # Fetch research papers from ArXiv and Google Scholar
        arxiv_papers = data_loader.fetch_arxiv_papers(query)
        #google_scholar_papers = data_loader.fetch_google_scholar_papers(query)
        all_papers = arxiv_papers #+ google_scholar_papers  # Combine results from both sources
        

        # If no papers are found, display an error message
        if not all_papers:
            st.error("Failed to fetch papers. Try again!")
        else:
            processed_papers = []

            # Process each paper: generate summary and key insights
            for paper in all_papers:
                #summary = agents.summarize_paper(paper['summary'])  # Generate summary
                key_insight = agents.key_insight(paper['summary'])  # key insights

                processed_papers.append({
                    "title": paper["title"],
                    "link": paper["pdf_url"],
                    "summary": paper['summary'],
                    "key_insight": key_insight,
                    "paper_content": paper["extracted_content"]
                })


            # Store the processed papers in session state
            st.session_state["processed_papers"] = processed_papers

# Display the processed research papers if available
if "processed_papers" in st.session_state:
    st.subheader("Top Research Papers:")
    for i, paper in enumerate(st.session_state["processed_papers"], 1):
        st.markdown(f"### {i}. {paper['title']}")  # Paper title
        st.markdown(f"🔗 [Read Paper]({paper['link']})")  # Paper link
        st.write(f"**Summary:** {paper['summary']}")  # Paper summary
        st.write(f"{paper['key_insight']}")  # key insight

        # Unique keys for each query input and button
        question = st.text_input(f"Enter your query from this paper:", key=f"question_{i}")

        # Unique session key to track button clicks
        clicked_key = f"submit_clicked_{i}"

        # Initialize session state variable if not set
        if clicked_key not in st.session_state:
            st.session_state[clicked_key] = False

        # Handle button click
        if st.button("Submit", key=f"submit_button_{i}"):
            st.session_state[clicked_key] = True  # Store click event

        # Process query only if the button was clicked
        if st.session_state[clicked_key]:
            vector = embed_content.create_vector([paper['paper_content']])
            retriever = rag.retriever(vector)
            que_ans_chain = rag.que_ans_chain()
            rag_chain = rag.rag_chain(retriever, que_ans_chain)
            response = rag_chain.invoke({'input': question})
            st.write(response['answer'])

            ## With a Streamlit expander
            with st.expander("Document similarity Search"):
                for j, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('---------------------')

        st.markdown("---")  # Separator between papers