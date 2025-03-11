# AI - Powered Research Assistant

![Untitledvideo-MadewithClipchamp-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/5d5289af-016c-4a91-8f5e-160f16280986)

## Project Overview:

**AI Assistant - Research Paper Retrieval & Answer System:**
This cutting edge AI-driven tool fetches the relevant research papers from ArXiv and provide answer to user query from the research paper.

## Key Features:

* **Information Retrieval from Source (ArXiv):** Fetches 5 most relevant ArXiv papers based on the topic given by user. It provides the Title, Summary and pdf links from ArXiv.
* **Intuitive Summarization:** Features an innovative summarizing agent using open source Large Language Model (LLAMA 3) that generates key insights; methods and conclusion for each paper.
* **Semantic Data Processing:** Employs state-of-the-art embedding techniques to convert textual content into semantic vectors, facilitating precise information retrieval.
* **Dynamic Response Generation:** Employs a Retrieval-Augmented Generation (RAG) framework to provide accurate response to user queries with context citation, by dynamically sourcing information from PDF document.

## Technical Workflow:

* **Data Acquisition:** A helper function is created to work seamlessly with ArXiv API. It includes creating search query from user input and handling the API response strategically.
* **Automated Summarization:** The open source LLM (LLAMA3) processes the retrieved information to summarize the key insight; methods and conclusions, providing a quick digest of the article.
* **Content Segmentation:** After converting pdf to text, it is strategically segmented into manageable chunks, optimizing both computational resources and data relevancy.
* **Vector Embedding and Storage:** Transforms text segments into mathematical vectors using Sentence Transformers HuggingFace Embeddings, storing them in a FAISS vector database for rapid, similarity-based retrieval.
* **Semantic Query Processing:** When a query is received, the system identifies the most relevant text vectors, pulling contextually appropriate information for response generation.
* **AI-Driven Generation:** The open source LLM (LLAMA3) processes the retrieved information, crafting responses that are precise and human-like in their articulation, also quoting the context from the article.

## Benefits:

* **Efficiency:** Reduces the time spent to read through similar articles on the topic and get key insights. Extracts from the articles, such as summaries, methods and conclusion helps with analysis and comparison.
* **Accuracy:** By integrating RAG, the accuracy of answer is inproved, reducing the likelihood of misinformation.

## Future Enhancements:

* Integrate caching for faster responses.
* Deploy as cloud-based API for scaling and production purposes.
