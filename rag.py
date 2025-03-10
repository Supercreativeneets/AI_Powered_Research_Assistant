from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

class Rag:
    def __init__(self, api_key):
        self.groq_api_key = api_key
        self.llm = ChatGroq (
                            model_name='llama-3.3-70b-versatile',
                            api_key=self.groq_api_key,
                            )
        self.prompt = ChatPromptTemplate.from_messages(
                                                        [
                                                        ("system", system_prompt),
                                                        ("human", "{input}"),
                                                        ]
                                                        )

    # ✅ Create Question-Answer chain   
    def que_ans_chain(self):
        return create_stuff_documents_chain(self.llm, self.prompt)
    
    # ✅ Create Retriever
    def retriever(self,vector):
        return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    def rag_chain(self,retriever,que_ans_chain):
        return create_retrieval_chain(retriever,que_ans_chain)
