import os
from dotenv import load_dotenv
from autogen import AssistantAgent


# Load environment variables
load_dotenv()

class ResearchAgents:
    def __init__(self, api_key):
        self.groq_api_key = api_key
        self.llm_config = {'config_list': [{'model': 'llama-3.3-70b-versatile', 'api_key': self.groq_api_key, 'api_type': "groq"}]}

        # Search Agent - Expand search for suggesting 1 related research topics
        self.search_agent = AssistantAgent(
            name="search_agent",
            system_message="Suggest 3 related research topics relevant to the user query",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            code_execution_config=False
        ) 

        # Summarizer Agent - Summarizes research papers
        '''self.summarizer_agent = AssistantAgent(
            name="summarizer_agent",
            system_message="Summarize the retrieved research papers and present concise summaries to the user, JUST GIVE THE RELEVANT SUMMARIES OF THE RESEARCH PAPER AND NOT YOUR THOUGHT PROCESS.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            code_execution_config=False
        )'''

        # Key insight Agent - provide methods and conclusion
        self.key_insight_agent = AssistantAgent(
            name="key_insight_agent",
            system_message="Analyze the research papers and provide a list of methods and conclusion for each paper in a pointwise format. JUST GIVE THE METHODS AND CONCLUSION, NOT YOUR THOUGHT PROCESS",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            code_execution_config=False
        )

    def summarize_paper(self, paper_summary):
        """Generates a summary of the research paper."""
        summary_response = self.summarizer_agent.generate_reply(
            messages=[{"role": "user", "content": f"Summarize this paper: {paper_summary}"}]
        )
        return summary_response.get("content", "Summarization failed!") if isinstance(summary_response, dict) else str(summary_response)

    def key_insight(self, summary):
        """Generates key insight of the research paper."""
        key_insight_response = self.key_insight_agent.generate_reply(
            messages=[{"role": "user", "content": f"Provide methods and conclusion for this paper: {summary}"}]
        )
        return key_insight_response.get("content", "Methods and Conclusion failed!")


        
    
    

    

