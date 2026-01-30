from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import create_retriever_tool
from langsmith import Client
from langchain_huggingface import HuggingFaceEmbeddings

from tools.pdf_qa_tool import pdf_qa_tool

load_dotenv()

from langchain_ollama import ChatOllama  # Changed import
# INDEX_DIR = "faiss_lcel_index"

class AgentService:
    def __init__(self,index_dir="faiss_lcel_index"):
        self.index_dir = index_dir
        self.chat_history = []
        self.agent = self._build_agent


    def _load_retriever(self):
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorStore = FAISS.load_local(self.index_dir, embeddings=embedding,allow_dangerous_deserialization=True)
        return vectorStore.as_retriever(search_kwargs={"k": 3})

    def _build_agent(self):
        model = ChatOllama(
            model="llama3.1:latest",
            temperature=0.1,
        )

        retriever_tools = create_retriever_tool(
            self._load_retriever(),
            "lcel_search",
            "Use this tool when searching for information about Langchain Expression Language (LCEL)."
        )
        tools = [retriever_tools,pdf_qa_tool]

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="""
            You are a helpful assistant that can search the web and retrieve information about LangChain Agent.
             Use the appropriate tools to answer user questions accurately.
             When using pdf_qa_tool ALWAYS include the complete 'Sources:' section in your response.,
            """

        )
        return agent

    def process_chat(self, user_input, chat_history):
        """Process a chat message and return the response"""
        messages = chat_history + [HumanMessage(content=user_input)]

        response = self.agent.invoke({
            "messages": messages
        })
        # Extract tools used
        tools_used = []
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tools_used.append(tool_call['name'])

        # Print summary
        if tools_used:
            print(f"\n🔧 Tools Used: {', '.join(set(tools_used))}\n")
        else:
            print("\n📝 No tools were used (direct response)\n")

        # Extract the final response
        return response["messages"][-1].content
    # return response["output"]


# if __name__ == '__main__':
#     chat_history = []
#
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break
#
#         response = process_chat(build_agent_executor(), user_input, chat_history)
#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=response))
#
#         print("Assistant:", response)