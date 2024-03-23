"""
This module is responsibled for the Augementation - Generation Step in the RAG App,
and is implemented an Agent Tool. As user send a query thru the Chat, the following sequence
happens in this module:
- The query (string) is first encoded to Embeddings by Qgrant FastEmbedding default model `BAAI/bge-small-en-v1.5`
- The query is passed to the Agent (backed by LLM) for it to determine intent
- Depending on the result of the first call returned by the LLM, the Agent then determine
if it wants to call `Retriever` from retrieve module
- Return the response to the caller

References:
https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
(You can customize your prompt following guidelines in)
https://community.openai.com/t/out-of-context-questions-in-retrieval-augmented-generation/434871
"""
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from weaviate import Client

LLM = "gpt-3.5-turbo-1106"
TEMPERATURE = 0
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"you are an assistant that help users answer question about the game Genshin Inpact. If you cannot answer the question based on retrieved context, "\
        f"tell the human you cannot answer the question, do not try to make up an answer.\n"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

class RAGController:
    def __init__(self):
        self.embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_store = Weaviate(
            client=Client(url=os.environ["WEAVIATE_URL"]),
            index_name="Document",
            text_key="text",
            embedding=self.embedding,
            by_text=False
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106", 
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0
        )

    def _init_agent_executor(self, search_kwargs):
        """
        Get a new instance of Agent Executor to clear all state
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs, return_source_documents=True
        )
        tools = [
            create_retriever_tool(retriever,
                             name="InformationRetrieval",
                             description="Searches and returns text from Genshin Impact document data.",
                             document_separator="\n\n\n"
            )
        ]
        agent = create_openai_tools_agent(self.llm, tools, PROMPT)
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,
                                                  return_intermediate_steps=True,
                                                  verbose=True, return_source_documents=True)


    def run(self, query: str, search_kwargs=None) -> str:
        agent_executor = self._init_agent_executor(search_kwargs)  # call to init_tools (tool need to be fresh each time)
        output = agent_executor.invoke({"input": query})
        return output["output"]