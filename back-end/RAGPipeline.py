from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_community.tools.tavily_search import TavilySearchResults


from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph
from pprint import pprint

from GraphState import GraphState


class RAGPipeline():



    LLM_name = "llama3"
    urls = [
        "https://en.wikipedia.org/wiki/The_House_in_Fata_Morgana",
        "https://en.wikipedia.org/wiki/Muv-Luv",
        "https://en.wikipedia.org/wiki/YU-NO:_A_Girl_Who_Chants_Love_at_the_Bound_of_this_World",
    ]

    def __init__(self):

        #Load documents from the web
        documents = [WebBaseLoader(url).load() for url in self.urls]


        #Flatten list
        documents_list = [items for sublist in documents for items in sublist]

        #Use RecursiveCharacterTextSplitter to split the text into chunks using the TikToken encoder
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0,
        )

        #Split the documents into chunks
        document_chunks = text_splitter.split_documents(documents_list)

        #Create a vector store
        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
        )

        self.retriever = vectorstore.as_retriever()

        #Instantiate all of the LLM models

        #Temparature is how much the model will answer creatively
        self.retrieval_llm = ChatOllama(model=self.LLM_name, format="json", temperature=0)

        #Promt to ask the user to grade the relevance of a document to a question
        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )

        # Pipe | instantiates promt outputs it to the LLM model and then parses the output
        self.retrieval_grader = self.prompt | self.retrieval_llm | JsonOutputParser()


        #Normal generic prompt for llama3
        self.normal_llm = ChatOllama(model=self.LLM_name, temperature=0)

        self.prompt = PromptTemplate(
            template="""<|eof_id|><|start_header_id|>user<|end_header_id|> {question} <|eof_id|><|start_header_id|>assistant<|end_header_id>""",
            input_variables=["question"],
        )

        self.normalLLm = self.prompt | self.normal_llm | StrOutputParser()


        #Generation using the RAG pipeline
        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "document"],
        )

        self.rag_llm = ChatOllama(model=self.LLM_name, temperature=0)
        self.rag_chain = self.prompt | self.rag_llm | StrOutputParser()



        # Hallucination Grader

        self.hallucination_llm = ChatOllama(model=self.LLM_name, format="json",temperature=0)

        self.prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )

        self.hallucination_grader = self.prompt | self.hallucination_llm | JsonOutputParser()



        # Answer Grader

        self.grader_llm = ChatOllama(model=self.LLM_name, format="json",temperature=0)

        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )

        self.answer_grader = self.prompt | self.grader_llm | JsonOutputParser()

        self.createWorkflow()





    #Graph Functions

    def retrieve(self, state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
            
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def generate_normal(self, state):
        """
        Generate answer using normal LLM

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Normal generation
        generation = self.normalLLm.invoke({"question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not useful"


    def createWorkflow(self):

        # Create the workflow graph
        self.workflow = StateGraph(GraphState)
        # Add nodes to the graph
        self.workflow.add_node("retrieve", self.retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.generate)  # generate
        self.workflow.add_node("generate_normal", self.generate_normal)  # generate


        # Make graph connections
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")

        # Check if documents are relevant to the question
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
            },
        )
        # Checks if the RAG pipeline LLM generation is grounded in the documents and answers the question
        # If its not grounded, it will re-try the generation using the normal LLM
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "useful": END,
                "not useful": "generate_normal",
            },
        )
        # Check if the generation is grounded in the documents and answers the question
        self.workflow.add_conditional_edges(
            "generate_normal",
            self.grade_generation_v_documents_and_question,
            {
                "useful": END,
                "not useful": "generate_normal",
            },
        )

        # Compile
        self.app = self.workflow.compile()

    def askQuestion(self, question):
        inputs = {"question": question}
        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
        pprint(value["generation"])
        return value["generation"]