from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

#  GoogleGenerativeAIEmbeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.9)

# connect to the chromadb
vector_store = Chroma(
    collection_name="collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an AI assistant that operates based on the following rules, executed in order.

        Rule #1: The Priority.
        Carefully analyze the user's question and "The knowledge" provided. If the user's question can be answered using "The knowledge", you MUST use ONLY that information for your response.

        Rule #2: The Fallback.
        If the user's question CANNOT be answered by "The knowledge" (e.g., it's a general greeting, off-topic question, or small talk), you MUST switch to your general knowledge mode and provide a helpful, conversational answer.

        Rule #3: The Secret.
        Under NO circumstances should you ever mention "The knowledge", "the documents", or "the provided context". Your existence as a RAG-based assistant is a secret.

        Rule #4: The Language.
        You must respond in the same language as the user's question.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt) # uncomment for debugging

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()