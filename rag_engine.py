import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false" 

def build_ai_engine():
    print("1. Creating the AI Brain and Embedding Models...")
    
    # Fix for Streamlit Cloud: manually create the cache folder so the model can download
    os.makedirs(os.path.expanduser("~/.cache/gpt4all"), exist_ok=True)
    
    # SECRET HACK: Using stable OpenAI software to talk to Google's free servers!
    llm = ChatOpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant"
    )
    
    # Using local computer for Embeddings
    embeddings = GPT4AllEmbeddings() 

    print("2. Loading the Syllabus...")
    loader = TextLoader("syllabus.txt")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("3. Building the Vector Database...")
    vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    print("4. Gluing everything together into a Chain...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful college teaching assistant. Use the provided context to answer the student's question. Context: {context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    chain = build_ai_engine()
    print("\n✅ RAG Engine is Ready!")
    question = "When is the midterm exam?"
    print(f"User: {question}")
    response = chain.invoke(question)
    print(f"AI: {response}")