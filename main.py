from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import Cohere
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
import os
from dotenv import load_dotenv


app = FastAPI(title="Hadith RAG API with Memory", description="Ask Hadith questions with chat memory", version="1.0")


DATABASE_URL = "sqlite:///./requests.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class RequestLog(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)

Base.metadata.create_all(bind=engine)


load_dotenv("app.env") 

cohereApi = os.getenv('COHERE_API')
llm = Cohere(cohere_api_key=cohereApi, model="command-r-plus")


df = pd.read_csv("hf://datasets/Ahmedhany216/Islamic-Books/my_dataframe.csv")
df.drop_duplicates(subset=["hadith"], inplace=True)
llmLoader = DataFrameLoader(df, page_content_column="hadith")
llmData = llmLoader.load()

text_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=800,
    chunk_overlap=100
)
tokens_chunks = text_splitter.create_documents(
    [i.page_content for i in llmData],
    metadatas=[i.metadata for i in llmData]
)

model_em = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
saveToDir = "./chroma_db"
docs_ids = [str(i) for i in range(len(tokens_chunks))]
vector_db = Chroma.from_documents(tokens_chunks, model_em, persist_directory=saveToDir, ids=docs_ids)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory
)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    result = qa_chain({"question": query.question})
    answer = result["answer"]

    # حفظ السؤال والإجابة في قاعدة البيانات
    db_session = SessionLocal()
    log = RequestLog(question=query.question, answer=answer)
    db_session.add(log)
    db_session.commit()
    db_session.close()

    return {"answer": answer, "chat_history": [str(m) for m in memory.chat_memory.messages]}

@app.get("/")
def root():
    return {"message": "Hadith RAG API with Memory is running. Send POST request to /ask"}
