from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv


# FastAPI App
app = FastAPI(
    title="Hadith RAG API with Memory",
    description="Ask Hadith questions with chat memory",
    version="1.0"
)

# Database Setup
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

# Load API keys
load_dotenv("app.env")

cohereApi = os.getenv('COHERE_API')
googleApi = os.getenv('GOOGLE_API_KEY')

# LLM (Cohere)
llm = ChatCohere(cohere_api_key=cohereApi, model="command-r-plus")

# Embeddings (Gemini)
model_em = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=googleApi
)

# Vector DB (load only - no rebuilding)
saveToDir = "./chroma_db"
vector_db = Chroma(persist_directory=saveToDir, embedding_function=model_em)

# Prompt Template
prompt_template = """
أنت مساعد ذكي مدرّب على الأحاديث.
جاوب على السؤال التالي فقط من الأحاديث المتاحة.
لو لقيت إجابة، ارجع الإجابة ثم أضف في الآخر "(استنادًا إلى الحديث)" مع ذكر المصدر لو موجود.
لو ملقيتش أي إجابة قول "لا يوجد إجابة".

السؤال: {question}
المحادثة السابقة: {chat_history}
المصادر: {context}

الإجابة:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template=prompt_template,
)

# Conversational RAG Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)


# Request Schema
class Query(BaseModel):
    question: str


# API Endpoints
@app.post("/ask")
def ask_question(query: Query):
    result = qa_chain({"question": query.question})
    answer = result["answer"]

    # Save to DB
    db_session = SessionLocal()
    log = RequestLog(question=query.question, answer=answer)
    db_session.add(log)
    db_session.commit()
    db_session.close()

    return {"answer": answer, "chat_history": [str(m) for m in memory.chat_memory.messages]}


@app.get("/")
def root():
    return {"message": "Hadith RAG API with Memory is running. Send POST request to /ask"}
