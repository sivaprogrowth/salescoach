from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.openai_info import OpenAICallbackHandler
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
import mysql.connector
from fastapi import HTTPException
import uuid

class SessionManager:
    def __init__(self):
        load_dotenv()
        self.setup_database()
        self.setup_llm()
        
    def setup_database(self):
        try:
            self.connection = mysql.connector.connect(
                user=os.getenv('DB_USER'),
                host=os.getenv('HOST'),
                database=os.getenv('DATABASE'),
                password=os.getenv('PASSWD')
            )
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Database connection failed: {err}")

    def setup_llm(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_session_id(self) -> str:
        return str(uuid.uuid4())
        
    def get_or_create_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not session_id:
            session_id = self.generate_session_id()
        
        with self.connection.cursor(dictionary=True) as cursor:
            query = "SELECT * FROM sessions WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            session = cursor.fetchone()
        
        if not session:
            session_id = self.generate_session_id()
            with self.connection.cursor() as cursor:
                query = "INSERT INTO sessions (session_id, created_at) VALUES (%s, %s)"
                cursor.execute(query, (session_id, datetime.now()))
                self.connection.commit()
        
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        
        if session:
            self.load_session_history(session_id, memory)
        
        return {"session": {"session_id": session_id}, "memory": memory}
        
    def process_pdf(self, file_path: str) -> str:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            splits = text_splitter.split_documents(pages)
            return "\n".join([split.page_content for split in splits])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")

    def process_message(self, session_id: Optional[str], input_content: str, input_type: str = "text") -> Dict[str, Any]:
        try:
            session_data = self.get_or_create_session(session_id)
            session_id = session_data["session"]["session_id"]
            memory = session_data["memory"]
            
            processed_input = self.process_pdf(input_content) if input_type == "pdf" else input_content
            
            chain = ConversationChain(llm=self.llm, memory=memory, verbose=True)
            
            callback = OpenAICallbackHandler()
            response = chain.run(processed_input, callbacks=[callback])
            
            self.store_in_history(session_id, processed_input, response, callback.total_tokens)
            
            return {"session_id": session_id, "response": response, "tokens_used": callback.total_tokens}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Message processing failed: {str(e)}")

    def store_in_history(self, session_id: str, input_text: str, response: str, tokens: int):
        query = """
        INSERT INTO message_history 
        (session_id, input_text, response, tokens_used, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (session_id, input_text, response, tokens, datetime.now()))
            self.connection.commit()

    def load_session_history(self, session_id: str, memory: ConversationBufferMemory):
        query = """
        SELECT input_text, response 
        FROM message_history 
        WHERE session_id = %s 
        ORDER BY timestamp ASC
        """
        with self.connection.cursor(dictionary=True) as cursor:
            cursor.execute(query, (session_id,))
            history = cursor.fetchall()
        
        for interaction in history:
            memory.save_context({"input": interaction["input_text"]}, {"output": interaction["response"]})

    def __del__(self):
        if hasattr(self, 'connection'):
            self.connection.close()

session = SessionManager()
app = FastAPI()

@app.post("/process_text")
async def process_text(message: str, session_id: Optional[str] = None):
    return session.process_message(session_id, message, input_type="text")

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=400, detail="Temporary file missing.")

    try:
        result = session.process_message(session_id, tmp_path, input_type="pdf")
        return result
    finally:
        os.remove(tmp_path)

@app.get("/get_history")
async def get_history(session_id: str):
    try:
        query = """
        SELECT input_text, response, timestamp 
        FROM message_history 
        WHERE session_id = %s 
        ORDER BY timestamp ASC
        """
        with session.connection.cursor(dictionary=True) as cursor:
            cursor.execute(query, (session_id,))
            history = cursor.fetchall()
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetching history failed: {str(e)}")
