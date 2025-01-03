from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from dotenv import load_dotenv
import os
import json
import wave
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import tiktoken

OK = "OK"

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

pc = Pinecone()
# openai_vectorizer = OpenAIEmbeddings() <- uncomment this 
embeddings = OpenAIEmbeddings()

def get_index_list():
    return pc.list_indexes()

def index_init(name: str, dims: int=1536):
    if name not in get_index_list().names():
        pc.create_index(
            name=name,
            dimension=dims,
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
    return get_index(name)
    
def get_index(index_name: str):
    idx = pc.Index(name = index_name)
    return idx

def get_all_docs(index_name: str):
    index = get_index(index_name)
    result = index.query(
        vector=[0] * index.describe_index_stats().get('dimension', 1536),
        top_k=1000,
        include_metadata=True
    )
    match_text = ''
    for match in result.get('matches', []):
        match_text += match.get('metadata', {}).get('text', '') + "\n"
        
    return match_text
    
    


def find_match(input, index_name,k = 1):
    # input_em = openai_vectorizer.embed_query(input)
    input_em = embeddings.embed_query(input) 
    index = get_index(index_name)
    result = index.query(vector=input_em, top_k=k, includeMetadata=True, include_values=False)
    match_text = ''
    for match in result.get('matches', []):
        match_text += match.get('metadata', {}).get('text', '') + "\n"
    return match_text

def query_refiner(conversation, query):

    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0,
    max_tokens=256,)
    return response.choices[0].text
    
    # return query
    
def generate_quiz(text, num):
    batches = 1
    texts = [text]
    toks = num_tokens_from_string(text)
    if toks > 4000:
        batches = toks // 4096 
        # divide the text into batches
        texts = [text[i:i+4096] for i in range(0, len(text), 4096)]
        
    ans = ""
        
    for i in range(min(batches, 2)):
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role":"user", "content": f"Given the following text, generate {max(num // batches, 1)} quiz questions with multiple choice options to help the user revise the knowledge present in the text. Do not output any text like 'Sure...', just output a list of questions. \n\n Each question should be formatted as follows:\n\n <Question> \n\n a) <Option> \n b) <Option> \n c) <Option> \n d) Option \n\n Text: {texts[i]}\n\n"}],
        temperature=0.2,
        max_tokens=1024,)
        ans += response.choices[0].message.content or ""
    
    return ans

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def upload_file_to_pinecone(raw_text, file_name, index_name):
    # Splitting up the text into smaller chunks for indexing
    try:
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 10000,
            chunk_overlap  = 200, #striding over the text
            length_function = len,
        )
        
        texts = text_splitter.split_text(raw_text)
        index = index_init(index_name)
        embeddings = OpenAIEmbeddings()
        batch_size = max(len(texts) // 10, 1)

        for i in range(0, len(texts), batch_size):
            embeds = []
            batch = texts[i:i+batch_size]
            vectors = embeddings.embed_documents(batch)
            for j, vector in enumerate(vectors):
                embed = {'id': f'{i}_{j}', "values": vector, "metadata": {"file_name": file_name, "text": batch[j]}}
                embeds.append(embed)
            
            index.upsert(
                vectors=embeds
            )
        
        return OK
        
    except Exception as e:
        return e


def transcribe_audio(path):
    audio_file = open(path, "rb")
    translation = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file
    )
    return translation.text


def video_to_text(path):
    pass

    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    frames = len(base64Frames)
    BATCH_SIZE=1000
    text = ""
    
    for i in range(0, frames, BATCH_SIZE):
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "These are frames from a video uploaded to a company's knowledge base. Generate an elaborate description of what's happening in the video, which can be used to ask further questions by users about the video.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[i:i+BATCH_SIZE:50]),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 2048,
        }
        
        result = client.chat.completions.create(**params)
        text += result.choices[0].message.content
    return frames, text

def audio_to_text(path):
    audio_file = open(path, "rb")
    translation = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file
    )
    return translation.text

def save_wav_file(filename, audio_data, sample_rate=22050, num_channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # assuming 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


def convert_pdf_to_txt_file(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()

    device.close()
    retstr.close()
    return t 

def delete_index(index_name: str):
    # Get the list of existing indexes
    existing_indexes = get_index_list().names()
    index_name = index_name.split(".")[-2]
    print(existing_indexes)
    # Check if the index exists
    if index_name in existing_indexes:
        print(f"Index '{index_name}' found. Deleting it...")
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")


def generate_QNA(title , objective , no_of_questions , idx, retry_count=0, max_retries=5):
    try:
        input  = title+"."+objective
        docs = find_match(input, idx)

        format = """
                {
                    "quiz": [
                        {
                            "question": "Your question here",
                            "options": "1) Option A, 2) Option B, 3) Option C"
                        },
                        {
                            "question": "Another question here",
                            "options": "1) Option X, 2) Option Y, 3) Option Z"
                        }
                    ],
                    "answers": {
                        "1": "1",
                        "2": "1"
                    }
                }
                """
        
        # Prepare prompt for OpenAI completion
        prompt = f"""You are an AI assessment generator. Use the provided title, objective, and retrieved data to create a multiple-choice quiz. Follow these instructions carefully:

        Question Structure: Create clear, concise, and relevant multiple-choice questions based on the retrieved data. Ensure questions are directly related to the title and objective.
        Answer Options: Provide three distinct answer options for each question, labeled 1), 2), and 3). Ensure only one answer is correct while the others are plausible distractors.
        Answers Section: List the correct answers corresponding to the questions. Use only the correct answer option numbers (e.g., 1, 2, 3).
        The number of questions generated will be {no_of_questions}.

        Title:{title}
        Objective:{objective}
        retrieved data:{docs}
        Output Format:{format}

        follow the output format strictly . 
        """
            
        # Get response from OpenAI's completion model
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        print("OpenAI response received")
        res = completion.choices[0].message.content
        print(res)
        # Remove extra whitespace
        cleaned_output = res.strip()
        quiz_data = json.loads(cleaned_output)
        formatted_questions = [
            {
                "question": item["question"],
                "options": item["options"]
            }
            for item in quiz_data['quiz']
        ]
        answers_str = json.dumps(list(quiz_data['answers']))
        return formatted_questions , answers_str

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error occurred: {e}")

        # Retry logic
        if retry_count < max_retries:
            print(f"Retrying... Attempt {retry_count + 1} of {max_retries}")
            return generate_QNA(title, objective, no_of_questions, idx, retry_count + 1, max_retries)
        else:
            print("Max retries reached. Returning empty data.")
            return [], "[]"

