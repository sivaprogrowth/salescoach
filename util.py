from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from dotenv import load_dotenv
import os,requests,io
from pydub import AudioSegment
import json
import wave
import fitz
import boto3
import tempfile
from botocore.exceptions import BotoCoreError, ClientError
import mimetypes
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
s3 = boto3.client('s3')
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AUDIO_DATA_FOLDER = os.getenv("AUDIO_DATA_FOLDER")

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

async def upload_file_to_pinecone(raw_text, file_name, index_name):
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
    try:
        for page in PDFPage.get_pages(path):
            interpreter.process_page(page)
        text = retstr.getvalue()
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")
    finally:
        device.close()
        retstr.close()
    return text 

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


def generate_QNA(title , objective , no_of_questions , idx):
    try:
        input  = title+"."+objective
        docs = find_match(input, idx)
        docs = ' '.join(docs.split()[:6000])  # limit to 1000 words

        # Prepare prompt for OpenAI completion
        prompt = f"""You are an AI assessment generator. Use the provided title, objective, and retrieved data to create a multiple-choice quiz. Follow these instructions carefully:

        Question Structure: Create clear, concise, and relevant multiple-choice questions based on the retrieved data. Ensure questions are directly related to the title and objective.
        Answer Options: Provide three distinct answer options for each question, labeled 1), 2), and 3). Ensure only one answer is correct while the others are plausible distractors.
        Answers Section: List the correct answers corresponding to the questions. Use only the correct answer option numbers (e.g., 1, 2, 3).
        The number of questions generated will be {no_of_questions}.

        Title:{title}
        Objective:{objective}
        retrieved data:{docs}
        ### Output Format (STRICT):
        {{
            "quiz": [
                {{
                    "question": "Your question here",
                    "options": "1) Option A, 2) Option B, 3) Option C",
                    "answer": "1"
                }},
                {{
                    "question": "Another question here",
                    "options": "1) Option X, 2) Option Y, 3) Option Z",
                    "answer": "1"
                }}
            ]
        }}

        Output ONLY a valid JSON in the exact structure shown below. Do not add explanations or commentary. 
        """
            
        # Get response from OpenAI's completion model
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
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
        answers = [item["answer"] for item in quiz_data["quiz"]]
        answers_str = json.dumps(answers)
        return formatted_questions , answers_str

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error occurred: {e}")
        return None, None


def download_audio(url, output_file, output_format="mp3"):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        audio = AudioSegment.from_ogg(io.BytesIO(response.content))
        audio.export(output_file, format=output_format)
        print(f"File downloaded and converted to {output_format.upper()}: {output_file}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the audio: {e}")
    except Exception as e:
        print(f"Error processing the audio: {e}")

def extract_text_from_pdf(pdf_stream):
    """
    Extracts text from a PDF file stream using PyMuPDF.
    """
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        text = chr(12).join([page.get_text() for page in doc])
    return text

def upload_audio_to_s3(audio_file_path):
    s3_folder = AUDIO_DATA_FOLDER
    audio_file_name = os.path.basename(audio_file_path)
    s3_key = os.path.join(s3_folder, audio_file_name)
    try:
        s3.upload_file(
            audio_file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': mimetypes.guess_type(audio_file_path)[0]}
        )
        audio_url = f"https://{S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_key}"
        print(f"Uploaded audio file to {audio_url}")
        return audio_url
    except Exception as e:
        print(f"Error uploading audio file to {S3_BUCKET_NAME}/{s3_key}: {e}")
        return None

def speak(text):
    audio_data = b''
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm"
    ) as response:
        for chunk in response.iter_bytes(1024):
            audio_data += chunk 

    return audio_data

def transcribe():
    print('i am in transcribe function..')
    audio_file= open("reply2.wav", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    language="en"
    )
    print(transcription.text)
    return transcription.text

def save_mp3_file(filename, audio_data, sample_rate=22050, num_channels=1):
    # Convert raw audio data into an AudioSegment
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # Assuming 16-bit audio
        frame_rate=sample_rate,
        channels=num_channels
    )
    
    # Export the audio segment as an MP3 file
    audio_segment.export(filename, format="mp3")

def download_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

        return response.content  # Return raw PDF content

    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None

def classify_sections_gpt(cv_text):

    function_schema = {
        "name": "extract_cv_sections",
        "description": "Extracts different sections from a resume and categorizes them into structured data.",
        "parameters": {
            "type": "object",
            "properties": {
                "Header": {"type": "string", "description": "Name, email, phone number, LinkedIn profile"},
                "Summary": {"type": "string", "description": "Professional summary or objective statement"},
                "Education": {"type": "string", "description": "Educational background including institutions and degrees"},
                "Experience": {"type": "string", "description": "Work experience with job titles, companies, and years"},
                "Skills": {"type": "string", "description": "List of relevant skills"},
                "Certifications": {"type": "string", "description": "Certifications, awards, and achievements"},
                "Projects": {"type": "string", "description": "Personal or professional projects"},
                "Additional_Info": {"type": "string", "description": "Other relevant details (languages, volunteer work, etc.)"}
            },
            "required": ["Header", "Education", "Experience", "Skills"]
        }
    }

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI expert at analyzing CVs."},
            {"role": "user", "content": f"Extract and categorize the following CV text:\n\n{cv_text}"}
        ],
        functions=[function_schema],  # Fix: Pass as a list
        function_call={"name": "extract_cv_sections"}  # Fix: Explicit function call
    )

    extracted_sections = json.loads(response.choices[0].message.function_call.arguments)
    return extracted_sections

def analyze_cv_with_gpt(text, prompt):
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"You are an expert CV reviewer. {prompt}"},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content.strip()

def cover_letter_with_gpt(cv, jd):
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"You are an expert Cover Letter writer. Write a cover letter for the following job description:\n\n{jd}. Use the provided CV to tailor the cover letter."},
            {"role": "user", "content": cv}
        ]
    )

    return response.choices[0].message.content.strip()

def job_title_with_gpt(cv):
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI expert at analyzing CVs. Identify the job title based on the provided CV. Just is needed in bullt points"},
            {"role": "user", "content": cv}
        ]
    )

    return response.choices[0].message.content.strip()

def classify_job_preferences_gpt(field_str):
    function_schema = {
        "name": "classify_job_preferences",
        "description": "Classifies job preferences into structured data.",
        "parameters": {
            "type": "object",
            "properties": {
                "Preferred_Location": {"type": "array", "items": {"type": "string"}, "description": "The geographical locations where the candidate prefers to work"},
                "Preferred_Job_Title": {"type": "array", "items": {"type": "string"}, "description": "The roles or job titles the candidate is seeking"},
                "Expected_Salary": {"type": "number", "description": "The lowest salary the candidate expects"}
            },
            "required": ["Preferred_Location", "Preferred_Job_Title", "Expected_Salary"]
        }
    }

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI expert at classifying job preferences."},
            {"role": "user", "content": f"Classify the following job preferences:\n\n{field_str}"}
        ],
        functions=[function_schema],
        function_call={"name": "classify_job_preferences"}
    )

    classified_preferences = json.loads(response.choices[0].message.function_call.arguments)
    return classified_preferences

def job_assistance_gpt(job_preferrance, query_str):
    prompt = f"""Goal: Think like an experienced career counsellor for 13-18 year old young girls in India. These girls come from lower income families and need guidance on making career choices. Please advise them on how to become {job_preferrance} and also provide them with some general advice regarding "{query_str}">
Return Format: Give up to 100 words describing all the areas of the advice. Use bullet points where necessary but also have formatted sections. Please give in English and Hindi languages.
Warnings: Be careful to make sure that the text should be simple, doesn't sound like AI generated and don't hallucinate. Also, follow ethical guidelines when dealing with adolescent girls. Be culturally sensitive and don't give biased advice based on religion, caste and other demographic details."""
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"You are an expert job assistant. {prompt} {query_str}"},
            {"role": "user", "content": job_preferrance}
        ]
    )

    return response.choices[0].message.content.strip()