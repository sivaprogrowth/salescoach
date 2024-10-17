from fastapi import FastAPI, UploadFile, File, Form , status, Request
from fastapi.responses import FileResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import scipy.io.wavfile as wavfile  # To save audio in .wav format
from openai import OpenAI
from util import *
import torch
import whisper
from datetime import datetime
import pyaudio
import os , json
import mysql.connector
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import queue
load_dotenv()


connection = mysql.connector.connect(
    user = 'root',
    host = 'localhost',
    database = 'shariq',
    passwd = 'Shariq@123'
)

cursor = connection.cursor()

app = FastAPI()
questions = []
answers = []
qn = 0 
client  = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
p = pyaudio.PyAudio()
# Parameters for recording
sample_rate = 44100
channels = 1
audio_queue = queue.Queue()
stream = None

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getQnA(text):

    messages = [
        {
            "role": "system",
            "content": """
            You are an AI that excels at analyzing complex text and generating high-quality, knowledge-assessment questions and their respective answers. 
            Your task is to evaluate the provided text and generate 5 relevant questions that assess a person's understanding of key points, insights, or concepts from the material. 
            Each question should test knowledge of significant information, and you should also provide the best possible answer for each question.

            Please format the response as a strict list of dictionaries in the following JSON format:

            [
                {
                    "Question": ".....",
                    "Answer": "......"
                },
                {
                    "Question": ".....",
                    "Answer": "......"
                },
                {
                    "Question": ".....",
                    "Answer": "......"
                }
            ]

            Ensure that the output strictly adheres to this format so it can be converted into JSON and stored in a database.
            """
        },
        {
            "role": "user",
            "content": f"Here is the text: {text}"
        }
    ]

    qna = client.chat.completions.create(
        model="gpt-4o", 
        messages=messages,
        max_tokens=1000,
        temperature=0.7, 
        n=1,
        stop=None
    )
    qna = qna.choices[0].message.content
    return qna

def saveQnA(text , index):
    text = text.replace('```json', '').replace('```', '')
    print(text)
    qnas = json.loads(text)
    print (qnas)
    for idx , qna in enumerate(qnas):
        current_datetime = datetime.now()
        timestamp = int(current_datetime.timestamp())
        data = (
            timestamp,
            qna['Question'],   # question
            qna['Answer'],  # answer
            index # index
        )
        insert_query = """
                            INSERT INTO qna (id, question, answer, idx) 
                            VALUES (%s, %s, %s, %s)
                        """
        cursor.execute(insert_query, data)
        connection.commit()

def getQuestionsAnswers():
    cursor.execute("select question , answer from qna")
    result = cursor.fetchall()
    ques = []
    ans = []
    for qna in result:
        ques.append(qna[0])
        ans.append(qna[1])
    return ques , ans

stream_p = p.open(format=pyaudio.paInt16,  # Format: 16-bit PCM (Pulse Code Modulation)
                channels=1,              # Channels: 1 (Mono)
                rate=24000,              # Sample rate: 24,000 Hz (samples per second)
                output=True)  
def speak(text):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",                   # Specify the TTS model to use
        voice="alloy",                   # Specify the voice to use for TTS
        input=text,  # Input text to be converted to speech
        response_format="pcm"            # Response format: PCM (Pulse Code Modulation)
    ) as response:
        # Iterate over the audio chunks in the response
        for chunk in response.iter_bytes(1024):  # Read 1024 bytes at a time
            stream_p.write(chunk)


def transcribe():
    print('i am in transcribe function..')
    audio_file= open("reply.wav", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    print(transcription.text)
    return transcription.text

def addToChat(type , message):
    data = (
            0,
            type,   # question
            message,  # answer
            'shariq', # index
            1
        )
    insert_query = """
                    INSERT INTO chats (id, message_type, message_content, `index`, user_id) 
                    VALUES (%s, %s, %s, %s, %s)
                """
    cursor.execute(insert_query, data)
    connection.commit()

def genReply(text,qnum,index):
    global questions
    global answers
    answer = answers[qnum]

    message = f"User Message:\n {text} \n\n AI Message:{answer}"
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are an AI assisstant. you are provided two answer to the same question, one is the user answer and the other is the AU generated answer your task is to analyse both the answers and compare them and comment on the user answer that how they can improve there answers to get closer the the one answer generated using AI "},
            {
                "role": "user",
                "content": message
            }
        ]
    )
    result = result.choices[0].message.content
    if(qnum+1) == len(questions):
        return result
    else :
        return result + f"That was the feed back I had . now your next question is \n {questions[qn+1]}"

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error during recording: {status}")
    audio_queue.put(indata.copy())

def start_recording():
    global stream, audio_queue
    audio_queue.queue.clear()  # Clear the queue before starting
    stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback)
    stream.start()
    print("Recording started...")

def stop_recording():
    global stream
    print('I am in stop recording..')
    if stream is not None:
        print('i am here to stop stream')
        stream.stop()
        stream.close()
        stream = None
        print("Recording stopped.")
        
        # Save audio data
        audio_data = np.concatenate(list(audio_queue.queue))
        audio_data = np.int16(audio_data * 32767)  # Convert float32 [-1, 1] to int16
        wavfile.write('reply.wav', 44100, audio_data)
        text = transcribe()
        # speak(text)
        return text
    else:
        return "No recording in progress."
    
def fetchChat():
    cursor.execute("select message_type, message_content from chats")
    result = cursor.fetchall()
    print(result)
    chats = []
    if len(result)>0:
        for chat in result[0]:
            entry = {'type':chat[0],'message':chat[1]}
            chats.append(entry)
    return chats

@app.post("/fetch_questions/{index}")
async def receive_data(index: str):
    global questions
    global answers
    questions , answers = getQuestionsAnswers()
    return {'status_code': status.HTTP_200_OK}

@app.get("/fetch_chats")
async def receive_data(req: Request):
    global questions
    global qn
    contents = await req.json()
    chats = fetchChat()
    if len(chats) == 0:
        message = f'Hello i am Siva your AI proctor, Are you ready!! first question for you is, {questions[0]}'
        speak(message)
        addToChat(type = 'AI',message = message)
        chats = fetchChat()
    return {'status_code': status.HTTP_200_OK, 'chat':chats}


@app.post("/stopRecording")
async def upload_file(req: Request):
    global qn
    text = stop_recording()
    contents = await req.json()
    qn = qn+1
    index = contents['index']
    addToChat(type = 'User',message = text)
    reply = genReply(text,qn,index)
    speak(reply)
    addToChat(type = 'AI',message = reply)
    chats = fetchChat()
    return {'status_code':status.HTTP_200_OK,'chat':chats}



@app.post("/start-recording")
def start_recording_endpoint():
    start_recording()
    return {"message": "Recording started"}

@app.post("/createQuestionAnswer")
async def stop_recording_endpoint(req : Request):
    print(req)
    content = await req.json()
    print(content)
    index = content['index']
    text = content['text']
    print(index)
    print(text)
    qna = getQnA(text)
    saveQnA(qna,index)
    return{'status_code':status.HTTP_200_OK}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
