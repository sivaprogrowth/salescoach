from fastapi import FastAPI, UploadFile, File, Form , status, Request , Response, HTTPException
from fastapi.responses import FileResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from util import *
from datetime import datetime
from pinecone import Pinecone
from botocore.exceptions import BotoCoreError, ClientError
import scipy.io.wavfile as wavfile 
import pyaudio
import os , json
import mysql.connector
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import queue
import requests
import boto3
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
connection = mysql.connector.connect(
    user = 'root',
    host = 'localhost',
    database = DATABASE,
    passwd = PASSWD
)

sns_client = boto3.client('sns', region_name='ap-south-1')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN') 
# Glific API credentials
GLIFIC_PHONE_NUMBER = "918420925890"  # Update with the required phone number
GLIFIC_PASSWORD = "ALLAHuAKBAR@123"   # Update with the required password
GLIFIC_AUTH_URL = "https://api.yogyabano.glific.com/api/v1/session"
GLIFIC_SEND_URL = "https://api.yogyabano.glific.com/api"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_database_name = os.getenv("PINECONE_DATABASE_NAME")

cursor = connection.cursor()

app = FastAPI()
questions = []
answers = []
qn = 0 
client  = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
p = pyaudio.PyAudio()
sample_rate = 44100
channels = 1
audio_queue = queue.Queue()
stream = None

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def getQuestionsAnswers(index):
    cursor.execute(f"select question , answer from qna where idx = '{index}'")
    result = cursor.fetchall()
    ques = []
    ans = []
    for qna in result:
        ques.append(qna[0])
        ans.append(qna[1])
    return ques , ans

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
    file=audio_file
    )
    print(transcription.text)
    return transcription.text

def addToChat(type , message , index):
    data = (
            0,
            type,   
            message, 
            index, 
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
            {"role": "system", "content": """You are an expert sales coach. You have experience in evaluating sales pitches and responses of salespersons and give them quality and unbiased focused feedback. You can also rate the responses from the salespersons with the ideal responses.
 For evaluation,you are provided two responses to the same question, one is the response from the salesperson and the other is the AI generated answer. Your task is to analyse both the answers and compare them and provide the following details
Rate the response from the salesperson against the ideal response for the question. Give your rating on a scale of 5 where 5 is the best and 1 is the worst.
Give 3 bulleted comments to the salesperson on how they can improve there answers to get closer the the one answer generated using AI
Give 4 action items that the salesperson has to follow for the next 1 month to improve their sales pitch
It is compulsory for you to rate the salesperson response. """},
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
        return text
    else:
        return "No recording in progress."
    
def fetchChat(index):
    cursor.execute(f"select message_type, message_content from chats where `index` = '{index}'")
    result = cursor.fetchall()
    chats = []
    if len(result)>0:
        for chat in result:
            entry = {'type':chat[0],'message':chat[1]}
            chats.append(entry)
    return chats

async def run_script(message: str) -> str:
    try:
        print('Message received from user:', message)
        
        # Process the message and generate an embedding
        response = client.embeddings.create(
            input=[message],
            model="text-embedding-ada-002",
        )
        embedding = response.data[0].embedding
        print('Embeddings created')
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_database_name)
        # Send embedding to Pinecone and perform a query
        pinecone_response = index.query(vector=embedding, top_k=1,include_metadata=True,)
        pinecone_result = "\n".join([match['metadata']['text'] for match in pinecone_response['matches']])
        print('Pinecone query completed')
        
        # Prepare prompt for OpenAI completion
        prompt = f"Based on the following data:\n{pinecone_result}\nGenerate a response to the query message {message}."
        
        # Get response from OpenAI's completion model
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        print("OpenAI response received")
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error in run_script: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

# Function to get the auth token from the Glific authentication API
def get_auth_token():
    try:
        response = requests.post(
            GLIFIC_AUTH_URL,
            json={
                "user": {
                    "phone": GLIFIC_PHONE_NUMBER,
                    "password": GLIFIC_PASSWORD,
                }
            }
        )
        response.raise_for_status()
        auth_token = response.json().get("data", {}).get("access_token")
        if not auth_token:
            raise HTTPException(status_code=500, detail="Auth token not received")
        print("Auth token received:", auth_token)
        return auth_token
    except requests.HTTPError as error:
        print("Error fetching auth token:", error)
        raise HTTPException(status_code=500, detail="Unable to fetch auth token")


def send_to_glific_api(flow_id: int, contact_id: int, result: str):
    try:
        auth_token = get_auth_token()
        response = requests.post(
            GLIFIC_SEND_URL,
            json={
                "flowId": flow_id,
                "contactId": contact_id,
                "result": json.dumps({"message": result}),
            },
            headers={
                "authorization": f"{auth_token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        print("Successfully sent data to Glific API:", response.json())
        return {"status": "Success", "data": response.json()}
    except requests.HTTPError as error:
        print("Error sending data to Glific API:", error)
        raise HTTPException(status_code=500, detail="Error sending data to Glific API")


@app.post("/backend/generateAUD")
async def stop_recording_endpoint(req : Request):
    content = await req.json()
    text = content['text']
    audio_data = speak(text)
    headers = {
        'Content-Disposition': 'inline; filename="response.wav"',
        'Content-Type': 'audio/wav'
    }

    return Response(content=audio_data, media_type="audio/wav", headers=headers)

@app.post("/backend/fetch_questions/{index}")
async def receive_data(index: str):
    global questions
    global answers
    print(index)
    questions , answers = getQuestionsAnswers(index)
    if(len(questions)==0):
        message  = "Please Upload a file to fetch Questions."
    else:
        message  = "Ok"
        print(questions)
    return {'status_code': status.HTTP_200_OK, 'message':message}

@app.post("/backend/fetch_chats")
async def receive_data(req: Request):
    global questions
    global qn
    print("json = ",req.json)
    contents = await req.json()
    print("index is ",contents['index'])
    chats = fetchChat(contents['index'])
    print("first chat fetch done")
    if len(chats) == 0:
        message = f'Hello i am Siva your AI proctor, Are you ready!! first question for you is, {questions[0]}'
        addToChat(type = 'AI',message = message,index = contents['index'])
        chats = fetchChat(contents['index'])
    return {'status_code': status.HTTP_200_OK, 'chat':chats}


@app.post("/backend/stopRecording")
async def upload_file(req: Request):
    global qn
    text = transcribe()
    contents = await req.json()
    qn = qn+1
    index = contents['index']
    addToChat(type = 'User',message = text , index=index)
    reply = genReply(text,qn,index)
    addToChat(type = 'AI',message = reply,index = index)
    chats = fetchChat(index)
    return {'status_code':status.HTTP_200_OK,'chat':chats}



@app.post("/backend/start-recording")
def start_recording_endpoint():
    start_recording()
    return {"message": "Recording started"}

@app.post("/backend/createQuestionAnswer")
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

@app.post("/backend/publish")
async def publish_to_sns(request: Request):
    try:
        # Publish message to SNS with attributes
        request = await request.json()
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=request['message'],
            MessageAttributes={
                'contactId': {
                    'DataType': 'Number',
                    'StringValue': str(request['contactId'])
                },
                'flowId': {
                    'DataType': 'Number',
                    'StringValue': str(request['flowId'])
                }
            }
        )
        print(f"Message sent to SNS, MessageId: {sns_response['MessageId']}")
        return {"status": "Message sent to SNS successfully", "MessageId": sns_response['MessageId']}

    except (BotoCoreError, ClientError) as error:
        print(f"Failed to publish message to SNS: {error}")
        raise HTTPException(status_code=500, detail="Failed to send message to SNS")
    


@app.post("/backend/sns")
async def sns_listener(request: Request):
    headers = request.headers
    message_type = headers.get("x-amz-sns-message-type")

    if not message_type:
        raise HTTPException(status_code=400, detail="Invalid message type header")

    body = await request.json()

    # Check if it's a SubscriptionConfirmation message
    if message_type == "SubscriptionConfirmation" and body['SubscribeURL']:
        # Confirm the subscription
        response = requests.get(body['SubscribeURL'])
        if response.status_code == 200:
            print("Subscription confirmed.")
        else:
            print("Failed to confirm subscription.")
        return {"status": "Subscription confirmation handled"}

    # Handle Notification message
    elif message_type == "Notification":
        print(f"Received notification: {body['Message']}")
        # Retrieve contactId and flowId from MessageAttributes
        message_attributes = body['MessageAttributes'] or {}
        contact_id = message_attributes.get('contactId', {}).get('Value')
        flow_id = message_attributes.get('flowId', {}).get('Value')
        result  = await run_script(body['Message'])
        await send_to_glific_api(result , flow_id , contact_id)
    # If unknown message type, return 400 error
    raise HTTPException(status_code=400, detail="Unknown message type")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
