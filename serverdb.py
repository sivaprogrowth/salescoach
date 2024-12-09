from fastapi import FastAPI, UploadFile, File, Form , status, Request , Response, HTTPException, Query
from fastapi.responses import FileResponse , JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from util import *
from service import *
from datetime import datetime
from pinecone import Pinecone
from botocore.exceptions import BotoCoreError, ClientError
from pydub import AudioSegment
from dotenv import load_dotenv
import scipy.io.wavfile as wavfile 
import numpy as np
import sounddevice as sd
import pyaudio
import os , json
import mysql.connector
import queue
import requests
import mimetypes
import io
import re
import boto3
import tempfile
import shutil
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
connection = mysql.connector.connect(
    user = 'admin',
    host = 'salescoachdatabase-1.ctkqiaw8wxwq.ap-south-1.rds.amazonaws.com',
    database = DATABASE,
    passwd = PASSWD
)

sns_client = boto3.client('sns', region_name='ap-south-1')
s3 = boto3.client('s3')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN') 
SNS_TOPIC_TEXT_RAG_ARN = os.getenv('SNS_TOPIC_TEXT_RAG_ARN')
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AUDIO_DATA_FOLDER = os.getenv("AUDIO_DATA_FOLDER")
# Glific API credentials
GLIFIC_PHONE_NUMBER = os.getenv("GLIFIC_PHONE_NUMBER")
GLIFIC_PASSWORD = os.getenv("GLIFIC_PASSWORD")
GLIFIC_AUTH_URL = os.getenv("GLIFIC_AUTH_URL")
GLIFIC_SEND_URL = os.getenv("GLIFIC_SEND_URL")
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
        return result + f"That was the feed back I had. Thanks for taking the test."
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

async def run_script(message: str, idx : str) -> str:
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
        index = pc.Index(idx)
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
        print("Error fetching auth token:", error)
        raise HTTPException(status_code=500, detail="Unable to fetch auth token")


async def send_to_glific_api(flow_id: int, contact_id: int, result: str):
    print("I am sending message:",result)
    try:
        auth_token = get_auth_token()
        graphql_query = """
        mutation resumeContactFlow($flowId: ID!, $contactId: ID!, $result: Json!) {
          resumeContactFlow(flowId: $flowId, contactId: $contactId, result: $result) {
            success
            errors {
                key
                message
            }
          }
        }
        """
        graphql_variables = {
            "flowId":flow_id,
            "contactId":contact_id,
            "result":f"{{\"result\":{{\"message\":\"{result}\"}}}}"
        }
        headers = {
            "authorization": f"{auth_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            GLIFIC_SEND_URL,
            json={
                "query": graphql_query,
                "variables": graphql_variables
            },
            headers=headers
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending data to Glific API: {e}")

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
    except ClientError as e:
        print(f"Error uploading audio file to {S3_BUCKET_NAME}/{s3_key}: {e}")
        return None
    
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

# Utility Function: Validate Email
def is_valid_email(email: str) -> bool:
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(email_regex, email))

# Utility Function: Fetch User by Email
def get_user_by_email(email: str):
    cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
    return cursor.fetchone()

# Utility Function: Add User to Database
def add_user(email: str, password: str):
    cursor.execute("INSERT INTO login (email, password) VALUES (%s, %s)", (email, password))
    connection.commit()
    
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
        print(SNS_TOPIC_ARN)
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
                },
                 'idx': {
                    'DataType': 'String',
                    'StringValue': request['idx']
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
        idx = message_attributes.get('idx', {}).get('Value')
        print(flow_id)
        print(contact_id)
        result  = await run_script(body['Message'],idx)
        result = repr(result).strip("'\"")
        #result_formatted = repr(result).strip("'").replace("\\", "\\\\")
        response =  await send_to_glific_api(flow_id , contact_id,result)
        print("sent to glific successfully: ",response)
    # If unknown message type, return 400 error
    else:
        raise HTTPException(status_code=400, detail="Unknown message type")

@app.post("/backend/fetchChatVoice")
async def fetchChatVoice(req: Request):

    print("Fetching chat voice...")
    content = await req.json()
    option = content['option']
    print(f"Option received: {option}")
    response = requests.post(f"https://salescoach.yogyabano.com/backend/fetch_chats", json={'index': option, 'user_id': 'shariq'}).json()

    if 'chat' not in response:
        print("Error: 'chat' key not found in the response.")
        return {'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR, 'message': 'Error fetching chat'}
    
    chats = response['chat']    
    length = len(chats) - 1
    audio_data = chats[length]['message']    
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    file_name = f"output_{datetime_string}.wav"
    print(f"Saving audio data to {file_name}...")

    save_wav_file(file_name, audio_data)
    audio_url = upload_audio_to_s3(file_name)
    print(f"Audio file uploaded to S3. URL: {audio_url}")
    
    # Remove the local file after uploading
    print(f"Removing local file {file_name}...")
    os.remove(file_name)
    return {'status_code': status.HTTP_200_OK, 'audio_url': audio_url, 'text':audio_data} 

@app.post("/backend/stopRecordingVoice")
async def fetchChatVoice(req: Request):
    content  = await req.json()
    url = content['url']
    download_audio(url,"reply2.wav", "wav")
    text = transcribe()
    qn = qn+1
    index = content['index']
    addToChat(type = 'User',message = text , index=index)
    reply = genReply(text,qn,index)
    addToChat(type = 'AI',message = reply,index = index)
    return {'status_code':status.HTTP_200_OK}

@app.post("/backend/clearChat")
async def fetchChatVoice(req: Request):
    content  = await req.json()
    index = content['index']
    user_id = 1
    cursor.execute(f"DELETE FROM chats WHERE `index` = {index}")
    return {'status_code':status.HTTP_200_OK}

app.post("/backend/publishTextRag")
async def publish_to_sns(request: Request):
    try:
        # Publish message to SNS with attributes
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
        print(SNS_TOPIC_TEXT_RAG_ARN)
        request = await request.json()
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=request['message'],
            MessageAttributes={
                'transactionId': {
                    'DataType': 'Number',
                    'StringValue': datetime_string
                },
                 'idx': {
                    'DataType': 'String',
                    'StringValue': request['idx']
                }
            }
        )
        print(f"Message sent to SNS, MessageId: {sns_response['MessageId']}")
        return {"status": "Message sent to SNS successfully", "MessageId": sns_response['MessageId'], "TransactionId":datetime_string}
    
    except (BotoCoreError, ClientError) as error:
        print(f"Failed to publish message to SNS: {error}")
        raise HTTPException(status_code=500, detail="Failed to send message to SNS")
    
@app.post("/backend/snsTextRag")
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
        transactionId = message_attributes.get('transactionId', {}).get('Value')
        idx = message_attributes.get('idx', {}).get('Value')
        result  = await run_script(body['Message'],idx)
        data = (
            result, # message
            transactionId,  # transactionId
            idx 
        )
        insert_query = """
                            INSERT INTO textRag (message, tranId, idx) 
                            VALUES (%s, %s, %s)
                        """
        cursor.execute(insert_query, data)
        connection.commit()
        print("sent to glific successfully: ",response)
    # If unknown message type, return 400 error
    else:
        raise HTTPException(status_code=400, detail="Unknown message type")
    
@app.post("/backend/retriever")
def getMessage(req :Request):

    tranId = req.json()['transactionId']
    cursor.execute(f"select message from textRag where tranId = '{tranId}'")
    result = cursor.fetchall()
    if result != []:
        return {"status_code": status.HTTP_200_OK, "Message": result[0][0]}
    else:
        print(f"Process in wait ")
        return {"status_code": status.HTTP_202_OK}

@app.post("/backend/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), idx: int = Form(...), file_name: str = Form(...)):
    # Create a temporary file
    if file is None:
        print("no file")
        raise HTTPException(status_code=400, detail="File not provided.")
    print("all okh")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_name = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
    
    try:
        # Open the temporary file in binary read mode and pass the file object
        with open(temp_file_name, "rb") as pdf_file:
            text = convert_pdf_to_txt_file(pdf_file)
        
        # Upload the extracted text to Pinecone
        upload_file_to_pinecone(text, file_name, idx)
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_name)
    
    return {"filename": file.filename, "content": text}


@app.post("/backend/login")
async def login(request: Request):
    data = await request.json()
    email = data.get("email")
    password = data.get("password")

    # Validate email format
    if not is_valid_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format."
        )

    # Fetch user by email
    user = get_user_by_email(email)

    if user:
        # Check if password matches
        if user[1] == password:  # Assuming `password` is the second column in `users`
            return {"status_code": status.HTTP_200_OK, "message": "Login successful"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password."
            )
    else:
        # Add new user to the database
        add_user(email, password)
        return {"status_code": status.HTTP_200_OK, "message": "User created successfully"}
    
@app.post("/backend/courses", status_code=status.HTTP_201_CREATED)
async def create_course(req: Request):
    try:
        # Parse the incoming JSON data
        data = await req.json()
        print(data)

        # Validation checks for required fields
        required_fields = ["title", "industry", "description", "company_id"]
        missing_fields = [field for field in required_fields if field not in data or not data[field]]

        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # If validation passes, proceed to create the course
        course_id = create_courses_service(data)
        return {"message": "Course created successfully", "course_id": course_id}

    except Exception as e:
        # Catch-all for any unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/backend/getCourses", status_code=status.HTTP_200_OK)
async def get_one_course(req:Request):
    print("i am here")
    data = await req.json()
    course_id = data['course_id']
    course = get_one_course_service(course_id)
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    return course

@app.post("/backend/allCourses", status_code=status.HTTP_200_OK)
async def get_all_courses(
    req: Request ,
    date: str = Query(None, description="Filter by a specific date (format: YYYY-MM-DD)"),
    latest_added: bool = Query(None, description="Filter by the most recently added courses"),
):
    data = await req.json()
    company_id = data['company_id']
    print("i am here")
    print(company_id)
    try:
        courses = get_all_courses_service(company_id=company_id, date=date, latest_added=latest_added)

        if not courses:
            raise HTTPException(status_code=404, detail="No courses found")

        return courses
    except Exception as e:
        return HTTPException(status_code=500, detail= str(e))
    
@app.post("/backend/updateCourses", status_code=status.HTTP_200_OK)
async def update_courses(req: Request):
    data = await req.json()
    course_id = data['course_id']
    update_course_service(course_id, data)
    return {"message": "Course updated successfully"}

@app.post("/backend/deleteCourses", status_code=status.HTTP_200_OK)
async def delete_courses(req: Request):
    data = await req.json()
    course_id = data['course_id']
    delete_course_services(course_id)
    return {"message": "Course deleted successfully"}

# Lessons APIs
@app.post("/backend/lessons",status_code=status.HTTP_201_CREATED)
async def create_lesson(req: Request,pdf_file: UploadFile = File(..., description="PDF file associated with the lesson")):
    try:
        data = await req.json()
        if pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_name = temp_file.name
                shutil.copyfileobj(pdf_file.file, temp_file)
            
            try:
                # Open the temporary file in binary read mode and pass the file object
                with open(temp_file_name, "rb") as pdf_file:
                    text = convert_pdf_to_txt_file(pdf_file)
                
                # Upload the extracted text to Pinecone
                upload_file_to_pinecone(text, pdf_file.file_name, pdf_file.file_name)
            finally:
                os.remove(temp_file_name)

        lesson_id = create_lesson(data,pdf_file.filename)
        # Validation checks for required fields
        required_fields = ["title", "course_id", "content"]
        missing_fields = [field for field in required_fields if field not in data or not data[field]]

        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # If validation passes, proceed to create the lesson
        lesson_id = create_lesson_service(data)
        return {"message": "Lesson created successfully", "lesson_id": lesson_id}

    except Exception as e:
        # Catch-all for any unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
@app.post("/backend/getLessons",status_code=status.HTTP_200_OK)
async def get_lesson(req: Request):
    data = await req.json()
    course_id = data['course_id']
    lesson = get_lessons_service(course_id)
    if not lesson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")
    return {"data": lesson}

@app.put("/backend/lessons",status_code=status.HTTP_200_OK)
def update_lesson(req: Request,pdf_file: UploadFile | None = File(None)):
    data = req.json()
    lesson_id = data['lesson_id']
    pdf_name = None
    if pdf_file is not None:
            
            #Deleting previous PDF
            prev_PDF = get_lesson_PDF(lesson_id)
            delete_index(prev_PDF)

            pdf_name = pdf_file.filename
            #Updating new pdf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_name = temp_file.name
                shutil.copyfileobj(pdf_file.file, temp_file)
            
            try:
                # Open the temporary file in binary read mode and pass the file object
                with open(temp_file_name, "rb") as pdf_file:
                    text = convert_pdf_to_txt_file(pdf_file)
                
                # Upload the extracted text to Pinecone
                upload_file_to_pinecone(text, pdf_file.file_name, pdf_file.file_name)
            finally:
                os.remove(temp_file_name)
    update_lesson(lesson_id, data,pdf_name)
    return {"message": "Lesson updated successfully"}

@app.delete("/backend/lessons",status_code=status.HTTP_200_OK)
async def delete_lesson(req: Request):
    data = await req.json()
    lesson_id = data['lesson_id']
    delete_lesson_service(lesson_id)
    return {"message": "Lesson deleted successfully"}

# Assessments APIs
@app.post("/backend/assessments",status_code=status.HTTP_201_CREATED)
def create_assessment(req: Request):
    data = req.json()
    assessment_id = create_assessment(data)
    return {"message": "Assessment created successfully", "assessment_id": assessment_id}

@app.post("/backend/assessments/{assessment_id}")
def get_assessment(assessment_id: int):
    assessment = get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")
    return {"status_code": status.HTTP_200_OK, "data": assessment}

@app.put("/backend/assessments/{assessment_id}")
def update_assessment(req: Request, assessment_id: int):
    data = req.json()
    update_assessment(assessment_id, data)
    return {"status_code": status.HTTP_200_OK, "message": "Assessment updated successfully"}

@app.delete("/backend/assessments/{assessment_id}")
def delete_assessment(assessment_id: int):
    delete_assessment(assessment_id)
    return {"status_code": status.HTTP_200_OK, "message": "Assessment deleted successfully"}

# Feedback APIs
@app.post("/backend/feedbacks",status_code=status.HTTP_201_CREATED)
async def create_feedback(req: Request):
    data = await req.json()
    feedback_id = create_feedback_service(data)
    return {"message": "Feedback created successfully", "feedback_id": feedback_id}

@app.post("/backend/getFeedbacks",status_code=status.HTTP_200_OK)
async def get_feedback(req: Request):
    data = await req.json()
    feedback_id = data['feedback_id']
    feedback = get_feedback_service(feedback_id)
    if not feedback:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    return feedback

@app.put("/backend/feedbacks",status_code=status.HTTP_200_OK)
def update_feedback(req: Request):
    data = req.json()
    feedback_id = data['feedback_id']
    update_feedback_service(feedback_id, data)
    return {"message": "Feedback updated successfully"}

@app.delete("/backend/feedbacks",status_code=status.HTTP_200_OK)
def delete_feedback(req : Request):
    data = req.json()
    feedback_id = data['feedback_id']
    delete_feedback_service(feedback_id)
    return {"message": "Feedback deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
