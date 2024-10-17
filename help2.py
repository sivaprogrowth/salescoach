from fastapi import FastAPI, BackgroundTasks, Request
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os, json
load_dotenv()
import mysql.connector

connection = mysql.connector.connect(
    user = 'root',
    host = 'localhost',
    database = 'shariq',
    passwd = 'Shariq@123'
)

cursor = connection.cursor()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI()

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
    lines = text.splitlines()
    if len(lines) > 2:
        text = '\n'.join(lines[2:-1])
    else:
        text = ''
    qnas = json.loads(text)
    for idx , qna in enumerate(qnas):
        data = (
            idx,
            qna['Question'],   # question
            qna['Answer'],  # answer
            index # index
        )
        insert_query = """
                            INSERT INTO question_answer (id, question, answer, `index`) 
                            VALUES (%s, %s, %s, %s)
                        """
        # Execute the insert query
        cursor.execute(insert_query, data)

        # Commit the transaction
        connection.commit()



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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)