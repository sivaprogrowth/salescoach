from dotenv import load_dotenv
from util import *
import os , json
import mysql.connector
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
USER = os.getenv('DB_USER')
HOST = os.getenv('HOST')
connection = mysql.connector.connect(
    user = USER,
    host = HOST,
    database = DATABASE,
    passwd = PASSWD
)

db = connection.cursor()

def create_courses_service(data):
    query = """
    INSERT INTO courses (title, industry, description, company_id, created_at, updated_at)
    VALUES (%s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (data['title'], data['industry'], data['description'], data['company_id']))
    connection.commit()
    return db.lastrowid

def get_one_course_service(course_id):
    query = """
        SELECT 
        c.id AS course_id,
        c.title,
        c.industry,
        c.description,
        c.company_id,
        c.created_at,
        c.updated_at,
        COUNT(DISTINCT l.id) AS lesson_count,
        COUNT(DISTINCT a.id) AS assessment_count,
        COUNT(DISTINCT f.id) AS feedback_count
        FROM 
            courses c
        LEFT JOIN 
            lessons l ON c.id = l.course_id
        LEFT JOIN 
            assessments a ON c.id = a.course_id
        LEFT JOIN 
            feedbacks f ON c.id = f.course_id
        WHERE c.id = %s
    """
    db.execute(query, (course_id,))
    result = db.fetchone()
    result_fomatted = {
        'course_id':result[0],
        'title':result[1],
        'industry':result[2],
        'description':result[3],
        'company_id':result[4],
        'created_at':result[5],
        'updated_at':result[6],
        'lesson_count':result[7],
        'assessment_count':result[8],
        'feedback_count':result[9]
    }
    return result_fomatted

def update_course_service(course_id, data):
    # Start building the query
    query = "UPDATE courses SET "
    updates = []
    values = []

    # Add fields to be updated dynamically
    if "title" in data:
        updates.append("title = %s")
        values.append(data["title"])
    if "industry" in data:
        updates.append("industry = %s")
        values.append(data["industry"])
    if "description" in data:
        updates.append("description = %s")
        values.append(data["description"])
    if "company_id" in data:
        updates.append("company_id = %s")
        values.append(data["company_id"])

    # Add the updated_at field
    updates.append("updated_at = NOW()")

    # Combine updates into the query
    query += ", ".join(updates)
    query += " WHERE id = %s"
    values.append(course_id)

    # Execute the query
    db.execute(query, tuple(values))
    connection.commit()

def delete_course_services(course_id):
    query = "DELETE FROM courses WHERE id = %s"
    db.execute(query, (course_id,))
    connection.commit()

def get_all_courses_service(company_id, date=None, latest_added=None):
    """
    Fetch courses filtered by company_id with optional date and latest_added filters.
    """
    # Base query with JOINs for counts
    query = """
        SELECT 
            c.id AS course_id,
            c.title,
            c.industry,
            c.description,
            c.company_id,
            c.created_at,
            c.updated_at,
            COALESCE(COUNT(DISTINCT l.id), 0) AS lesson_count,
            COALESCE(COUNT(DISTINCT a.id), 0) AS assessment_count,
            COALESCE(COUNT(DISTINCT f.id), 0) AS feedback_count
        FROM 
            courses c
        LEFT JOIN lessons l ON c.id = l.course_id
        LEFT JOIN assessments a ON c.id = a.course_id
        LEFT JOIN feedbacks f ON c.id = f.course_id
        WHERE c.company_id = %s
    """

    # Filtering and ordering logic
    filters = []
    values = [company_id]

    if date:
        filters.append("c.created_at::date = %s")
        values.append(date)

    if filters:
        query += " AND " + " AND ".join(filters)

    if latest_added:
        query += " ORDER BY c.created_at DESC"

    # Grouping to ensure counts work correctly
    query += " GROUP BY c.id"

    # Execute the query with parameterized values
    db.execute(query, tuple(values))
    courses = db.fetchall()

    # Format the result like get_one_course_service
    formatted_courses = [
        {
            "course_id": course[0],
            "title": course[1],
            "industry": course[2],
            "description": course[3],
            "company_id": course[4],
            "created_at": course[5],
            "updated_at": course[6],
            "lesson_count": course[7],
            "assessment_count": course[8],
            "feedback_count": course[9]
        }
        for course in courses
    ]

    return formatted_courses


# CRUD functions for lessons
def create_lesson_service(data):
    query = """
    INSERT INTO lessons (course_id, title, role, topic, industry, convert_type, pdf, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('role'), data.get('topic'),
        data.get('industry'), data.get('convert_type'), data.get('file_name')
    ))
    connection.commit()
    return db.lastrowid


def get_lessons_service(course_id):
    """
    Fetch all lessons associated with a given course_id.
    The result includes formatted created_at and updated_at timestamps.
    """
    # Query to fetch lessons for a specific course
    query = """
    SELECT 
        l.id AS lesson_id,
        l.title,
        l.role,
        l.topic,
        l.industry,
        l.convert_type,
        l.pdf,
        l.created_at,
        l.updated_at
    FROM 
        lessons l
    WHERE 
        l.course_id = %s
    ORDER BY 
        l.created_at ASC
    """

    # Execute the query with the course_id parameter
    db.execute(query, (course_id,))
    lessons = db.fetchall()

    # Format the result
    formatted_lessons = []
    for lesson in lessons:
        formatted_lessons.append({
            "lesson_id": lesson[0],
            "title": lesson[1],
            "role":lesson[2],
            "topic":lesson[3],
            "industry":lesson[4],
            "convert_type":lesson[5],
            "pdf":lesson[6],
            "created_at": lesson[7].strftime("%d %b %Y"),  # Format date as 15 Dec 2001
            "updated_at": lesson[8].strftime("%d %b %Y")   # Format date as 15 Dec 2001
        })

    return formatted_lessons

def update_lesson_service(data):
    """
    Update the specified fields of a lesson by lesson_id.
    """
    # Start building the query
    query = "UPDATE lessons SET "
    updates = []
    values = []

    if "title" in data:
        updates.append("title = %s")
        values.append(data["title"])
    if "role" in data:
        updates.append("role = %s")
        values.append(data["role"])
    if "topic" in data:
        updates.append("topic = %s")
        values.append(data["topic"])
    if "industry" in data:
        updates.append("industry = %s")
        values.append(data["industry"])
    if "convert_type" in data:
        updates.append("convert_type = %s")
        values.append(data["convert_type"])
    if "file_name" in data:
        updates.append("file_name = %s")
        values.append(data["file_name"])

    # Add the updated_at field
    updates.append("updated_at = NOW()")

    # Combine updates into the query
    query += ", ".join(updates)
    query += " WHERE id = %s"
    values.append(data["lesson_id"])

    # Execute the query
    db.execute(query, tuple(values))
    connection.commit()

def delete_lesson_service(lesson_id):
    prev_idx = (get_lesson_PDF(lesson_id))[0]
    print(prev_idx)
    delete_index(prev_idx)
    query = "DELETE FROM lessons WHERE id = %s"
    db.execute(query, (lesson_id,))
    connection.commit()

# CRUD functions for assessments
def create_assessment_service(data):
    query = """
    INSERT INTO assessments (lesson_id, title, objective, number_of_questions, mcq_id , created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('objective'), data.get('number_of_questions'),data.get('mcq_id')
    ))
    connection.commit()
    return db.lastrowid

def get_all_assessment_service(lesson_id):
    """
    Fetch all lessons associated with a given course_id.
    The result includes formatted created_at and updated_at timestamps.
    """
    # Query to fetch lessons for a specific course
    query = """
    SELECT 
        l.id AS assessment_id,
        l.title,
        l.created_at,
        l.updated_at
    FROM 
        lessons l
    WHERE 
        l.course_id = %s
    ORDER BY 
        l.created_at ASC
    """

    # Execute the query with the course_id parameter
    db.execute(query, (lesson_id,))
    lessons = db.fetchall()

    # Format the result
    formatted_lessons = []
    for lesson in lessons:
        formatted_lessons.append({
            "assesment_id": lesson[0],
            "title": lesson[1],
            "created_at": lesson[2].strftime("%d %b %Y"),  # Format date as 15 Dec 2001
            "updated_at": lesson[3].strftime("%d %b %Y")   # Format date as 15 Dec 2001
        })

    return formatted_lessons

def update_assessment_service(assessment_id, data):  
    """
    Update the specified fields of a lesson by lesson_id.
    """
    # Start building the query
    query = "UPDATE assessments SET "
    updates = []
    values = []

    if "title" in data:
        updates.append("title = %s")
        values.append(data["title"])
    if "objective" in data:
        updates.append("objective = %s")
        values.append(data["objective"])
    if "number_of_questions" in data:
        updates.append("number_of_questions = %s")
        values.append(data["number_of_questions"])
    if "MCQ_id" in data:
        updates.append("MCQ_id = %s")
        values.append(data["MCQ_id"])
    
    # Add the updated_at field
    updates.append("updated_at = NOW()")

    # Combine updates into the query
    query += ", ".join(updates)
    query += " WHERE id = %s"
    values.append(assessment_id)

    # Execute the query
    db.execute(query, tuple(values))
    connection.commit()
    query = """
    UPDATE assessments
    SET course_id = %s, title = %s, objective = %s, number_of_questions = %s, updated_at = NOW()
    WHERE id = %s
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('objective'),
        data.get('number_of_questions'), assessment_id
    ))
    db.connection.commit()

def delete_assessment_service(assessment_id):
    prev_MCQ = get_lesson_PDF(assessment_id)[0]
    delete_MCQ(prev_MCQ)
    query = "DELETE FROM lessons WHERE id = %s"
    db.execute(query, (assessment_id,))
    connection.commit()


# CRUD functions for feedbacks
def create_feedback_service(data):
    query = """
    INSERT INTO feedbacks (course_id, feedback_question, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    """
    db.execute(query, (data['course_id'], data.get('feedback_question')))
    connection.commit()
    return db.lastrowid

def get_feedback_service(feedback_id):
    query = "SELECT * FROM feedbacks WHERE id = %s"
    db.execute(query, (feedback_id,))
    feedback = db.fetchone()
    print(feedback)
    feedback={
        "feedback": feedback[2],
        "created_at": feedback[3].strftime("%d %b %Y")
    }

    return feedback

def get_all_feedback_services(course_id):
    """
    Fetch courses filtered by company_id with optional date and latest_added filters.
    """
    # Base query
    query = "SELECT * FROM feedbacks WHERE course_id = %s"
    values = [course_id]  # Initial value for the company_id filter

    # Execute the query with parameterized values
    db.execute(query, tuple(values))
    feedbacks = db.fetchall()

    # Format the results similarly to the get_one_course API
    formatted_feedbacks = []
    for feedback in feedbacks:
        formatted_feedbacks.append({
            "feedback_id":feedback[0],
            "feedback": feedback[2],
            "created_at": feedback[3].strftime("%d %b %Y")
        })

    return formatted_feedbacks

def update_feedback_service(feedback_id, data):
    """
    Update the feedback record dynamically based on provided data.
    Only updates the fields that are present in the data dictionary.
    """
    # Start building the query
    query = "UPDATE feedbacks SET "
    updates = []
    values = []

    # Add fields to be updated dynamically
    if "course_id" in data:
        updates.append("course_id = %s")
        values.append(data["course_id"])
    if "feedback_question" in data:
        updates.append("feedback_question = %s")
        values.append(data["feedback_question"])

    # Ensure at least one field is provided for update
    if not updates:
        raise ValueError("No valid fields provided for update")

    # Add updated_at field to the query
    updates.append("updated_at = NOW()")

    # Complete the query with WHERE clause
    query += ", ".join(updates) + " WHERE id = %s"
    values.append(feedback_id)

    # Execute the query
    db.execute(query, tuple(values))
    connection.commit()

def delete_feedback_service(feedback_id):
    query = "DELETE FROM feedbacks WHERE id = %s"
    db.execute(query, (feedback_id,))
    connection.commit()

def get_lesson_PDF(lesson_id):
    query = "SELECT pdf FROM lessons WHERE id = %s"
    db.execute(query, (lesson_id,))
    return db.fetchone()

def get_MCQ(assessment_id):
    query = "SELECT MCQ_id FROM assessments WHERE id = %s"
    db.execute(query, (assessment_id,))
    return db.fetchone()

def create_MCQ_service(data):
    lesson_id = data['lesson_id']
    title = data['title']
    objective = data['objective']
    no_of_questions = data['no_of_question']

    query = "SELECT pdf FROM lessons WHERE id = %s"
    db.execute(query, (lesson_id,))
    idx = (db.fetchone())[0]
    idx = idx.split('.')[-2]
    print(idx)
    questions , answers = generate_QNA(title , objective , no_of_questions , idx)

    query = """
    INSERT INTO MCQ (question_count, questions, answers, created_at, updated_at)
    VALUES (%s, %s, %s, NOW(), NOW())
    """
    
    db.execute(query, (no_of_questions, json.dumps(questions), answers))
    connection.commit()
  
    return db.lastrowid

def delete_MCQ(MCQ_id):
    query = "DELETE FROM MCQ WHERE id = %s"
    db.execute(query, (MCQ_id,))
    connection.commit()

def get_company_by_user_service(user_id: int):
    query = """
    SELECT company_id 
    FROM company_users 
    WHERE user_id = %s
    """
    db.execute(query, (user_id,))
    result = db.fetchone()
    return result

def get_MCQ_by_assessment_service(assessment_id: int):
    query = """
    SELECT mcq_id 
    FROM assessments 
    WHERE assessment_id = %s
    """
    db.execute(query, (assessment_id,))
    result = db.fetchone()
    return result

def get_course_id_by_name(course_name):
    query = """
    SELECT id 
    FROM courses 
    WHERE LOWER(title) = LOWER(%s)
    LIMIT 1
    """
    db.execute(query, (course_name,))
    result = db.fetchone()
    return result['id'] if result else None

def get_lesson_id_by_name(lesson_name):
    query = """
    SELECT id 
    FROM lessons 
    WHERE LOWER(title) = LOWER(%s)
    LIMIT 1
    """
    db.execute(query, (lesson_name,))
    result = db.fetchone()
    return result['id'] if result else None

def get_assessment_id_by_name(assessment_name):
    query = """
    SELECT id 
    FROM assessments 
    WHERE LOWER(title) = LOWER(%s)
    LIMIT 1
    """
    db.execute(query, (assessment_name,))
    result = db.fetchone()
    return result['id'] if result else None

def get_mcq_question_message(mcq_id):
    # Query to fetch questions and answers
    query = """
    SELECT questions, answers 
    FROM MCQ 
    WHERE id = %s
    """
    db.execute(query, (mcq_id,))
    result = db.fetchone()

    if not result:
        return "No MCQ found with the given ID."

    # Extract data
    questions = result['questions'].split(';')
    answers = eval(result['answers'])  # Ensure correct format if stored as a string

    # Format the MCQ message
    message = "MCQ Questions:\n\n"
    for idx, question in enumerate(questions, 1):
        message += f"{idx}. {question}\n"
        options = answers.get(f"Q{idx}", [])
        for opt_idx, option in enumerate(options, 1):
            message += f"   {chr(64+opt_idx)}. {option}\n"
        message += "\n"

    return message

def get_feedback_questions_service(course_id):
    """
    Fetch feedback questions associated with a specific course ID.
    """
    # Query to fetch feedback questions
    query = """
    SELECT feedback_question 
    FROM feedbacks 
    WHERE course_id = %s
    """
    db.execute(query, (course_id,))
    feedbacks = db.fetchall()

    if not feedbacks:
        return "No feedback questions found for the given course ID."

    # Format the feedback questions
    message = "Feedback Questions:\n\n"
    for idx, feedback in enumerate(feedbacks, 1):
        message += f"{idx}. {feedback['feedback_question']}\n"

    return message


def add_feedback_service(course_id, user_id, feedback):
    """
    Insert a new feedback entry into the database.
    """
    query = """
    INSERT INTO user_feedback (user_id, course_id, feedback, created_at, updated_at)
    VALUES (%s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (user_id, course_id, feedback))
    connection.commit()
    return db.lastrowid

def get_correct_answers_service(assessment_name):
    query = """
    SELECT mq.mcq_id, mq.questions, mq.correct_answers
    FROM assessments a
    JOIN mcqs mq ON a.id = mq.assessment_id
    WHERE a.title = %s
    """
    db.execute(query, (assessment_name,))
    results = db.fetchall()

    # Format answers as a dictionary
    correct_answers = {
        str(result["mcq_id"]): result["correct_answers"]
        for result in results
    }
    return correct_answers
