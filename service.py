from dotenv import load_dotenv
from util import *
import os , json
import mysql.connector
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
connection = mysql.connector.connect(
    user = 'admin',
    host = 'salescoachdatabase-1.ctkqiaw8wxwq.ap-south-1.rds.amazonaws.com',
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
    # Base query
    query = "SELECT * FROM courses WHERE company_id = %s"
    filters = []
    values = [company_id]  # Initial value for the company_id filter

    # Apply date filter if provided
    if date:
        filters.append("created_at::date = %s")
        values.append(date)

    # Add ordering for the latest added filter
    if latest_added:
        query += " ORDER BY created_at DESC"

    # Append additional filters to the WHERE clause
    if filters:
        query += " AND " + " AND ".join(filters)

    # Execute the query with parameterized values
    db.execute(query, tuple(values))
    return db.fetchall()


# CRUD functions for lessons
def create_lesson(data):
    query = """
    INSERT INTO lessons (course_id, title, role, topic, industry, convert_type, pdf, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('role'), data.get('topic'),
        data.get('industry'), data.get('convert_type'), data.get('pdf')
    ))
    db.connection.commit()
    return db.lastrowid

def get_lesson(lesson_id):
    query = "SELECT * FROM lessons WHERE id = %s"
    db.execute(query, (lesson_id,))
    return db.fetchone()

def update_lesson(lesson_id, data, db):
    query = """
    UPDATE lessons
    SET course_id = %s, title = %s, role = %s, topic = %s, industry = %s, convert_type = %s, pdf = %s, updated_at = NOW()
    WHERE id = %s
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('role'), data.get('topic'),
        data.get('industry'), data.get('convert_type'), data.get('pdf'), lesson_id
    ))
    db.connection.commit()

def delete_lesson(lesson_id):
    query = "DELETE FROM lessons WHERE id = %s"
    db.execute(query, (lesson_id,))
    db.connection.commit()

# CRUD functions for assessments
def create_assessment(data):
    query = """
    INSERT INTO assessments (course_id, title, objective, number_of_questions, created_at, updated_at)
    VALUES (%s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (
        data['course_id'], data['title'], data.get('objective'), data.get('number_of_questions')
    ))
    db.connection.commit()
    return db.lastrowid

def get_assessment(assessment_id):
    query = "SELECT * FROM assessments WHERE id = %s"
    db.execute(query, (assessment_id,))
    return db.fetchone()

def update_assessment(assessment_id, data):
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

def delete_assessment(assessment_id):
    query = "DELETE FROM assessments WHERE id = %s"
    db.execute(query, (assessment_id,))
    db.connection.commit()

# CRUD functions for feedbacks
def create_feedback(data):
    query = """
    INSERT INTO feedbacks (course_id, feedback_question, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    """
    db.execute(query, (data['course_id'], data.get('feedback_question')))
    db.connection.commit()
    return db.lastrowid

def get_feedback(feedback_id):
    query = "SELECT * FROM feedbacks WHERE id = %s"
    db.execute(query, (feedback_id,))
    return db.fetchone()

def update_feedback(feedback_id, data):
    query = """
    UPDATE feedbacks
    SET course_id = %s, feedback_question = %s, updated_at = NOW()
    WHERE id = %s
    """
    db.execute(query, (data['course_id'], data.get('feedback_question'), feedback_id))
    db.connection.commit()

def delete_feedback(feedback_id):
    query = "DELETE FROM feedbacks WHERE id = %s"
    db.execute(query, (feedback_id,))
    db.connection.commit()
