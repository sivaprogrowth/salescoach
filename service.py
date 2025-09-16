import uuid
from dotenv import load_dotenv
from fastapi import HTTPException
from util import *
import os , json
from datetime import datetime
import mysql.connector
load_dotenv()

DATABASE = os.getenv('DATABASE')
PASSWD = os.getenv('PASSWD')
USER = os.getenv('DB_USER')
HOST = os.getenv('HOST')
print(DATABASE)
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
    # First, fetch course_id from lessons
    course_query = "SELECT course_id FROM lessons WHERE id = %s"
    db.execute(course_query, (data['lesson_id'],))
    result = db.fetchone()   # <-- no arguments
    if not result:
        raise ValueError(f"No lesson found with id {data['lesson_id']}")

    course_id = result['course_id'] if isinstance(result, dict) else result[0]

    query = """
    INSERT INTO assessments (lesson_id, course_id, title, objective, number_of_questions, mcq_id, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (
        data['lesson_id'], course_id, data['title'], data.get('objective'),
        data.get('number_of_questions'), data.get('mcq_id')
    ))
    connection.commit()
    return db.lastrowid

def get_all_assessment_service(course_id):
    """
    Fetch all lessons associated with a given course_id.
    The result includes formatted created_at and updated_at timestamps.
    """
    # Query to fetch lessons for a specific course
    query = """
        SELECT 
            l.id AS lesson_id,
            l.title AS lesson_title,
            a.id AS assessment_id,
            a.title AS assessment_title,
            a.objective,
            a.number_of_questions,
            a.mcq_id,
            a.created_at,
            a.updated_at
        FROM 
            lessons l
        LEFT JOIN assessments a ON l.id = a.lesson_id
        WHERE 
            l.course_id = %s
        ORDER BY 
            l.created_at ASC, a.created_at ASC
    """

    # Execute the query with the course_id parameter
    db.execute(query, (course_id,))
    results = db.fetchall()

    formatted_results = []
    for row in results:
        # Ensure that the assessment exists in the row before processing it
        if row[2]:  # Check if there is an assessment_id (row[2] corresponds to assessment_id)
            # Prepare the assessment details
            assessment = {
                "assessment_id": row[2],
                "title": row[3],
                "objective": row[4],
                "number_of_questions": row[5],
                "mcq_id": row[6],
                "created_at": row[7].strftime("%d %b %Y"),
                "updated_at": row[8].strftime("%d %b %Y"),
                "lesson_id": row[0],  # Add lesson_id for reference
                "lesson_title": row[1]  # Add lesson_title for reference
            }
            # Append assessment to the formatted results list
            formatted_results.append(assessment)

    # Return the list of assessments with lesson information
    return formatted_results

def get_all_assessment_by_lesson_service(lesson_id):
    """
    Fetch all lessons associated with a given course_id.
    The result includes formatted created_at and updated_at timestamps.
    """
    # Query to fetch lessons for a specific course
    query = """
        SELECT 
            *
        FROM 
            assessments a
        WHERE 
            lesson_id = %s
    """

    # Execute the query with the course_id parameter
    db.execute(query, (lesson_id,))
    results = db.fetchall()

    formatted_results = []
    for row in results:
        # Ensure that the assessment exists in the row before processing it
        if row[2]:  # Check if there is an assessment_id (row[2] corresponds to assessment_id)
            # Prepare the assessment details
            assessment = {
                "assessment_id": row[0],
                "lesson_id": row[1],  # Add lesson_id for reference
                "title": row[2],
                "objective": row[3],
                "number_of_questions": row[4],
                "mcq_id": row[5],
                "created_at": row[6].strftime("%d %b %Y"),
                "updated_at": row[7].strftime("%d %b %Y"),
            }
            # Append assessment to the formatted results list
            formatted_results.append(assessment)

    # Return the list of assessments with lesson information
    return formatted_results


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
    
def delete_assessment_service(assessment_id):
    prev_MCQ = get_MCQ_by_assessment_service(assessment_id)[0]
    delete_MCQ(prev_MCQ)
    query = "DELETE FROM assessments WHERE id = %s"
    db.execute(query, (assessment_id,))
    connection.commit()

# CRUD functions for feedbacks
def create_feedback_service(data):
    query = """
    INSERT INTO feedbacks (course_id, feedback_question, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    """
    db.execute(query, (data['course_id'], data.get('feedback_question').strip()))
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

def get_MCQ_service(mcq_id):
    query = "SELECT questions, answers FROM MCQ WHERE mcq_id = %s"
    db.execute(query, (mcq_id,))
    mcq = db.fetchone()
    mcq={
            "questions": mcq[0],
            "answers": mcq[1],
        }

    return mcq

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
  
    return db.lastrowid , questions , answers

def delete_MCQ(MCQ_id):
    query = "DELETE FROM MCQ WHERE mcq_id = %s"
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
    WHERE id = %s
    """
    db.execute(query, (assessment_id,))
    result = db.fetchone()
    return result

def get_course_id_by_name(course_name, company_id):
    query = """
    SELECT id 
    FROM courses 
    WHERE LOWER(title) = LOWER(%s) 
      AND company_id = %s
    LIMIT 1
    """
    db.execute(query, (course_name, company_id))
    result = db.fetchone()
    return result[0] if result else None

def get_lesson_id_by_name(lesson_name,course_id):
    query = """
    SELECT id 
    FROM lessons 
    WHERE LOWER(title) = LOWER(%s)
    AND course_id = %s
    LIMIT 1
    """
    db.execute(query, (lesson_name,course_id))
    result = db.fetchone()
    return result[0] if result else None

def get_index_by_lesson(lesson_name, course_id):
    query = """
    SELECT pdf 
    FROM lessons 
    WHERE LOWER(title) = LOWER(%s)
    AND course_id = %s
    LIMIT 1
    """
    db.execute(query, (lesson_name,course_id))
    result = db.fetchone()
    return result[0] if result else None

def get_assessment_id_by_name(assessment_name, lesson_id):
    query = """
    SELECT id 
    FROM assessments 
    WHERE LOWER(title) = LOWER(%s)
    AND lesson_id = %s
    LIMIT 1
    """
    db.execute(query, (assessment_name,lesson_id))
    result = db.fetchone()
    return result[0] if result else None

def get_mcq_question_message(mcq_id):
    # Query to fetch questions and answers
    query = """
    SELECT questions, answers 
    FROM MCQ 
    WHERE mcq_id = %s
    """
    db.execute(query, (mcq_id[0],))
    result = db.fetchone()

    if not result:
        return "No MCQ found with the given ID."

    # Extract data
    questions = result[0].split(';')
    answers = eval(result[1])  # Ensure correct format if stored as a string
    questions = json.loads(questions[0])

    print("questions ",questions)
    print("answers ",answers)
    # Format the MCQ message
    message = "MCQ Questions:\n\n"
    for idx, question in enumerate(questions):
        print(question)
        message += f"{idx}. {question['question']}\n"
        options = question['options']
        message += options
        message += "\n\n"

    return message , ",".join(answers)

def get_feedback_questions_service(course_id):
    """
    Fetch feedback questions associated with a specific course ID.
    """
    # Query to fetch feedback questions
    query = """
    SELECT feedback_question , id
    FROM feedbacks 
    WHERE course_id = %s
    """
    db.execute(query, (course_id,))
    feedbacks = db.fetchall()

    if not feedbacks:
        return "No feedback questions found for the given course ID."

    # Format the feedback questions
    message = "Feedback Questions:\n"
    for idx, feedback in enumerate(feedbacks, 1):
        message += f"{idx}. {feedback[0]}\n"

    return {"message":message}

def add_feedback_service(course_id,feedback_question,feedback_question_id, user_id, feedback):
    """
    Insert a new feedback entry into the database.
    """
    query = """
    INSERT INTO user_feedback (user_id, course_id, feedback_question_id, feedback_question, feedback, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    db.execute(query, (user_id, course_id, feedback_question_id,feedback_question, feedback))
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

def get_user_count_by_company_service(company_id: int) -> int:
    """
    Fetch the count of users associated with a given company ID.
    """
    query = """
        SELECT COUNT(*) AS user_count
        FROM company_users
        WHERE company_id = %s
    """
    
    # Execute the query
    db.execute(query, (company_id,))
    result = db.fetchone()

    # Return the user count
    return result[0] if result else 0

def get_course_count_by_company_service(company_id: int) -> int:
    """
    Fetch the count of courses associated with a given company ID.
    """
    query = """
        SELECT COUNT(*) AS course_count
        FROM courses
        WHERE company_id = %s
    """
    
    # Execute the query
    db.execute(query, (company_id,))
    result = db.fetchone()

    # Return the course count
    return result[0] if result else 0

def get_assessment_count_by_company_service(company_id: int) -> int:
    """
    Fetch the count of assessments associated with a given company ID.
    """
    query = """
        SELECT COUNT(DISTINCT a.id) AS assessment_count
        FROM courses c
        JOIN lessons l ON c.id = l.course_id
        JOIN assessments a ON l.id = a.lesson_id
        WHERE c.company_id = %s
    """
    
    # Execute the query
    db.execute(query, (company_id,))
    result = db.fetchone()

    # Return the assessment count
    return result[0] if result else 0

def get_user_count_by_city(company_id: int):
    """
    Fetch the user count per city, grouped by monthly and weekly basis for a given company ID.
    
    Args:
    - company_id (int): The ID of the company.
    
    Returns:
    - Dictionary with user counts for monthly and weekly periods.
    """
    # SQL Query for Monthly User Count
    monthly_query = """
        SELECT 
            u.city,
            COUNT(u.user_id) AS user_count,
            EXTRACT(YEAR FROM u.created_at) AS year,
            EXTRACT(MONTH FROM u.created_at) AS month
        FROM 
            users u
        JOIN 
            company_users cu ON u.user_id = cu.user_id
        WHERE 
            cu.company_id = %s
        GROUP BY 
            u.city, EXTRACT(YEAR FROM u.created_at), EXTRACT(MONTH FROM u.created_at)
        ORDER BY 
            year DESC, month DESC;
    """
    
    # SQL Query for Weekly User Count
    weekly_query = """
        SELECT 
            u.city,
            COUNT(u.user_id) AS user_count,
            EXTRACT(YEAR FROM u.created_at) AS year,
            EXTRACT(WEEK FROM u.created_at) AS week
        FROM 
            users u
        JOIN 
            company_users cu ON u.user_id = cu.user_id
        WHERE 
            cu.company_id = %s
        GROUP BY 
            u.city, EXTRACT(YEAR FROM u.created_at), EXTRACT(WEEK FROM u.created_at)
        ORDER BY 
            year DESC, week DESC;
    """

    try:
        # Execute the monthly query
        db.execute(monthly_query, (company_id,))
        monthly_result = db.fetchall()

        # Execute the weekly query
        db.execute(weekly_query, (company_id,))
        weekly_result = db.fetchall()

        # Format the result into a dictionary with two keys: "monthly" and "weekly"
        formatted_result = {
            "monthly": [
                {
                    "city": row[0],
                    "user_count": row[1],
                    "year": row[2],
                    "month": row[3],
                }
                for row in monthly_result
            ],
            "weekly": [
                {
                    "city": row[0],
                    "user_count": row[1],
                    "year": row[2],
                    "week": row[3],
                }
                for row in weekly_result
            ]
        }

        return formatted_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")

def get_feedback_for_company(company_id):
    try:
        # Query to fetch feedback for all courses linked to a company
        query = """
            SELECT 
                u.name as user_name, 
                uf.feedback,
                uf.created_at
            FROM 
                user_feedback uf
            JOIN 
                company_users cu ON cu.user_id = uf.user_id
            JOIN 
                users u ON u.user_id = cu.user_id
            JOIN 
                courses c ON c.id = uf.course_id
            WHERE 
                c.company_id = %s
        """
        values = (company_id,)

        # Execute the query
        db.execute(query, values)
        results = db.fetchall()

        # Format the feedback data
        feedback_list = [
            {
                "user_name": row[0],
                "feedback": row[1],
                "created_at":row[2],
                "profile":"https://app-yogyabano.s3.ap-south-1.amazonaws.com/image.png"
            }
            for row in results
        ]

        return feedback_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback for company: {str(e)}")

def get_dashboard_data_service(company_id):
    try:
        # Get all the necessary data
        userCount = get_user_count_by_company_service(company_id)
        courseCount = get_course_count_by_company_service(company_id)
        assessmentCount = get_assessment_count_by_company_service(company_id)
        trainingCompletionRate = 0  # You can calculate this based on your logic
        userCountByCity = get_user_count_by_city(company_id)
        feedbackInfo = get_feedback_for_company(company_id)

        # Prepare the data as a dictionary
        dashboard_data = {
            "user_count": userCount,
            "course_count": courseCount,
            "assessment_count": assessmentCount,
            "training_completion_rate": trainingCompletionRate,
            "user_count_by_city": userCountByCity,
            "feedback_info": feedbackInfo
        }

        return dashboard_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard data: {str(e)}")

def get_feedback_question_id_by_question(question_text):
    """
    Fetch the feedback_question_id from the feedbacks table by matching question text.
    """
    try:
        query = """
            SELECT id 
            FROM feedbacks 
            WHERE LOWER(feedback_question) = LOWER(%s)
        """
        db.execute(query, (question_text,))
        result = db.fetchone()
        if not result:
            raise ValueError(f"No feedback question found for text: {question_text}")

        return result[0]  # Return the ID
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feedback question ID: {str(e)}")

def get_question_count_voice(user_id , idx):
    check_query = "SELECT COUNT(*) FROM user_qna_progress WHERE user_id = %s AND idx = %s"
    db.execute(check_query, (user_id,idx))
    result = db.fetchone()

    return result

def reset_progress_qna_service(user_id, idx):
    reset_query = """
            UPDATE user_qna_progress 
            SET is_answered = 0, user_response = NULL
            WHERE user_id = %s
            AND idx = %s
            """
    db.execute(reset_query, (user_id,idx))
    connection.commit()

def initialize_progress_qna_service(user_id,idx):
    initialize_query = """
    INSERT INTO user_qna_progress (user_id, qna_id, idx)
    SELECT %s, id , %s
    FROM qna
    WHERE idx = %s
    """
    db.execute(initialize_query, (user_id,idx , idx))
    connection.commit()

def get_next_question_service(user_id , idx):
    query = """
        SELECT q.audio_link , q.question, q.id
        FROM qna q
        JOIN user_qna_progress uqp ON q.id = uqp.qna_id
        WHERE uqp.user_id = %s AND uqp.idx = %s AND uqp.is_answered = 0
        ORDER BY qna_id ASC
        LIMIT 1
    """
    db.execute(query, (user_id,idx))
    next_question = db.fetchone()
    if not next_question:
        return {"message": "No unanswered questions found."}
    audio_url = next_question[0]
    question = next_question[1]   
    qna_id = next_question[2]
    return {'audio_url': audio_url, 'message':question, "qna_id":qna_id}

def submit_answer_service(user_response,user_id,qna_id):
    try:
        download_audio(user_response,"reply2.wav", "wav")
        user_response = transcribe()
        query = """
            UPDATE user_qna_progress
            SET is_answered = 1, user_response = %s
            WHERE user_id = %s AND qna_id = %s
        """
        db.execute(query, (user_response, user_id, qna_id))
        connection.commit()
        if db.rowcount == 0:
            raise HTTPException(status_code=404, detail="Progress entry not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_lesson_content_type_service(lesson_id: int):
    """
    Retrieve the content type for a specific lesson by its ID.
    """
    try:
        query = "SELECT convert_type FROM lessons WHERE id = %s"
        db.execute(query, (lesson_id,))
        result = db.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Lesson not found")

        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_cv_feedback(user_id):
    query = """
        SELECT *
        FROM user_cv_data
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
    # Ensure any previous result is consumed
    while db.nextset():  # Use nextset to process all pending results
        pass
    db.execute(query, (user_id,))
    result = db.fetchone()

    aspect_text_mapping = {
            "Clarity": result[2],  
            "Professional Summary": result[4],
            "Skills and Work Experience": "\n".join([
                item for item in [result[6], result[8], result[9]] if item
            ]),
            "Formatting of the Document": result[2],  
            "Minor Points": result[2] 
        }


    # Define aspect-specific prompts
    aspect_prompts = {
        "Clarity": "Analyze the clarity of the following CV text. Identify any ambiguous or unclear sections and suggest improvements. Give the output in 50 words only",
        "Professional Summary": "Evaluate the professional summary of this CV. Is it concise, impactful, and relevant? Suggest improvements if necessary.Give the output in 50 words only",
        "Skills and Work Experience": "Analyze the skills and work experience sections of this CV. Identify missing details, redundancies, or areas for enhancement.Give the output in 50 words only",
        "Formatting of the Document": "Review the formatting of this CV. Is it structured well, visually appealing, and professional? Suggest improvements.Give the output in 50 words only",
        "Minor Points": "Check for minor issues in this CV, such as typos, inconsistencies, missing details, or unnecessary information.Give the output in 50 words only"
    }

    feedback = {}
    for aspect, text in aspect_text_mapping.items():
        print(f"Analyzing aspect: {aspect}")
        print("Text: ", text)
        prompt = aspect_prompts.get(aspect, "Analyze the following CV section and provide feedback.")
        # Convert aspect to lowercase and replace spaces with underscores
        aspect_key = aspect.lower().replace(' ', '_')
        print(aspect_key)
        feedback[aspect_key] = analyze_cv_with_gpt(text, prompt)  # Pass both text and prompt

    # # Formatting the consolidated feedback
    # consolidated_feedback = "\n\n".join([f"**{key}**:\n{value}" for key, value in feedback.items()])
    
    return feedback

def upload_cv_to_db_service(pdf_url, user_id, id):

    pdf_stream = download_pdf_from_url(pdf_url)
    cv_text = extract_text_from_pdf(pdf_stream)
    sections = classify_sections_gpt(cv_text)

    query = """
        INSERT INTO user_cv_data (id, user_id, cv_text, header, summary, education, experience, 
                                  skills, certificate, projects, additional_info)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    # Replace None or empty fields with empty strings
    user_id = user_id if user_id else ""
    cv_text = cv_text if cv_text else ""
    header = sections.get("Header", "")
    summary = sections.get("Summary", "")
    education = sections.get("Education", "")
    experience = sections.get("Experience", "")
    skills = sections.get("Skills", "")
    certificate = sections.get("Certifications", "")
    projects = sections.get("Projects", "")
    additional_info = sections.get("Additional_Info", "")
    # Ensure any previous result is consumed
    while db.nextset():  # Use nextset to process all pending results
        pass
    db.execute(query, (id, user_id, cv_text, header, summary, education, experience, skills, certificate, projects, additional_info))
    connection.commit()
    print("CV data uploaded successfully")

def get_cover_letter_service(jd , user_id):
    query = """
        SELECT cv_text
        FROM user_cv_data
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
    db.execute(query, (user_id,))
    cv = db.fetchone()[0]

    if not cv:
        raise HTTPException(status_code=404, detail="Data not found")
    
    return cover_letter_with_gpt(cv, jd)

def get_job_title_service(user_id):
    query = """
        SELECT cv_text
        FROM user_cv_data
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
    db.execute(query, (user_id,))
    cv = db.fetchone()[0]
    if not cv:
        raise HTTPException(status_code=404, detail="Data not found")
    result = job_title_with_gpt(cv)
    print(result)
    return result

def add_job_preferrance_service(data):
    try:
        print(data)
        preferred_location = data.get('preferred_location', "")
        preferred_job_title = data.get('preferred_job_title', "")
        expected_salary = data.get('expected_salary', "")
        cv_id = data.get('cv_id', "")
        name = data.get('name', "")
        phone_number = data.get('phone_number', "")

        fields_str = f"preferred_location = {preferred_location},\npreferred_job_title = {preferred_job_title},\nexpected_salary = {expected_salary}"
        preferrance  = classify_job_preferences_gpt(fields_str)
        print(preferrance)
        
        # Convert the preferences to JSON format
        preferred_locations_json = json.dumps(preferrance.get('Preferred_Location', []))
        preferred_job_titles_json = json.dumps(preferrance.get('Preferred_Job_Title', []))
        expected_salary = preferrance.get('Expected_Salary', "")

        # Check if the phone number already exists
        check_query = "SELECT COUNT(*) FROM users_job_assistance WHERE phone_number = %s"
        db.execute(check_query, (phone_number,))
        count = db.fetchone()[0]

        if count > 0:
            # Update the existing record
            update_query = """
            UPDATE users_job_assistance
            SET name = %s, preferred_locations = %s, preferred_job_titles = %s, expected_salary = %s, cv_id = %s
            WHERE phone_number = %s
            """
            values = (
                name,
                preferred_locations_json,
                preferred_job_titles_json,
                expected_salary,
                cv_id,
                phone_number
            )
            db.execute(update_query, values)
        else:
            # Insert a new record
            insert_query = """
            INSERT INTO users_job_assistance (user_id, phone_number, name, preferred_locations, preferred_job_titles, expected_salary, cv_id, credits)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 70)
            """
            values = (
                uuid.uuid4().hex,
                phone_number,
                name,
                preferred_locations_json,
                preferred_job_titles_json,
                expected_salary,
                cv_id
            )
            db.execute(insert_query, values)

        connection.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
def recommend_jobs_service(phone_number, count):
# Get user preferences
    db.execute("SELECT expected_salary, preferred_locations, preferred_job_titles FROM users_job_assistance WHERE phone_number = %s", (phone_number,))
    user_pref = db.fetchone()
    
    if not user_pref:
        raise HTTPException(status_code=404, detail="User preferences not found")
    
    query = """
        WITH user_pref AS (
            SELECT expected_salary AS salary, preferred_locations AS location, preferred_job_titles AS role 
            FROM users_job_assistance 
            WHERE phone_number = %s
        ) 
        SELECT * FROM jobs 
        WHERE 
            (location = (SELECT location FROM user_pref) 
            OR (SELECT COUNT(*) FROM jobs WHERE location = (SELECT location FROM user_pref)) = 0) 
            AND 
            (role = (SELECT role FROM user_pref) 
            OR (SELECT COUNT(*) FROM jobs WHERE role = (SELECT role FROM user_pref)) = 0) 
            AND 
            salary >= (SELECT salary FROM user_pref)
        ORDER BY 
            CASE 
                WHEN location = (SELECT location FROM user_pref) 
                AND role = (SELECT role FROM user_pref) THEN 1
                WHEN location = (SELECT location FROM user_pref) THEN 2
                WHEN role = (SELECT role FROM user_pref) THEN 3
                ELSE 4
            END,
            salary DESC 
        LIMIT %s;
    """

    db.execute(query, (phone_number,count))
    jobs = db.fetchall()
    formatted_jobs = ""
    for job in jobs:
        url = f"https://app.yogyabano.com/jobs/{job[0]}_{phone_number}"
        formatted_jobs += f"title: {job[1]}\n"
        formatted_jobs += f"description: {job[4]}\n"
        formatted_jobs += f"salary: {job[2]}\n"
        formatted_jobs += f"location: {job[3]}\n"
        formatted_jobs += f"url: {url}\n\n"

    return formatted_jobs

def refine_job_feedback(user_id, job_feedback):
    try:
        # Query to extract user preferences
        query = """
        SELECT preferred_locations, preferred_job_titles, expected_salary 
        FROM users_job_assistance 
        WHERE phone_number = %s
        """
        db.execute(query, (user_id,))
        user_pref = db.fetchone()
        if not user_pref:
            raise HTTPException(status_code=404, detail="User preferences not found")

        preferred_location = user_pref[0]
        preferred_job_title = user_pref[1]
        expected_salary = user_pref[2]
        fields_str = f"preferred_location = {preferred_location},\npreferred_job_title = {preferred_job_title},\nexpected_salary = {expected_salary},\nfeedback = {job_feedback}"
        refined_pref = classify_job_preferences_gpt(fields_str)

        # Update the user preferences with refined classifications
        query = """
        UPDATE users_job_assistance 
        SET preferred_locations = %s, preferred_job_titles = %s, expected_salary = %s 
        WHERE phone_number = %s
        """
        values = (
            refined_pref.get('preferred_location', preferred_location),
            refined_pref.get('preferred_job_title', preferred_job_title),
            refined_pref.get('expected_salary', expected_salary),
            user_id
        )
        db.execute(query, values)
        connection.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
def get_user_credits(user_id):
    try:
        query = "SELECT credits FROM users_job_assistance WHERE phone_number = %s"
        db.execute(query, (user_id,))
        result = db.fetchone()
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user credits: {str(e)}")

def check_user_credits(user_id):
    credits = get_user_credits(user_id)
    if (credits / 10) >= 1:
        return {"permitted": "Yes", "credits": credits, "number_of_jobs": (credits / 10)}
    else:
        return {"permitted": "No", "credits": credits, "number_of_jobs": 0}
    
def update_user_credits(user_id, credits):
    query = "UPDATE users_job_assistance SET credits = %s WHERE phone_number = %s"
    db.execute(query, (credits, user_id))
    connection.commit()
    
def job_assistance_gpt_service(job_preferrance,query_str):
    try:
        # Generate job assistance response using GPT
        response = job_assistance_gpt(job_preferrance, query_str)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job assistance: {str(e)}")
    
def get_job_url_by_id_service(job_id):

    query = "SELECT apply_url FROM jobs WHERE id = %s"
    db.execute(query, (job_id,))
    result = db.fetchone()[0]
    return result

def job_activity_log_service(job_id, phone_number):
    try:
        # Insert a new job activity log entry
        query = "INSERT INTO job_activity_log (job_id, phone_number, activity_type) VALUES (%s, %s, %s)"
        values = (job_id, phone_number, "Viewed")
        db.execute(query, values)
        connection.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting job activity log: {str(e)}")
