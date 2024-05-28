import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="text_mining_db"
    )

def save_document(title, content):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO documents (title, content) VALUES (%s, %s)", (title, content))
    connection.commit()
    connection.close()

def get_documents():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM documents")
    documents = cursor.fetchall()
    connection.close()
    return documents

def delete_document(doc_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    connection.commit()
    connection.close()
