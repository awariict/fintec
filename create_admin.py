from werkzeug.security import generate_password_hash
import mysql.connector
from datetime import datetime

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="fintech_db"
)

cur = conn.cursor()

# delete old admin
cur.execute("DELETE FROM users WHERE username='admin'")

# create new secure admin
cur.execute("""
INSERT INTO users(username,password,role,status,created_at)
VALUES(%s,%s,%s,%s,%s)
""", (
    "admin",
    generate_password_hash("admin123"),
    "admin",
    "approved",
    datetime.now()
))

conn.commit()
conn.close()

print("✅ Admin recreated successfully")