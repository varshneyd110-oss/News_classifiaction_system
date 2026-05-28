import sqlite3



# =========================
# CONNECTION
# =========================

def get_connection():

    conn = sqlite3.connect(
        "users.db",
        check_same_thread=False
    )

    return conn



# =========================
# CREATE TABLE
# =========================

def create_login_table():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE,

        password TEXT,

        role TEXT
    )
    """)

    conn.commit()

    conn.close()



# =========================
# INSERT ADMIN
# =========================

def insert_login_data():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO users
    (username,password,role)

    VALUES
    ('admin','1234','admin')
    """)

    conn.commit()

    conn.close()



# =========================
# CREATE USER
# =========================

def create_user(username,password):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO users
        (username,password,role)

        VALUES
        (?,?,?)
        """,(username,password,"user"))

        conn.commit()

        conn.close()

        return True

    except Exception as e:

        print(e)

        return False



# =========================
# LOGIN USER
# =========================

def login_user(username,password):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM users

    WHERE username=? AND password=?
    """,(username,password))

    data = cursor.fetchone()

    conn.close()

    return data

# 📊 GET USER LOGIN DATA

def get_user_login_data():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT username, role, login_count
    FROM users
    """)

    data = cursor.fetchall()

    conn.close()

    return data

# 🔥 UPDATE LOGIN COUNT

def update_login_count(username):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    UPDATE users
    SET login_count = login_count + 1
    WHERE username=?
    """, (username,))

    conn.commit()

    conn.close()

# 👥 TOTAL USERS

def get_total_users():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*) FROM users
    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total

# 📦 Create Table
def create_news_table():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        News TEXT,
        Prediction TEXT,
        Confidence REAL
    )
    """)

    conn.commit()
    conn.close()


# 💾 Save Prediction
def save_news_prediction(News, Prediction, Confidence):
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO news_predictions (News, Prediction, Confidence) VALUES (?, ?, ?)",
        (News, Prediction, Confidence)
    )

    conn.commit()
    conn.close()


# 📊 Get All Data
def get_all_news_predictions():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM news_predictions ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data


# 🗑️ Clear History
def clear_news_history():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM news_predictions")

    conn.commit()
    conn.close()



