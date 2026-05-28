import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from database import create_login_table,insert_login_data,login_user,create_user,create_news_table,save_news_prediction,get_all_news_predictions,clear_news_history
import sqlite3
from database import update_login_count,get_user_login_data,get_total_users


# ✅ Page config (sirf yahi file me hona chahiye)
st.set_page_config(layout="wide")

create_login_table()
create_news_table()
insert_login_data()

tk=joblib.load("tokenizer.pkl")
model=load_model("news_model.keras")
encoder=joblib.load("encoder.pkl")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = ""

if "redirect_login" not in st.session_state:
    st.session_state.redirect_login = False

if "page" not in st.session_state:
        st.session_state.page = "home"

def cleaning(doc):
    doc= str(doc).lower()
    doc = re.sub(r"\s+", " ", doc)
    return doc.strip()

label_map = {

    1: "🌍 World",

    2: "🏅 Sports",

    3: "💼 Business",

    4: "💻 Sci/Tech"
}



def predict_news(text):

    # EMPTY CHECK
    
    if text.strip() == "":

        return "⚠️ No Input", 0

    # CLEAN TEXT
    
    text = cleaning(text)

    # TEXT TO SEQUENCE
   
    seq = tk.texts_to_sequences([text])

    # PADDING

    padded = pad_sequences(
        seq,
        maxlen=70,
        padding="post",
        truncating="post"
    )


    # PREDICTION

    pred = model.predict(
        padded,
        verbose=0
    )

    # CLASS INDEX

    class_index = np.argmax(pred)


    # ORIGINAL LABEL

    label = encoder.inverse_transform(
        [class_index]
    )[0]

    # CATEGORY NAME

    category = label_map[label]


    # CONFIDENCE

    confidence = float(np.max(pred)) * 100

    return category, confidence



def login_news_page():


        
    import streamlit.components.v1 as components
    import base64

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_base64 = get_base64_image("background_1.png")

    components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>

        .banner {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            padding: 70px 20px;
            border-radius: 20px;
            text-align: center;
            color: white;
            position: relative;
        }}

        .banner::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.6);
            border-radius: 20px;
        }}

        .banner-content {{
            position: relative;
            z-index: 2;
        }}

        .title {{
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #ffcc70, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: 0.3s;
    }}

    .title:hover {{
        transform: scale(1.05);
        background: linear-gradient(90deg, #00ffcc, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

        .subtitle {{
            font-size: 18px;
            margin-top: 10px;
        }}

        .rotator {{
            height: 180px;
            overflow: hidden;
        }}

        .rotator-inner {{
            display: flex;
            flex-direction: row;
            width: 250%;
            animation: scrollLeft 8s linear infinite;
        }}

        .slide {{
            min-width: 50%;
            flex-shrink: 0;
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        @keyframes scrollLeft {{
        0% {{
            transform: translateX(0);
        }}
        100% {{
            transform: translateX(-50%);
        }}
    }}

        </style>
        </head>

        <body>

        <div class="banner">
            <div class="banner-content">

                <div class="rotator-inner">

                            <!-- ORIGINAL -->
                            <div class="slide">
                                <div class="title">📰 News Classifier AI 🤖</div>
                                <div class="subtitle">Smartly categorizing news into World News, Sports News, Business News & more using AI 🚀</div>
                                <div id="clock"></div>
                            </div>

                            <div class="slide">
                                <div class="title">📰 News Classifier AI 🤖</div>
                                <div class="subtitle">Transforming headlines into meaningful insights with Machine Learning 📊</div>
                                <div id="clock2"></div>
                            </div>

                            <!-- DUPLICATE (same content again) -->
                            <div class="slide">
                                <div class="title">📰 News Classifier AI 🤖</div>
                                <div class="subtitle">Smartly categorizing news into World News, Sports News, Business News & more using AI 🚀</div>
                                <div id="clock3"></div>
                            </div>

                            <div class="slide">
                                <div class="title">📰 News Classifier AI 🤖</div>
                                <div class="subtitle">Transforming headlines into meaningful insights with Machine Learning 📊</div>
                                <div id="clock4"></div>
                            </div>

                        </div>

            </div>
        </div>

        <script>
        function updateClock() {{
            var now = new Date();

            var options = {{
                weekday: 'long',
                day: '2-digit',
                month: 'short',
                year: 'numeric'
            }};

            var date = now.toLocaleDateString('en-IN', options);
            var time = now.toLocaleTimeString();

            var html = `
            <div style="
                font-size:30px;
                font-weight:bold;
                color:#00ffcc;
                text-shadow:0 0 10px #00ffcc, 0 0 20px #00ffcc;
                text-align:center;
            ">
            ⏰ ${{date}} ${{time}}
            </div>
            `;

            document.getElementById("clock").innerHTML = html;
            document.getElementById("clock2").innerHTML = html;
            document.getElementById("clock3").innerHTML = html;
            document.getElementById("clock4").innerHTML = html;
        }}

        setInterval(updateClock, 1000);
        updateClock();
        </script>

        </body>
        </html>
        """, height=300)
    
    st.sidebar.markdown("""
        <style>
        .sidebar-banner {
            padding: 12px 15px;
            border-radius: 12px;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            transition: all 0.4s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }

        .sidebar-banner:hover {
            color: black;
            transform: scale(1.05);
        }
        </style>

        <div class="sidebar-banner">
            📰 News Control Panel
        </div>
        """, unsafe_allow_html=True)
    
    

    menu = ["🔐 Login", "🆕 Signup"]

    if "menu_choice" not in st.session_state:
        st.session_state.menu_choice = "🔐 Login"

    choice = st.sidebar.selectbox(
        "Menu",
        menu,
        index=0 if st.session_state.get("redirect_login", False) else 1
    )

    # 🖼️ Image
    st.sidebar.image("news_1.png")

        # 🔹 Welcome
    st.sidebar.markdown(
        "<h2 style='color:#00FFFF;'>👋 Welcome</h2>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        "<p style='color:#FFD700;'>Explore AI-powered News Classification 📰</p>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 About App
    st.sidebar.markdown(
        "<h3 style='color:#FF4B4B;'>🤖 About App</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        "<p style='color:#E0E0E0;'>Classifies news headlines into categories like "
        "<b style='color:orange;'>World News</b>, <b style='color:green;'>Sports News</b>, "
        "<b style='color:#FF69B4;'>Business News</b>, <b style='color:#FFD700;'>Science/Technology</b> using Machine Learning & NLP.</p>",
    unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Features
    st.sidebar.markdown(
        "<h3 style='color:#00FFAA;'>🚀 Features</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        ✔ <span style='color:#FF6B6B;'>Automatic News Categorization</span><br>
        ✔ <span style='color:#4D96FF;'>Real-time Prediction</span><br>
        ✔ <span style='color:#FFD93D;'>AI-powered Text Analysis</span><br>
        ✔ <span style='color:#6BCB77;'>Fast & Accurate Classification</span><br>
        ✔ <span style='color:#00FFFF;'>User-friendly Interface</span>
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Categories
    st.sidebar.markdown(
        "<h3 style='color:#FFA500;'>📊 Categories</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        🏛️ <span style='color:#FFD93D;'>World News</span><br>
        ⚽ <span style='color:#4D96FF;'>Sports News</span><br>
        🎬 <span style='color:#FF6B6B;'>Business News</span><br>
        💰 <span style='color:#6BCB77;'>Science/Technology</span><br>
        
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 How to Use
    st.sidebar.markdown(
        "<h3 style='color:#00BFFF;'>📱 How to Use?</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        1️⃣ Enter News Headline<br>
        2️⃣ Click <b style='color:#FFD700;'>Classify</b><br>
        3️⃣ Get Category Prediction with Probability
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Model Info (NEW 🔥)
    st.sidebar.markdown(
        "<h3 style='color:#FFA500;'>📊 Model Info</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        🤖 <span style='color:#FFD700;'>Model:</span>Embeddings<br>
        📈 <span style='color:#00FFAA;'>High Accuracy Text Classification</span><br>
        ⚡ <span style='color:#4D96FF;'>Fast Prediction Speed</span><br>
        🔍 <span style='color:#FF6B6B;'>Text Preprocessing (Cleaning + Stopwords Removal)</span>
        🔍 <span style='color:#FF6B6B;'>LSTM</span>
        🤖 <span style='color:#FFD700;'>Model:Sequential (DL Model)</span>Sequential<br>
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Use Cases (NEW 🔥)
    st.sidebar.markdown(
        "<h3 style='color:#9D4EDD;'>🚀 Use Cases</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        ✔ News Recommendation Systems<br>
        ✔ Personalized News Feed<br>
        ✔ Content Filtering<br>
        ✔ Fake News Detection (Future Scope)
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Did You Know (updated)
    st.sidebar.markdown(
        "<h3 style='color:#FF69B4;'>💡 Did You Know?</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        📰 <span style='color:#FFD700;'>Millions of news articles published daily</span><br>
        🤖 <span style='color:#00FFAA;'>AI helps organize massive data</span><br>
        ⚡ <span style='color:#4D96FF;'>Used in Google News & media apps</span>
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Developer
    st.sidebar.markdown(
        "<h3 style='color:#9D4EDD;'>🧑‍💻 Developer</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        "<p style='color:white;'>Made by: <b style='color:#00FFFF;'>Dev Varshney</b><br>AI/ML Enthusiast 🚀</p>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🔹 Contact
    st.sidebar.markdown(
        "<h3 style='color:#FF4B4B;'>📞 Contact</h3>",
        unsafe_allow_html=True
        )
    st.sidebar.markdown(
        """
        📱 <span style='color:#FFD700;'>9058068999</span><br>
        📧 <a href='mailto:varshneyd110@gmail.com' style='color:#00FFFF;'>Email Me</a>
        """,
        unsafe_allow_html=True
        )
    st.sidebar.markdown("---")


        # 🏁 Footer Image
    st.sidebar.image("flag.jpg")



    if choice == "🔐 Login":
            
            st.markdown("""
                <div style="
                    background: rgba(255,255,255,0.08);
                    padding: 14px;
                    border-radius: 12px;
                    border-left: 5px solid #ffcc70;
                    color: #f1f1f1;
                    font-size: 15px;
                    margin-bottom: 20px;
                ">

                ℹ️ If you do not have an account yet, please go to the 
                <b>🆕 Signup</b> page using the sidebar menu and create a new account using your preferred username and password.

                </div>
                """, unsafe_allow_html=True)
        

            # 🔐 Login Box
            st.markdown("## 🔐 Login to Continue")

            st.write("\n")
            st.write("### 👤 Enter Username")
            username = st.text_input("",placeholder="please type User Name ......")
            st.write("### 🔑 Enter Password")
            password = st.text_input("",type="password",placeholder="please type Password ......")
            col1,col2,col3=st.columns(3)

            with col2:

                if st.button("Login"):

                    result = login_user(
                        username,
                        password
                    )

                    if result:

                        update_login_count(username)

                        st.success(f"Welcome {username}")

                        role = result[3]

                        if role == "admin":

                                st.write("👑 Admin Panel")
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.role = result[3]
                                st.rerun()
                                                        
                        else:

                                st.write("👤 User Dashboard")
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.role = result[3]
                                st.rerun()

                    else:

                                st.error("Invalid Credentials")

                
    elif choice == "🆕 Signup":
            
            st.session_state.redirect_login = False

            st.markdown("""
                <div style="
                    background: rgba(255,255,255,0.08);
                    padding: 14px;
                    border-radius: 12px;
                    border-left: 5px solid #00c6ff;
                    color: #f1f1f1;
                    font-size: 15px;
                    margin-bottom: 20px;
                ">

                ℹ️ If you already have an account, please go to the 
                <b>🔐 Login</b> page using the sidebar menu and sign in with your registered username and password.

                </div>
                """, unsafe_allow_html=True)

            # 🔐 Login Box
            st.markdown("## 🆕 Create New Account")

            st.write("\n")
            st.write("### 👤 Create New Username")



            # SESSION STATE DEFAULTS
            if "signup_user" not in st.session_state:
                st.session_state.signup_user = ""

            if "signup_pass" not in st.session_state:
                st.session_state.signup_pass = ""

            new_user = st.text_input("",placeholder="please type Your New User Name ......",key="signup_user")

            st.write("### 🔑 Create New Password")

            new_password = st.text_input(
                "",
                type="password",
                placeholder="please type Your New Password ......",
                key="signup_pass"

            )

            col1,col2,col3=st.columns(3)

            with col2:

                if st.button("Signup"):

                    success = create_user(
                        new_user,
                        new_password
                    )

                    if success:

                        st.success("Account Created Successfully")
                        # REDIRECT FLAG
                        st.session_state.redirect_login = True

                        st.rerun()        
                    else:

                        st.error("Username Already Exists")


    st.markdown("""
    <style>

    /* 🎯 All Buttons Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #ff8008, #ffc837);
        color: white;
        font-size: 40px;
        font-weight: bold;
        padding: 18px 50px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(0, 114, 255, 0.4);
        width: 100%;
        margin-top: 20px;
    }

    /* 🔥 Hover Effect */
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        box-shadow: 0 10px 25px rgba(0, 114, 255, 0.6);
        font-size: 50px;
        font-weight: bold;
        padding: 20px 55px;
    }

    /* Click Effect */
    div.stButton > button:active {
        transform: scale(0.98);
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>

        /* Input box styling */
        .stTextInput > div > div > input {
            background: linear-gradient(135deg, #1f1f2e, #2c2c54);
            color: #ffffff;
            font-size: 20px;
            padding: 12px;
            border-radius: 12px;
            border: 2px solid #00c6ff;
            outline: none;
            transition: all 0.4s ease;
        }

        /* Placeholder color */
        .stTextInput > div > div > input::placeholder {
            color: #bbbbbb;
            font-size: 16px;
        }

        /* Hover effect */
        .stTextInput > div > div > input:hover {
            border: 2px solid #00f2fe;
            box-shadow: 0 0 10px #00c6ff;
        }

        /* Focus (click) effect 🔥 */
        .stTextInput > div > div > input:focus {
            border: 2px solid #00f2fe;
            box-shadow: 0 0 15px #00f2fe, 0 0 25px #00c6ff;
            background: linear-gradient(135deg, #141e30, #243b55);
        }

        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        @keyframes fadeText {
            0% {opacity: 0;}
            5% {opacity: 1;}
            12% {opacity: 1;}
            17% {opacity: 0;}
            100% {opacity: 0;}
        }

        .text-container {
            position: relative;
            height: 40px;
            text-align: left;
            color: #00c6ff;
            font-size: 22px;
            font-weight: bold;
            margin-top: 40px;
            overflow: hidden; /* 🔥 important */
        }

        .text-container span {
            position: absolute;
            width: 100%;
            opacity: 0;
            animation: fadeText 16s linear infinite; /* 🔥 total = 8 × 2s */
        }

        /* Proper spacing */
        .text-container span:nth-child(1) { animation-delay: 0s; }
        .text-container span:nth-child(2) { animation-delay: 2s; }
        .text-container span:nth-child(3) { animation-delay: 4s; }
        .text-container span:nth-child(4) { animation-delay: 6s; }
        .text-container span:nth-child(5) { animation-delay: 8s; }
        .text-container span:nth-child(6) { animation-delay: 10s; }
        .text-container span:nth-child(7) { animation-delay: 12s; }
        .text-container span:nth-child(8) { animation-delay: 14s; }
        </style>

        <div class="text-container">
            <span>📰 Classify News Headlines in Seconds</span>
            <span>🤖 AI-Powered News Categorization</span>
            <span>📊 Get Smart Insights from News Data</span>
            <span>⚡ Real-Time News Classification</span>
            <span>🔍 Identify News Categories Instantly</span>
            <span>🧠 Powered by Machine Learning & NLP</span>
            <span>📂 Organize News into Smart Categories</span>
            <span>🚀 Fast & Accurate News Predictions</span>
        </div>
        """, unsafe_allow_html=True)

        
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1f4037, #99f2c8);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>📰 Total News Classified</h4>
                <h2>120k+</h2>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #141e30, #243b55);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>🎯 Model Accuracy</h4>
                <h2>95%</h2>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #42275a, #734b6d);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>📊 Categories Covered</h4>
                <h2>4 Types</h2>
            </div>
            """, unsafe_allow_html=True)


    
    st.markdown("""
    <style>

    /* Sidebar background */
    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #0f172a,
            #1e293b
        );

        border-right: 2px solid #334155;
    }



    /* Sidebar title */
    .sidebar-title {

        color: white;

        font-size: 28px;

        font-weight: bold;

        text-align: center;

        margin-bottom: 20px;
    }



    /* Selectbox label */
    [data-testid="stSidebar"] label {

        color: #f8fafc !important;

        font-size: 18px !important;

        font-weight: bold;
    }



    /* Selectbox */
    [data-baseweb="select"] {

        background-color: #1e293b !important;

        border-radius: 12px !important;

        border: 2px solid #38bdf8 !important;
    }



    /* Hover effect */
    [data-baseweb="select"]:hover {

        border: 2px solid #00ffcc !important;

        box-shadow: 0 0 10px #00ffcc;
    }

    </style>
    """, unsafe_allow_html=True)

    

def main_app():
    st.sidebar.markdown("""
        <style>
        .sidebar-banner {
            padding: 12px 15px;
            border-radius: 12px;
            background: linear-gradient(135deg, #007BFF, #00C6FF);
            color: white;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            transition: all 0.4s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }

        .sidebar-banner:hover {
            color: black;
            transform: scale(1.05);
        }
        </style>

        <div class="sidebar-banner">
            📰 News Control Panel
        </div>
        """, unsafe_allow_html=True)


        # 🖼️ Image (change spam.png → news image)
    st.sidebar.image("news_2.png")
    

        # 🔹 About Project
    st.sidebar.markdown(
        "<h2 style='color:#00FFFF;'>🤖 About Project</h2>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        "<p style='color:#E0E0E0;'>AI-powered system to classify news into categories like "
        "<b style='color:orange;'>world</b>, <b style='color:green;'>Sports</b>, "
        "<b style='color:#FF69B4;'>Business</b>, <b style='color:#FFD700;'>Science/Technology</b> 📰</p>"
        "<p style='color:#FFD700;'>⚡ Real-time prediction using Machine Learning</p>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Categories Supported
    st.sidebar.markdown(
        "<h3 style='color:#FF4B4B;'>📂 Categories</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        ✔ 🏛️ <span style='color:#FFD93D;'>World</span><br>
        ✔ ⚽ <span style='color:#4D96FF;'>Sports</span><br>
        ✔ 💰 <span style='color:#6BCB77;'>Business</span><br>
        ✔ 💻 <span style='color:#00FFFF;'>Science/Technology</span>
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Libraries Used
    st.sidebar.markdown(
        "<h3 style='color:#00FFAA;'>🐍 Libraries Used</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        ✔ <span style='color:#FFD93D;'>Tensorflow</span><br>
        ✔ <span style='color:#4D96FF;'>Keras</span><br>
        ✔ <span style='color:#6BCB77;'>Pandas</span><br>
        ✔ <span style='color:#FF6B6B;'>NumPy</span><br>
        ✔ <span style='color:#00FFFF;'>Streamlit</span>
        ✔ <span style='color:#00FFFF;'>Joblib</span>
        ✔ <span style='color:#00FFFF;'>Matplotlib</span>
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Model Info
    st.sidebar.markdown(
        "<h3 style='color:#FFA500;'>📊 Model Insights</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        🤖 <span style='color:#FFD700;'>Model:</span>Embeddings<br>
        📈 <span style='color:#00FFAA;'>High Accuracy Text Classification</span><br>
        ⚡ <span style='color:#4D96FF;'>Fast Prediction Speed</span><br>
        🔍 <span style='color:#FF6B6B;'>Text Preprocessing (Cleaning + Stopwords Removal)</span>
        🔍 <span style='color:#FF6B6B;'>LSTM</span>
        🤖 <span style='color:#FFD700;'>Model:</span>Sequential<br>
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Smart Facts (updated)
    st.sidebar.markdown(
        "<h3 style='color:#FF69B4;'>💡 Smart Facts</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        📰 <span style='color:#FFD700;'>Thousands of news articles published daily</span><br>
        🤖 <span style='color:#00FFAA;'>AI helps organize and filter information</span><br>
        ⚡ <span style='color:#4D96FF;'>Classification improves content recommendation</span><br>
        📊 <span style='color:#FF4B4B;'>Used in Google News & media platforms</span>
        📊 <span style='color:#FF4B4B;'>This Model is Trained on 120000 Total Rows</span>
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Use Case (NEW ADD 🔥)
    st.sidebar.markdown(
        "<h3 style='color:#9D4EDD;'>🚀 Use Cases</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        ✔ Personalized News Feed<br>
        ✔ Fake News Detection (Extension)<br>
        ✔ News Recommendation Systems<br>
        ✔ Media Content Filtering
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 About App
    st.sidebar.markdown(
        "<h3 style='color:#00FFFF;'>📱 About App</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        "<p style='color:white;'>Enter any news headline and AI will classify it into the correct category 📰</p>"
        "<p style='color:#00FFFF;'>Built using Deep Learning & RNN 🚀</p>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🔹 Contact
    st.sidebar.markdown(
        "<h3 style='color:#FF4B4B;'>📞 Contact Us</h3>",
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        📱 <span style='color:#FFD700;'>9058068999</span><br>
        📧 <span style='color:#00FFFF;'>Varshneyd110@gmail.com</span>
        """,
        unsafe_allow_html=True
        )

    st.sidebar.markdown("---")


        # 🏁 Footer Image
    st.sidebar.image("flag.jpg")


    col1,col2,col3,col4=st.columns([2,1,1,1])

    with col1:
        st.markdown(" ###  Welcome to 🛡️ AI-Powered News Classification System")

    with col4:

        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()


    import streamlit.components.v1 as components
    import base64

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_base64 = get_base64_image("spam_1.jpg")

    components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>

        .banner {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            padding: 70px 20px;
            border-radius: 20px;
            text-align: center;
            color: white;
            position: relative;
        }}

        .banner::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.6);
            border-radius: 20px;
        }}

        .banner-content {{
            position: relative;
            z-index: 2;
        }}

        .title {{
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #ffcc70, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: 0.3s;
    }}

    .title:hover {{
        transform: scale(1.05);
        background: linear-gradient(90deg, #00ffcc, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

        .subtitle {{
            font-size: 18px;
            margin-top: 10px;
        }}

        .rotator {{
            height: 180px;
            overflow: hidden;
        }}

        .rotator-inner {{
            display: flex;
            flex-direction: row;
            width: 250%;
            animation: scrollLeft 8s linear infinite;
        }}

        .slide {{
            min-width: 50%;
            flex-shrink: 0;
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        @keyframes scrollLeft {{
        0% {{
            transform: translateX(0);
        }}
        100% {{
            transform: translateX(-50%);
        }}
    }}

        </style>
        </head>

        <body>

        <div class="banner">
            <div class="banner-content">

                <div class="rotator-inner">

                            <!-- ORIGINAL -->
                            <div class="slide">
                                <div class="title">📰 News Classification System</div>
                                <div class="subtitle">Classifying Headlines into World News, Sports News, Business News, Science/Technology News</div>
                                <div id="clock"></div>
                            </div>

                            <div class="slide">
                                <div class="title">📰 News Classification System</div>
                                <div class="subtitle">Classifying Headlines into World News, Sports News, Business News, Science/Technology News</div>
                                <div id="clock2"></div>
                            </div>

                            <!-- DUPLICATE (same content again) -->
                            <div class="slide">
                                <div class="title">📰 News Classification System</div>
                                <div class="subtitle">Classifying Headlines into World News, Sports News, Business News, Science/Technology News</div>
                                <div id="clock3"></div>
                            </div>

                            <div class="slide">
                                <div class="title">📰 News Classification System</div>
                                <div class="subtitle">Classifying Headlines into World News, Sports News, Business News, Science/Technology News</div>
                                <div id="clock4"></div>
                            </div>

                        </div>

            </div>
        </div>

        <script>
        function updateClock() {{
            var now = new Date();

            var options = {{
                weekday: 'long',
                day: '2-digit',
                month: 'short',
                year: 'numeric'
            }};

            var date = now.toLocaleDateString('en-IN', options);
            var time = now.toLocaleTimeString();

            var html = `
            <div style="
                font-size:30px;
                font-weight:bold;
                color:#00ffcc;
                text-shadow:0 0 10px #00ffcc, 0 0 20px #00ffcc;
                text-align:center;
            ">
            ⏰ ${{date}} ${{time}}
            </div>
            `;

            document.getElementById("clock").innerHTML = html;
            document.getElementById("clock2").innerHTML = html;
            document.getElementById("clock3").innerHTML = html;
            document.getElementById("clock4").innerHTML = html;
        }}

        setInterval(updateClock, 1000);
        updateClock();
        </script>

        </body>
        </html>
        """, height=300)
    
    st.write("\n")
    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("""
            <style>
            .banner-enter {
                padding: 12px 20px;
                border-radius: 12px;
                background: linear-gradient(135deg, #ff416c, #ff4b2b);
                color: white;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                transition: all 0.4s ease;
                cursor: pointer;
                margin-top:40px;
            }

            .banner-enter:hover {
                background: linear-gradient(135deg, #ff416c, #ff4b2b);
                color: black;
                transform: scale(1.05);
            }
            </style>

            <div class="banner-enter">
                📰 Enter News Headlines
            </div>
            """, unsafe_allow_html=True)

    News = st.text_input("", placeholder="📰 Enter your News Headlines here")

    col1, col2, col3 = st.columns([1,3,1])

    with col2:

        if st.button(
            "🚀 Predict",
            use_container_width=True
        ):

            category, confidence = predict_news(News)

            # CATEGORY STYLES

            category_styles = {

                "🌍 World": {
                    "color": "linear-gradient(90deg, #1e3c72, #2a5298)",
                    "emoji": "🌍"
                },

                "🏅 Sports": {
                    "color": "linear-gradient(90deg, #11998e, #38ef7d)",
                    "emoji": "🏅"
                },

                "💼 Business": {
                    "color": "linear-gradient(90deg, #f7971e, #ffd200)",
                    "emoji": "💼"
                },

                "💻 Sci/Tech": {
                    "color": "linear-gradient(90deg, #8e2de2, #4a00e0)",
                    "emoji": "💻"
                }
            }

            # DEFAULT STYLE
    
            style = category_styles.get(
                category,
                {
                    "color": "linear-gradient(90deg, #333, #777)",
                    "emoji": "📰"
                }
            )

                # 🔥 SAVE TO DATABASE
            save_news_prediction(News, category, confidence)

            # PREDICTION CARD

            # 🎨 UI Card
            st.markdown(f"""
                    <div style="
                        background: {style['color']};
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        font-size: 22px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        transition: 0.3s;
                    ">
                        {style['emoji']} <b>{category}</b><br><br>
                        Confidence Score: <b>{confidence:.2f}%</b>
                    </div>
                """, unsafe_allow_html=True)
            
            
            # ANIMATIONS

            st.balloons()
            st.snow()

    col1, col2 , col3 = st.columns(3)

    with col1:

        st.markdown("""
            <style>
            .banner-bulk {
                padding: 12px 20px;
                border-radius: 12px;
                background: linear-gradient(135deg, #1d976c, #93f9b9);
                color: white;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                transition: all 0.4s ease;
                cursor: pointer;
                margin-top:40px;
                margin-bottom:20px;
                
            }

            .banner-bulk:hover {
                background: linear-gradient(135deg, #1d976c, #93f9b9);
                color: black;
                transform: scale(1.05);
            }
            </style>

            <div class="banner-bulk">
                📊 Bulk Prediction
            </div>
            """, unsafe_allow_html=True)
        
    file=st.file_uploader("select file",type=["csv","txt"])
    if file:
            df=pd.read_csv(file,names=["Message"])
            placeholder=st.empty()
            placeholder.dataframe(df)

        
    col1, col2, col3 = st.columns([1,3,1])

    with col2:

        if st.button("Predict", key="b2", use_container_width=True):

            df.columns = df.columns.str.strip().str.lower()

            df.rename(columns={df.columns[0]: "news"}, inplace=True)

            predictions = []
            confidences = []


            # 🔥 LOOP THROUGH ALL NEWS
            for text in df["news"]:

                category, confidence = predict_news(text)

                predictions.append(category)

                confidences.append(round(confidence, 2))

            df["Predictions"] = predictions
            df["Confidences"] = confidences


            placeholder.dataframe(df)

            # 🔥 SAVE BULK DATA
            for i in range(len(df)):
                save_news_prediction(
                    df["news"][i],
                    df["Predictions"][i],
                    df["Confidences"][i]
                )

            # 📊 Summary Calculation
            counts = df["Predictions"].value_counts()

            # 🎉 Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🌍 World",
                    counts.get("🌍 World", 0)
                )

            with col2:
                st.metric(
                    "🏅 Sports",
                    counts.get("🏅 Sports", 0)
                )

            with col3:
                st.metric(
                    "💼 Business",
                    counts.get("💼 Business", 0)
                )

            with col4:
                st.metric(
                    "💻 Sci/Tech",
                    counts.get("💻 Sci/Tech", 0)
                )

            st.metric(
                "📊 Total News",
                len(df)
            )

            st.success("🎉 Bulk Prediction Completed!")

            st.markdown("""
                <div style="
                    background: linear-gradient(90deg, #1e3c72, #2a5298);
                    padding: 12px;
                    border-radius: 10px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                ">
                📰 AI classifying news into multiple categories in real-time 🚀
                </div>
                """, unsafe_allow_html=True)
            
            # 🎨 Metric Styling
            st.markdown("""
                <style>
                [data-testid="metric-container"] {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 15px;
                    border-radius: 12px;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)


                # 📊 Category Count (IMPORTANT)
            counts = df["Predictions"].value_counts()

            data = {
                    "Predictions": counts.index.tolist(),
                    "Count": counts.values.tolist()
                }

            import plotly.express as px

            fig = px.pie(
                    values=data["Count"],
                    names=data["Predictions"],
                    hole=0.4
                )

            fig.update_layout(
                    width=400,
                    height=400
                )

            st.plotly_chart(fig, use_container_width=False)


                # 🎉 Effects
            st.balloons()
            st.snow()

        sample_news_data = pd.DataFrame({

            "news": [
    "United Nations discusses climate crisis and global peace",
    "India defeats Australia in thrilling cricket world cup final",
    "Stock markets rise after strong technology company earnings",
    "Scientists develop advanced artificial intelligence system for healthcare",
    "Russia and Ukraine hold peace talks after international pressure",
    "Lionel Messi scores hat trick in international football match",
    "Tesla reports record profits in quarterly financial results",
    "NASA prepares mission to explore distant planets in space",
    "World leaders gather for emergency meeting on global warming",
    "Olympic committee announces host country for next games",
    "Oil prices fall as global demand slows down",
    "New smartphone launch introduces powerful AI camera features",
    "China signs new trade agreement with European countries",
    "Tennis champion wins grand slam title after dramatic comeback",
    "Amazon announces major investment in logistics expansion",
    "Researchers discover breakthrough in renewable energy technology",
    "Middle East tensions rise after border conflict",
    "Manchester City wins premier league title again",
    "Central bank cuts interest rates to support economy",
    "Cybersecurity experts warn about rising online hacking attacks",
    "Global food crisis worsens due to extreme weather conditions",
    "Basketball star breaks scoring record in playoff game",
    "Bitcoin price jumps after institutional investments increase",
    "SpaceX successfully launches satellite into earth orbit",
    "NATO announces new international security partnership",
    "Virat Kohli scores century in important tournament match",
    "Apple becomes the worlds most valuable company again",
    "Tech company unveils next generation quantum computer",
    "Several countries demand ceasefire during UN assembly",
    "Formula One driver wins rain affected championship race",
    "Global recession fears impact international financial markets",
    "Medical researchers create innovative treatment using biotechnology",
    "International summit focuses on renewable energy solutions",
    "Football fans celebrate victory after intense final match",
    "Major startup secures billion dollar funding from investors",
    "Google announces major updates to artificial intelligence tools",
    "Global health organizations warn about new virus outbreak",
    "National team begins preparation for upcoming world tournament",
    "Gold prices increase amid global economic uncertainty",
    "Electric vehicle technology continues to transform transportation industry",
    "European Union introduces stricter immigration policies",
    "Young athlete wins gold medal in international championship",
    "International companies expand operations into Asian markets",
    "Scientists use robotics to improve surgical precision in hospitals",
    "Massive earthquake impacts several countries across Asia",
    "Cricket board announces new domestic league schedule",
    "Banking sector reports strong growth in annual revenue",
    "Software engineers develop faster cloud computing systems",
    "UN calls for urgent humanitarian aid in war affected region",
    "Coach reveals strategy ahead of major football competition",
    "Retail sales surge during holiday shopping season",
    "New battery technology promises longer electric vehicle range",
    "World economies prepare for international climate agreement",
    "Star striker signs record breaking contract with club",
    "Technology firms announce thousands of new job openings",
    "Artificial intelligence helps detect diseases at early stage",
    "Peacekeeping forces deployed after regional conflict escalates",
    "India secures series victory after dominant team performance",
    "Economic experts predict steady market recovery next year",
    "Global tech conference showcases futuristic innovations and gadgets"
]
        })

        # CSV DOWNLOAD
        csv = sample_news_data.to_csv(
            index=False,
            header=False
        ).encode("utf-8")

        # 🎨 Banner
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin-top: 20px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                    ">
                <h1 style="
                    color: white;
                    margin-bottom: 10px;
                    font-size: 24px;
                    letter-spacing: 1px;
                        ">
                    📥 Don't have a dataset?
                </h1>
                <p style="
                    color: #dcdcdc;
                    font-size: 14px;
                    margin-bottom: 20px;
                        ">
                    Download a ready-to-use sample dataset and start predicting instantly 🚀
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>

        /* 🔥 Base Button */
        .custom-btn {
            display: block;
            width: 85%;              /* width increase */
            margin: auto;            /* center align */
            padding: 18px 0;
            border-radius: 15px;
            font-weight: bold;
            font-size: 18px;
            text-decoration: none;
            text-align: center;
            color: #ffffff;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
                    
            animation: glow 2s infinite alternate;

            @keyframes glow {
                from { box-shadow: 0 0 10px rgba(255,255,255,0.2); }
                to { box-shadow: 0 0 25px rgba(255,255,255,0.6); }
            }
        }

        /* ✨ Hover Common */
        .custom-btn:hover {
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        }

        

        /* 🔵 Email */
        .email {
            background: rgba(0, 114, 255, 0.25);
            border: 1px solid rgba(0, 114, 255, 0.4);
            color: #ffffff;
            text-shadow: 0 0 8px rgba(255,255,255,0.7);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
            
        }
        .email:hover {
            background: rgba(0, 114, 255, 0.4);
            transform: translateY(-5px) scale(1.07);
            box-shadow: 0 12px 40px rgba(0, 114, 255, 0.6);
        }

        

        </style>
        """, unsafe_allow_html=True)


        import base64

        def create_download_link(csv, filename, text, cls):
            b64 = base64.b64encode(csv).decode()
            return f'<a class="custom-btn {cls}" href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        
        # 🔥 Layout
        col1, col2, col3 = st.columns(3)

        

        with col2:
            st.markdown(
                create_download_link(csv, "news.csv", "⬇️ Different News Dataset", "news"),
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)

        with col1:
            show_btn = st.button("📂 Show History",use_container_width=True)

        

        with col2:
            if st.button("🗑️ Clear History"):
                clear_news_history()
                st.success("🗑️ History Cleared!")

        if show_btn:
            data = get_all_news_predictions()

            
            history_df = pd.DataFrame(
                data,
                columns=["ID", "News", "Prediction", "Confidence"]
            )

            st.dataframe(history_df)

        



        





        
    st.markdown("""
        <style>

        /* File uploader container */
        .stFileUploader {
            border: 2px dashed #00c6ff;
            border-radius: 15px;
            padding: 20px;
            background: linear-gradient(135deg, #1f1f2e, #2c2c54);
            transition: all 0.4s ease;
        }

        /* Hover effect */
        .stFileUploader:hover {
            border: 2px solid #00f2fe;
            box-shadow: 0 0 15px #00c6ff;
        }

        /* Upload button */
        .stFileUploader button {
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 8px 15px;
            border: none;
            transition: all 0.3s ease;
        }

        /* Button hover */
        .stFileUploader button:hover {
            background: linear-gradient(135deg, #ff512f, #dd2476);
            color: black;
            transform: scale(1.05);
        }

        /* Uploaded file text */
        .stFileUploader div {
            color: #ffffff;
            font-size: 16px;
        }

        </style>
        """, unsafe_allow_html=True)



    st.markdown("""
        <style>

        /* Input box styling */
        .stTextInput > div > div > input {
            background: linear-gradient(135deg, #1f1f2e, #2c2c54);
            color: #ffffff;
            font-size: 20px;
            padding: 12px;
            border-radius: 12px;
            border: 2px solid #00c6ff;
            outline: none;
            transition: all 0.4s ease;
        }

        /* Placeholder color */
        .stTextInput > div > div > input::placeholder {
            color: #bbbbbb;
            font-size: 16px;
        }

        /* Hover effect */
        .stTextInput > div > div > input:hover {
            border: 2px solid #00f2fe;
            box-shadow: 0 0 10px #00c6ff;
        }

        /* Focus (click) effect 🔥 */
        .stTextInput > div > div > input:focus {
            border: 2px solid #00f2fe;
            box-shadow: 0 0 15px #00f2fe, 0 0 25px #00c6ff;
            background: linear-gradient(135deg, #141e30, #243b55);
        }

        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>

    /* 🎯 All Buttons Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #ff8008, #ffc837);
        color: white;
        font-size: 40px;
        font-weight: bold;
        padding: 18px 50px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(0, 114, 255, 0.4);
        width: 100%;
        margin-top: 20px;
    }

    /* 🔥 Hover Effect */
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        box-shadow: 0 10px 25px rgba(0, 114, 255, 0.6);
        font-size: 50px;
        font-weight: bold;
        padding: 20px 55px;
    }

    /* Click Effect */
    div.stButton > button:active {
        transform: scale(0.98);
    }

    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
        <style>
        @keyframes fadeText {
            0% {opacity: 0;}
            5% {opacity: 1;}
            12% {opacity: 1;}
            17% {opacity: 0;}
            100% {opacity: 0;}
        }

        .text-container {
            position: relative;
            height: 40px;
            text-align: left;
            color: #00c6ff;
            font-size: 22px;
            font-weight: bold;
            margin-top: 40px;
            overflow: hidden; /* 🔥 important */
        }

        .text-container span {
            position: absolute;
            width: 100%;
            opacity: 0;
            animation: fadeText 16s linear infinite; /* 🔥 total = 8 × 2s */
        }

        /* Proper spacing */
        .text-container span:nth-child(1) { animation-delay: 0s; }
        .text-container span:nth-child(2) { animation-delay: 2s; }
        .text-container span:nth-child(3) { animation-delay: 4s; }
        .text-container span:nth-child(4) { animation-delay: 6s; }
        .text-container span:nth-child(5) { animation-delay: 8s; }
        .text-container span:nth-child(6) { animation-delay: 10s; }
        .text-container span:nth-child(7) { animation-delay: 12s; }
        .text-container span:nth-child(8) { animation-delay: 14s; }
        </style>

        <div class="text-container">
            <span>📰 Classify News Headlines in Seconds</span>
            <span>🤖 AI-Powered News Categorization</span>
            <span>📊 Get Smart Insights from News Data</span>
            <span>⚡ Real-Time News Classification</span>
            <span>🔍 Identify News Categories Instantly</span>
            <span>🧠 Powered by Machine Learning & NLP</span>
            <span>📂 Organize News into Smart Categories</span>
            <span>🚀 Fast & Accurate News Predictions</span>
        </div>
        """, unsafe_allow_html=True)

        
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1f4037, #99f2c8);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>📰 Total News Classified</h4>
                <h2>120k+</h2>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #141e30, #243b55);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>🎯 Model Accuracy</h4>
                <h2>95%</h2>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #42275a, #734b6d);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-top: 40px;
            ">
                <h4>📊 Categories Covered</h4>
                <h2>4 Types</h2>
            </div>
            """, unsafe_allow_html=True)
        
    st.markdown("""
    <style>

    /* Sidebar background */
    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #0f172a,
            #1e293b
        );

        border-right: 2px solid #334155;
    }



    /* Sidebar title */
    .sidebar-title {

        color: white;

        font-size: 28px;

        font-weight: bold;

        text-align: center;

        margin-bottom: 20px;
    }



    /* Selectbox label */
    [data-testid="stSidebar"] label {

        color: #f8fafc !important;

        font-size: 18px !important;

        font-weight: bold;
    }



    /* Selectbox */
    [data-baseweb="select"] {

        background-color: #1e293b !important;

        border-radius: 12px !important;

        border: 2px solid #38bdf8 !important;
    }



    /* Hover effect */
    [data-baseweb="select"]:hover {

        border: 2px solid #00ffcc !important;

        box-shadow: 0 0 10px #00ffcc;
    }

    </style>
    """, unsafe_allow_html=True)

    



    

    








if st.session_state.logged_in == False:
    login_news_page()
else:
    main_app()



    

    