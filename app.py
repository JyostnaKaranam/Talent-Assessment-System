import time
import streamlit as st
import mysql.connector
import bcrypt
import pickle
import docx
import PyPDF2
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config should be called first
st.set_page_config(page_title="Talent Assessment System", layout="wide")

# Database connection
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root@TES",
        database="Project"
    )

# Hashing functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Signup function
def signup_user(username, email, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_pw))
        conn.commit()
        st.success("Signup successful! You can now log in.")
    except mysql.connector.IntegrityError:
        st.error("This email is already registered.")
    finally:
        cursor.close()
        conn.close()

# Login function
def login_user(email, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        hashed_pw = result[0]
        if check_password(password, hashed_pw.encode('utf-8')):
            return True
    return False

# Resume matching functions (same as provided above)
model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")
    return text

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = model.predict(vectorized_text)
    predicted_category_name = encoder.inverse_transform(predicted_category)
    return predicted_category_name[0]

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# Function to display the pie chart
def display_pie_chart(score):
    labels = ['Match', 'Mismatch']
    sizes = [score * 100, (1 - score) * 100]
    colors = ['#66b3ff', '#ff6666']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    
    st.pyplot(fig)

# Streamlit app layout
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

    if not st.session_state.logged_in:
        menu = st.sidebar.selectbox("Talent Assessment System", ["Signup", "Login"])

        if menu == "Signup":
            st.subheader("Signup")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            if st.button("Signup"):
                if username and email and password and confirm_password:
                    if not email.endswith("@gmail.com"):
                        st.error("Please enter a valid Gmail address.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        signup_user(username, email, password)
                        st.session_state.logged_in = True
                        st.session_state.username = username
            else:
                st.error("Please fill in all fields.")

        elif menu == "Login":
            st.subheader("Login to Your Account")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if email and password:
                    if login_user(email, password):
                        st.success("Login successful!")
                        st.session_state.logged_in = True
                        st.session_state.username = email
                    else:
                        st.error("Invalid email or password.")
                else:
                    st.error("Please fill in all fields.")

    else:
        # Dashboard page for logged-in users
        st.sidebar.selectbox("Dashboard", ["Resume Matching"])

        st.title("Resume Matching")
        uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx"])

        if uploaded_file is not None:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")
            
            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=200)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"**{category}**")

            # Job Description Input
            job_description = st.text_area("Enter Job Description:", height=200)

            # Process Resume and Calculate Score
            if st.button("Calculate Match Score"):
                if not job_description.strip():
                    st.error("Please enter a job description.")
                elif not uploaded_file:
                    st.error("Please upload a resume file.")
                else:
                    resume_text = handle_file_upload(uploaded_file)
                    if not resume_text.strip():
                        st.error("Could not extract text from the resume. Please upload a valid file.")
                    else:
                        vectors = tfidf.transform([job_description, resume_text]).toarray()
                        similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
                        
                        st.success(f"Match Score: {round(similarity_score * 100, 2)}%")

                        # Display the match score in a pie chart
                        display_pie_chart(similarity_score)

        if st.button("Logout"):
            logout()  # Calls the logout function to reset the session state


if __name__ == "__main__":
    main()
