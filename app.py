import streamlit as st
import sqlite3
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import pandas as pd

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('grievances.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT DEFAULT 'Pending',
                date_submitted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

# Function to determine complaint priority based on sentiment
def determine_priority(sentiment):
    if sentiment == 'negative':
        return 1
    elif sentiment == 'neutral':
        return 2
    else:
        return 3

# Function to analyze sentiment
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < 0:
        return 'negative'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'positive'

# Streamlit interface
st.title("AI-Based Grievance and Feedback System")
st.write("Please fill out the form below to submit your grievance or feedback.")

user = st.text_input("Your Name:")
category = st.selectbox("Category:", ["Hostel", "Labs", "Classrooms", "Other"])
content = st.text_area("Describe the issue:")

if st.button("Submit"):
    if user and content:
        sentiment = analyze_sentiment(content)
        priority = determine_priority(sentiment)
        c.execute("INSERT INTO grievances (user, category, content, sentiment, priority) VALUES (?, ?, ?, ?, ?)", 
                  (user, category, content, sentiment, priority))
        conn.commit()
        st.success("Your grievance has been submitted successfully!")
    else:
        st.error("Please fill out all fields.")

# Display the grievances
st.subheader("Submitted Grievances")
grievances_df = pd.read_sql_query("SELECT * FROM grievances ORDER BY priority", conn)
st.write(grievances_df)
