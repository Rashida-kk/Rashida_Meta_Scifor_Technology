# -*- coding: utf-8 -*-
"""quizApp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gdn60G-5hJ2_6_NQXCrrVx1Txx0v_Qli
"""

import streamlit as st

# Title of the app
st.title("Simple Quiz App")

# Instruction text
st.write("Answer the following questions:")

# Create a form for the quiz
with st.form("quiz_form"):

    # First question
    question1 = st.radio(
        "1. What is the capital of France?",
        ('London', 'Berlin', 'Paris', 'Rome') )

    # Second question
    question2 = st.selectbox(
        "2. Which planet is known as the Red Planet?",
        ('Earth', 'Mars', 'Jupiter', 'Venus'))

    # Third question
    question3 = st.text_input("3. Who developed the theory of relativity?")

    # Submit button
    submitted = st.form_submit_button("Submit")

# Logic for evaluating the quiz after submission
if submitted:
    # Initial score
    score = 0

    # Check answers
    if question1 == 'Paris':
        score += 1
    if question2 == 'Mars':
        score += 1
    if question3.lower() == 'einstein':
        score += 1

    # Display score
    st.write(f"Your score is: {score}/3")

    # Provide feedback based on the score
    if score == 3:
        st.success("Excellent! You got everything right!")
    elif score == 2:
        st.info("Good job! You got 2 out of 3 correct.")
    else:
        st.warning("Keep trying! You can improve.")