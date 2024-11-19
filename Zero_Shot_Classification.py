import streamlit as st
from transformers import pipeline

# Initialize Hugging Face Zero-Shot Classification pipeline
classifier = pipeline('zero-shot-classification')

# Define candidate labels (design parameters)
candidate_labels = ['material', 'color', 'size', 'engine', 'horsepower', 'feature', 'design', 'finish']

# Streamlit App
st.title("Design Requirements Chat Interface")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.write(f"**User:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")

# User input
user_input = st.text_input("Enter your design requirements:", "")

if st.button("Submit"):
    if user_input:
        # Append user message to chat history
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        # Perform zero-shot classification
        classification_results = classifier(user_input, candidate_labels)
        labels_scores = list(zip(classification_results['labels'], classification_results['scores']))

        # Generate assistant response
        assistant_response = f"""
**Extracted Parameters:**

**Classification Results:**
{labels_scores}
"""

        # Append assistant message to chat history
        st.session_state['messages'].append({'role': 'assistant', 'content': assistant_response})

        # Display assistant response
        st.write(f"**Assistant:** {assistant_response}")
    else:
        st.warning("Please enter your design requirements.")
