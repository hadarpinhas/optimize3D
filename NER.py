# Import necessary libraries
import streamlit as st
from transformers import pipeline
import re

# Initialize Hugging Face NER pipeline
ner_pipeline = pipeline('ner', model="dslim/bert-base-NER", grouped_entities=True)

# Streamlit App
st.title("Design Requirements Chat Interface")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
st.header("Chat History")
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.markdown(f"**User:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# User input
st.header("Enter Your Design Requirements")
user_input = st.text_input("What do you need?", "")

# Process input when the "Submit" button is pressed
if st.button("Submit"):
    if user_input.strip():
        # Append user message to chat history
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        # Process input with Hugging Face NER pipeline
        ner_results = ner_pipeline(user_input)

        # Extract dimensions using regex
        dimension_pattern = r'(\d+(\.\d+)?)\s*(feet|meters|cm|inches)\s*(long|wide|tall)?'
        dimensions = re.findall(dimension_pattern, user_input)

        # Extract material and other design keywords
        keywords = []
        if "wooden" in user_input.lower():
            keywords.append("wooden")

        # Combine NER and regex results
        entities = [(entity['word'], entity['entity_group']) for entity in ner_results]
        assistant_response = f"""
        **Extracted Parameters:**

        **Hugging Face Entities:** {entities}  
        **Dimensions:** {dimensions}  
        **Keywords:** {keywords}  
        """

        # Append assistant message to chat history
        st.session_state['messages'].append({'role': 'assistant', 'content': assistant_response})

        # Display assistant response
        st.markdown(f"**Assistant:** {assistant_response}")
    else:
        st.warning("Please enter your design requirements.")

### run: streamlit run NER.py


# Examople: I need a red sports car with a V8 engine, 400 horsepower, and leather seats. It should have a sunroof and advanced navigation system.

# output:
# **Extracted Parameters:**

# **Hugging Face Entities:**
# [('red', 'MISC'), ('V8', 'MISC'), ('400', 'MISC')]