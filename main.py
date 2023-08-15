import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import assemblyai as aai
import openai
import re
import time
import os
import io
import requests
import json  # Make sure to import the json module
import streamlit as st
import copy
import torch
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile
from transformers import pipeline, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertForQuestionAnswering, BertForQuestionAnswering, BertTokenizer





# Set your AssemblyAI API key
ASSEMBLYAI_API_KEY = "Enter your key here"

# Set your OpenAI API key
OPENAI_API_KEY = 'Enter your key here'

# Initialize AssemblyAI and OpenAI API keys
aai.settings.api_key = ASSEMBLYAI_API_KEY
openai.api_key = OPENAI_API_KEY



# Initialize sentiment analysis pipeline
#classifier = pipeline('sentiment-analysis')
# Initialize AssemblyAI API key
headers = {
    'authorization': ASSEMBLYAI_API_KEY,
    'content-type': 'application/json'
}

def transcribe_audio(audio_url):
    url = 'https://api.assemblyai.com/v2/transcript'
    
    data = {
        'audio_url': audio_url
    }

    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()

    # Check if response indicates success and has the transcript ID
    if 'id' in response_data:
        transcript_id = response_data['id']
        return transcript_id
    else:
        st.write("Failed to initiate transcription.")
        st.write(response_data)  # You can print the response to diagnose the issue
        return None

def get_transcript_text(transcript_id):
    url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
    
    response = requests.get(url, headers=headers)
    transcript_text = response.json()['text']
    return transcript_text

def main():
    st.title("J.A.C. AI-Driven Analytics Demo")

    # Set your AssemblyAI API key
    ASSEMBLYAI_API_KEY = "Enter your key here"

    # Set your Hugging Face model name
    HF_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

    # Initialize AssemblyAI API key
    headers = {
        'authorization': ASSEMBLYAI_API_KEY,
        'content-type': 'application/json'
    }

    transcript_id = st.session_state.transcript_id if hasattr(st.session_state, 'transcript_id') else None
    transcript_text = st.session_state.transcript_text if hasattr(st.session_state, 'transcript_text') else None
    question = None

    # Input audio URL and transcribe
    audio_url = st.text_input("Enter the URL of the audio file:")
    if st.button("Transcribe"):
        transcript_id = transcribe_audio(audio_url)
        if transcript_id is not None:
            st.session_state.transcript_id = transcript_id

            # Wait for the transcription to complete and get the transcript text
            st.write("Waiting for transcription to complete...")
            while transcript_text is None:
                transcript_text = get_transcript_text(transcript_id)
                st.session_state.transcript_text = transcript_text
                time.sleep(5)

    # Display transcript text
    if transcript_text is not None:
        st.subheader("Transcript Text:")
        st.write(transcript_text)

        # Analyze section
        st.subheader("Question Analysis")
        question = st.text_input("Enter a question:")
        if st.button("Analyze") and question:
            st.write("Analyzing question:", question)

            try:
                # Load the fine-tuned BERT model and tokenizer
                model = BertForQuestionAnswering.from_pretrained(HF_MODEL_NAME)
                tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)

                # Initialize pipeline for question answering
                nlp_qa = pipeline("question-answering", model=HF_MODEL_NAME, tokenizer=HF_MODEL_NAME)

                # Initialize variables for the best answer
                best_answer = None
                best_score = 0


                best_sentence = None  # Initialize best_sentence outside the loop

                # Loop through the transcript and consider overlapping segments
                window_size = 512  # Adjust this based on your needs
                stride = 100  # Adjust this based on your needs
                for start in range(0, len(transcript_text) - window_size + 1, stride):
                    end = start + window_size
                    segment = transcript_text[start:end]

                    # Use pipeline for question answering
                    result = nlp_qa(question=question, context=segment)

                    # Use fine-tuned BERT model for question answering
                    inputs = tokenizer(question, segment, add_special_tokens=True, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        start_probs = outputs.start_logits.softmax(dim=-1)
                        end_probs = outputs.end_logits.softmax(dim=-1)
                        answer_start = torch.argmax(start_probs)
                        answer_end = torch.argmax(end_probs) + 1
                        if answer_start < answer_end:
                            answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end], skip_special_tokens=True)
                            score = start_probs[0][answer_start] + end_probs[0][answer_end - 1]
                            if score > best_score:
                                best_score = score
                                best_answer = answer
                                best_segment = segment  # Store the current segment as the best segment

                                # Find the sentence containing the answer in the best segment
                                best_segment_sentences = best_segment.split('.')
                                for sentence in best_segment_sentences:
                                    if best_answer in sentence:
                                        best_sentence = sentence.strip()
                                        break
                if best_answer:
                    st.write("Answer (BERT):", best_answer)
                    st.write("Answer (Pipeline):", result['answer'])
                    st.write("Segment:", best_segment)  # Print the best segment

                    # Find the sentence containing the answer in the best segment
                    best_segment_sentences = best_segment.split('.')
                    for sentence in best_segment_sentences:
                        if best_answer in sentence:
                            best_sentence = sentence.strip()
                            st.write("Best Sentence:", best_sentence)  # Print the best sentence
                            break
                    else:
                        st.write("No answer found for the provided question.")
                else:
                    st.write("No answer found for the provided question.")

            
            except Exception as e:
                st.write("Exception during question analysis:", e)

if __name__ == "__main__":
    main()

