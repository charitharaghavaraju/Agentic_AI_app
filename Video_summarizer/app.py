import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="📹",
    layout="wide"
)

st.title("Multimodal AI Agent - Video Summarizer")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():

    return Agent(
        name="Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

## Initialize the agent
multimodal_agent = initialize_agent()

## File uploader 
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video file to summarize.")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    
    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeing from the video?",
        placeholder="Ask a question about the video content.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("Analyse Video", key="analyse_video_button"):
        if not user_query:
            st.warning("Please provide a query or insight to analyze the video.")

        else:
            try:
                with st.spinner("Analyzing video..."):
                    ## Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    ## Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and suppllementary web search:
                        {user_query}
                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    ## AI agent processing
                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])

                ## Display response
                st.subheader("Analysis Results")
                st.markdown(response.content)

            except Exception as e:
                st.error(f"An error occurred during video analysis: {e}")

            finally:
                ## Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis")


# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)