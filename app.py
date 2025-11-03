import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Cache the model loading for better performance
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

# Load Hugging Face sentiment pipeline
try:
    sentiment = load_sentiment_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI components
st.title("😊 Sentiment Analysis App")
st.markdown("Analyze the sentiment of your text using AI!")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    show_confidence = st.checkbox("Show confidence score", value=True)
    show_raw_output = st.checkbox("Show raw output", value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here...",
        height=150
    )

with col2:
    st.markdown("### Examples:")
    example_texts = [
        "I love this product! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today."
    ]

    for example in example_texts:
        if st.button(example, key=example):
            user_text = example

if st.button("Analyze Sentiment", type="primary") or user_text:
    if user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            # Run the sentiment model
            results = sentiment(user_text)

        for res in results:
            label = res["label"]
            score = res["score"]

            # Display results with emojis
            st.subheader("Results:")

            if label == "POSITIVE":
                emoji = "😊"
                color = "green"
            elif label == "NEGATIVE":
                emoji = "😞"
                color = "red"
            else:
                emoji = "😐"
                color = "orange"

            st.markdown(f"### {emoji} Sentiment: **:{color}[{label}]**")

            if show_confidence:
                # Create a progress bar for confidence score
                st.metric("Confidence Score", f"{score:.2%}")
                st.progress(score)

            if show_raw_output:
                st.json(res)
    else:
        st.warning("Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("Powered by Hugging Face Transformers 🤗")