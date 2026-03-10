import streamlit as st
import numpy as np 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Next Word Prediction with LSTM",
    page_icon="\U0001f4dd",
    layout="wide"
)

st.markdown(
    """
    <style>
        .subtitle {
            color: var(--text-color, inherit);
            opacity: 0.85;
            margin-bottom: 1rem;
        }
        .result-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

## Load the trained model and tokenizer
model = load_model('C:\\Krish Naik (Data Science)\\Projects\\next_word_prediction_lstm_model.h5')

with open('C:\\Krish Naik (Data Science)\\Projects\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return None, None
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Keep only the last max_sequence_len-1 tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    predicted_word_index = int(np.argmax(predicted))
    confidence = float(predicted[predicted_word_index])
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word, confidence
    return None, confidence


def generate_sequence(model, tokenizer, seed_text, max_sequence_len, steps):
    generated_words = []
    running_text = seed_text
    confidences = []

    for _ in range(steps):
        next_word, confidence = predict_next_word(model, tokenizer, running_text, max_sequence_len)
        if not next_word:
            break
        generated_words.append(next_word)
        confidences.append(confidence if confidence is not None else 0.0)
        running_text = f"{running_text} {next_word}"

    return generated_words, confidences

## Streamlit app
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Next Word Prediction with LSTM")
st.markdown(
    '<div class="subtitle">Type a phrase and generate the next word or an entire continuation interactively.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Generation Controls")
    num_words = st.slider("Words to generate", min_value=1, max_value=10, value=3)
    show_confidence = st.toggle("Show confidence bars", value=True)
    st.divider()
    st.caption("Tip: Start with a phrase similar to your training corpus for better predictions.")

sample_prompt = st.selectbox(
    "Try a sample prompt",
    ["to be or not to", "the king", "shall i compare thee", "in the world"],
)

col1, col2 = st.columns([4, 1])
with col1:
    input_text = st.text_input("Enter a sequence of words", value=sample_prompt)
with col2:
    use_sample = st.button("Use Sample")

if use_sample:
    input_text = sample_prompt

max_sequence_len = model.input_shape[1] + 1

action_col1, action_col2 = st.columns(2)
with action_col1:
    predict_one = st.button("Predict Next Word", use_container_width=True)
with action_col2:
    generate_many = st.button("Generate Continuation", use_container_width=True)

if predict_one:
    next_word, confidence = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.markdown(
            f'<div class="result-card"><b>Predicted Next Word:</b> "{next_word}"</div>',
            unsafe_allow_html=True,
        )
        if show_confidence and confidence is not None:
            st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence:.2%}")
        st.session_state.history.append(
            {
                "input": input_text,
                "output": next_word,
                "confidence": confidence,
                "steps": 1,
            }
        )
    else:
        st.warning("Could not predict the next word. Please try a different input.")

if generate_many:
    generated_words, confidences = generate_sequence(
        model, tokenizer, input_text, max_sequence_len, num_words
    )
    if generated_words:
        generated_text = " ".join(generated_words)
        st.markdown(
            f'<div class="result-card"><b>Generated Continuation:</b> {generated_text}</div>',
            unsafe_allow_html=True,
        )
        st.code(f"{input_text} {generated_text}")
        if show_confidence:
            st.subheader("Word-by-word confidence")
            for word, conf in zip(generated_words, confidences):
                st.write(f"{word}")
                st.progress(min(max(conf, 0.0), 1.0), text=f"{conf:.2%}")
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        st.session_state.history.append(
            {
                "input": input_text,
                "output": generated_text,
                "confidence": avg_conf,
                "steps": len(generated_words),
            }
        )
    else:
        st.warning("Could not generate continuation. Try a longer or more familiar prompt.")

if st.session_state.history:
    st.divider()
    st.subheader("Recent Predictions")
    history_to_show = list(reversed(st.session_state.history[-5:]))
    for item in history_to_show:
        st.write(
            f'Input: "{item["input"]}" | Output: "{item["output"]}" | '
            f'Steps: {item["steps"]} | Confidence: {item["confidence"]:.2%}'
        )