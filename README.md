# Next Word Prediction using LSTM

A Deep Learning Natural Language Processing project that predicts the **next word in a sentence** using an **LSTM (Long Short-Term Memory) neural network** trained on Shakespeare's *Hamlet* corpus.

The project demonstrates the complete **end-to-end NLP pipeline**, including text preprocessing, sequence generation, LSTM model training, and deployment through a **Streamlit web application**.


---

## Key Features

- LSTM based **language model**
- Trained on **Shakespeare Hamlet dataset**
- Predicts the **next most probable word**
- Generates **multiple word continuations**
- Displays **prediction confidence scores**
- Interactive **Streamlit web interface**
- Maintains **recent prediction history**

---

## Project Workflow

- Raw Text Dataset
↓
- Text Cleaning & Tokenization
↓
- N-gram Sequence Generation
↓
- Sequence Padding
↓
- LSTM Model Training
↓
- Softmax Prediction
↓
- Next Word Generation

---


---

## Dataset

The dataset is obtained from the **NLTK Gutenberg Corpus**.

Corpus used:

Shakespeare Hamlet

Example sentence:

----


These sequences help the model learn **contextual word prediction**.

---

## Model Architecture

The model is implemented using **TensorFlow / Keras**.

- Embedding Layer
↓
- LSTM Layer
↓
- Dense Layer
↓
- Softmax Output Layer

---


### Model Configuration

- Loss Function - categorical_crossentropy
- Optimizer - adam
- Output Layer - Softmax


---

## Tech Stack

### Programming Language
- Python

### Libraries
- TensorFlow
- Keras
- Numpy
- NLTK
- Streamlit
- Pickle

### Tools
- Jupyter Notebook
- GitHub
- Streamlit

---

## Project Structure

- next-word-prediction-lstm

- ├── Next Word Prediction Using LSTM.ipynb
   │ Model training notebook
│
- ├── app.py
    │ Streamlit web interface
│
- ├── tokenizer.pickle
       │ Saved tokenizer
│
- ├── next_word_prediction_lstm_model.h5
│ Trained LSTM model
│
- ├── requirements.txt
    │ Project dependencies
│
- └── README.md
