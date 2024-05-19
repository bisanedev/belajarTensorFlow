import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Load dataset
with open('chatbot_data.json') as f:
    data = json.load(f)

# Ensure the data includes <start> and <end> tokens
questions = [item['question'] for item in data]
answers = ['<start> ' + item['answer'] + ' <end>' for item in data]

# Initialize the tokenizer and add special tokens
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(['<start>', '<end>'] + questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert texts to sequences
sequences_questions = tokenizer.texts_to_sequences(questions)
sequences_answers = tokenizer.texts_to_sequences(answers)

# Padding sequences
max_length = max(max(len(seq) for seq in sequences_questions), max(len(seq) for seq in sequences_answers))
padded_questions = pad_sequences(sequences_questions, maxlen=max_length, padding='post')
padded_answers = pad_sequences(sequences_answers, maxlen=max_length, padding='post')

# Create decoder input data by shifting the answers for teacher forcing
decoder_input_data = np.zeros_like(padded_answers)
decoder_input_data[:, 1:] = padded_answers[:, :-1]
decoder_input_data[:, 0] = tokenizer.word_index['<start>']  # '<start>' token should now exist

# Split dataset (consistent splitting for all arrays)
X_train, X_val, y_train, y_val, decoder_train, decoder_val = train_test_split(
    padded_questions, padded_answers, decoder_input_data, test_size=0.2
)

# Check shapes for consistency
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("decoder_input_data shape:", decoder_input_data.shape)
print("decoder_train shape:", decoder_train.shape)

# Encoder
encoder_inputs = Input(shape=(max_length,))
encoder_embedding = Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_length,))
decoder_embedding = Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train, decoder_train], np.expand_dims(y_train, -1), 
          batch_size=64, epochs=100, 
          validation_data=([X_val, decoder_val], np.expand_dims(y_val, -1)))

# Save the model
model.save('chatbot_model.keras')

# Save the tokenizer
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
