import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Input

# Load the trained model
model = load_model('chatbot_model.keras')

# Load the tokenizer
with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Define constants
max_length = model.input_shape[0][1]
vocab_size = len(tokenizer.word_index) + 1

# Split the model into encoder and decoder
encoder_inputs = model.input[0]  # encoder input
encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # encoder output and states
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # decoder input
decoder_state_input_h = Input(shape=(256,), name='input_3')  # Adjust to your LSTM units
decoder_state_input_c = Input(shape=(256,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding = model.layers[3](decoder_inputs)  # Use the same embedding layer
decoder_lstm = model.layers[5]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Function to preprocess input sentence
def preprocess_input(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

# Function to generate a response
def generate_response(input_sequence):
    # Encode the input sequence to get the states
    states_value = encoder_model.predict(input_sequence)

    # Initialize the decoder input sequence with the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample the next token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer.index_word.get(sampled_token_index, '')

        # Append the token to the response
        decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the end token
        if sampled_token == '<end>' or len(decoded_sentence.split()) >= max_length:
            stop_condition = True

        # Update the target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.replace('<end>', '').strip()

# Interactive loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    input_sequence = preprocess_input(user_input)
    response = generate_response(input_sequence)
    print("Bot:", response)
