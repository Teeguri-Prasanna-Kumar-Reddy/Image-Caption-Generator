import streamlit as st
import nltk
from PIL import Image
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

# Load the VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Load features from pickle
features = pickle.load(open("features.pkl", "rb"))

# Load captions
with open("captions.txt", "r",encoding='utf-8') as f:
    next(f)
    captions_doc = f.read()

# Create mapping of image to captions
mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split('|')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[2:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Clean captions
for key, captions in mapping.items():
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.lower()
        caption = caption.replace('[^A-Za-z]', '')
        caption = caption.replace('\s+', ' ')
        caption = '<start> ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' <end>'
        captions[i] = caption

# Tokenize the text
all_captions = [caption for key in mapping for caption in mapping[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

# Text vectorization layer
text_dataset = tf.data.Dataset.from_tensor_slices(all_captions)
max_features = vocab_size
max_len = max_length
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len
)
vectorize_layer.adapt(text_dataset.batch(64))

# Create the model that uses the vectorize text layer
model_vect = tf.keras.models.Sequential()
model_vect.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model_vect.add(vectorize_layer)

# Encoder model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Load the pre-trained weights
caption_model.load_weights("best_model.h5")

# Function to predict captions
def predict_caption(image, tokenizer, max_length):
    in_text = '<start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ind_to_word(yhat, tokenizer)
        if word is None or word == 'end':
            break
        in_text += " " + word
    return in_text

# Function to convert index to word
def ind_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Streamlit app
st.title("Image Caption Generator")

# Upload image through the Streamlit interface
image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Extract features
    img_features = model.predict(img_array)

    # Generate caption
    generated_caption = predict_caption(img_features, tokenizer, max_length)

    generated_caption=generated_caption[7:]

    # Display the generated caption
    st.subheader("Generated Caption:")
    st.write(generated_caption)

# Add any additional features or controls as needed
