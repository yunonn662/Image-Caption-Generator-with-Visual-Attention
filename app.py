from io import BytesIO
import streamlit as st
from streamlit_option_menu import option_menu
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pylab as plt
from tensorflow.keras.layers import (
    StringLookup
)

IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3
ATTENTION_DIM = 512
MAX_CAPTION_LEN = 32

def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(
        inputs, r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~]?", ""
    )

@st.cache_resource
def load_models(base_model_path):
    # Define the paths for the components of the model
    tokenizer_model_path = os.path.join(base_model_path, 'tokenizer_model')
    encoder_path = os.path.join(base_model_path, 'encoder')
    decoder_pred_model_path = os.path.join(base_model_path, 'decoder_pred_model')

    # Load the tokenizer model
    loaded_model = tf.keras.models.load_model(tokenizer_model_path, custom_objects={'standardize': standardize})
    tokenizer = loaded_model.layers[1]

    # Lookup tables
    word_to_index = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
    index_to_word = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

    # Load the encoder and decoder models
    encoder = tf.keras.models.load_model(encoder_path)
    decoder_pred_model = tf.keras.models.load_model(decoder_pred_model_path)

    return tokenizer, word_to_index, index_to_word, encoder, decoder_pred_model


def beam_search(img, tokenizer, word_to_index, index_to_word, encoder, decoder_pred_model, beam_width=3):

    # Extract features using the encoder
    features = encoder(tf.expand_dims(img, axis=0))

    # Start with the <start> token for each beam
    start_token = word_to_index("<start>")
    dec_input = tf.expand_dims([start_token], 0)
    gru_state = tf.zeros((1, ATTENTION_DIM))

    # Initial states of the beams
    beams = [(dec_input, gru_state, 0, [])]  # Each beam is a tuple (decoder_input, gru_state, score, caption_sequence)

    for i in range(MAX_CAPTION_LEN):
        new_beams = []
        for dec_input, gru_state, score, caption_sequence in beams:
            # Get predictions and update GRU state
            predictions, gru_state = decoder_pred_model([dec_input, gru_state, features])

            # Apply softmax to convert logits to probabilities
            predictions = tf.nn.softmax(predictions, axis=-1)

            # Get top k probabilities and their indices
            top_probs, top_idxs = tf.math.top_k(predictions, k=beam_width)

            # For each beam, add the top k possibilities
            for j in range(beam_width):
                next_word_idx = top_idxs[0][0][j].numpy()
                next_word = tokenizer.get_vocabulary()[next_word_idx]
                next_score = score + tf.math.log(top_probs[0][0][j])

                # If the next word is <end>, we complete the caption
                if next_word == "<end>":
                    return img, caption_sequence + [next_word]

                # Otherwise, we add the next word to the beam
                new_beam = (tf.expand_dims([next_word_idx], 0), gru_state, next_score, caption_sequence + [next_word])
                new_beams.append(new_beam)

        # Sort all possible beams by their score and select the top k to continue
        beams = sorted(new_beams, key=lambda beam: beam[2], reverse=True)[:beam_width]

    # If the loop ends because MAX_CAPTION_LEN is reached, return the highest scoring beam
    return img, max(beams, key=lambda beam: beam[2])[3]

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="icons8-camera-96.png",
    layout="wide",
)

# Load Models
tokenizer1, word_to_index1, index_to_word1, encoder1, decoder_pred_model1 = load_models('Final Models/Attention Flickr/Attention Flickr 30K EffNetV2B3 (5674) v4')
tokenizer2, word_to_index2, index_to_word2, encoder2, decoder_pred_model2 = load_models('Final Models/Additive Attention Flickr/Additive Attention Flickr 30K EffNetV2B3 (662) v10')
# Streamlit webpage layout
st.title("Image Caption Generator")

# Sidebar for navigation
with st.sidebar:
    selected_section=option_menu(
        menu_title="Section",
        options=["Generate Captions", "EDA", "Model Architecture", "Evaluations"],
        icons=["house", "graph-up", "motherboard","clipboard-data"],
        menu_icon="cast",
        default_index=0,
    )

# Generate Captions Section
if selected_section == "Generate Captions":
    selected_model=option_menu(
        menu_title=None,
        options=["Attention", "Additive Attention"],
        icons=["1-square-fill", "2-square-fill"],
        default_index=0,
        orientation="horizontal",
    )

    # Instructions and file uploader
    st.write("Upload an Image to Generate its Caption")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # User input for beam_width
    beam_width = st.number_input("Enter Beam Width", min_value=1, max_value=10, value=3, step=1)

    if selected_model == "Attention":
        tokenizer, word_to_index, index_to_word, encoder, decoder_pred_model = tokenizer1, word_to_index1, index_to_word1, encoder1, decoder_pred_model1
    else:
        tokenizer, word_to_index, index_to_word, encoder, decoder_pred_model = tokenizer2, word_to_index2, index_to_word2, encoder2, decoder_pred_model2

    if uploaded_file is not None:
        # Convert the uploaded file to bytes, then to an Image
        file_bytes = uploaded_file.getvalue()
        image_stream = BytesIO(file_bytes)
        image = Image.open(image_stream).convert('RGB')
        col1, col2, col3 = st.columns([2.5, 2, 2.5])
        with col2:
            st.image(file_bytes, caption="Uploaded Image", width=400)

        # Convert the PIL image to a format suitable for your model
        img_array = np.array(image)  # convert to numpy array
        img_array = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        # Generate caption
        image, caption = beam_search(img_array, tokenizer, word_to_index, index_to_word, encoder, decoder_pred_model, beam_width=beam_width)
        final_caption = ' '.join(caption[:-1]) + '.'
        st.markdown(
                f"""
                <style>
                    .caption {{
                        text-align: center;
                        padding: 10px;
                        border-radius: 10px;
                        font-size: 20px;
                        border: 1px solid #eaeaea;
                    }}
                    @media (prefers-color-scheme: dark) {{
                        .caption {{
                            color: white;
                        }}
                    }}
                    @media (prefers-color-scheme: light) {{
                        .caption {{
                            color: black;
                        }}
                    }}
                </style>
                <div class="caption">{final_caption}</div>
                """,
                unsafe_allow_html=True
            )
         # Show success message
        st.success("Caption generated successfully!",icon="✅")
        
# EDA Section
elif selected_section == "EDA":
    
    head1, head2, head3 = st.columns([1, 8, 1])
    with head2:
        st.header("Exploratory Data Analysis (EDA)")

    # Option menu for visualizations
    visualization = option_menu(
        menu_title="Select Visualization",
        options=["Caption Length Distribution", "Wordcloud of Captions", "Top 30 Most Common Words", 
                 "Cumulative Distribution of Word Frequencies", "Top 20 POS Tag Frequencies in Captions",
                 "Distribution of Sentiment in Captions", "Word Co-occurrence Heatmap", "LDA Visualization"],
        icons=["bar-chart", "cloud", "bar-chart-fill", "graph-up", "tags", "emoji-smile-fill", "border-all", "diagram-3-fill"],
        default_index=0,
        orientation="horizontal",
    )
    # Create columns for centering content
    g1, g2, g3 = st.columns([1, 3, 1])
    col1, col2, col3 = st.columns([1, 6, 1])  # Adjust the ratio as needed for better centering

    # Display selected visualization
    with g2:  # This column is used to center the content
        if visualization != "LDA Visualization":
            st.image(f'EDA Images/{visualization}.png', caption=visualization, use_column_width=True)
        else:
            with col2:
                st.write("LDA Visualization")
                HtmlFile = open('EDA Images/lda_visualization.html', 'r', encoding='utf-8')
                source_code = HtmlFile.read() 
                st.components.v1.html(source_code, height=780, width=1200)  # Adjust width and height as needed

# Model Architecture Section
elif selected_section == "Model Architecture":

    head1, head2, head3 = st.columns([1, 8, 1])

    with head2:
        st.header("Model Architecture")

    selected_model=option_menu(
        menu_title=None,
        options=["Attention", "Additive Attention"],
        icons=["1-square-fill", "2-square-fill"],
        default_index=0,
        orientation="horizontal",
    )

    # First row: encoder and decoder images side by side
    r1, r2, r3 = st.columns([2, 1, 2])
    with r1:
        st.image(f'Model Architecture/{selected_model}/encoder.png', caption='Encoder', use_column_width=True)
    with r3:
        st.image(f'Model Architecture/{selected_model}/decoder.png', caption='Decoder', use_column_width=True)
    
    # Second row: image caption train model centered
    st.write("")  # Spacer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(f'Model Architecture/{selected_model}/image_caption_train_model.png', caption='Encoder-Decoder Training Model', use_column_width=True)
    
    # Third row: decoder pred model centered
    st.write("")  # Spacer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(f'Model Architecture/{selected_model}/decoder_pred_model.png', caption='Prediction Model', use_column_width=True)

# Evaluations Section
elif selected_section == "Evaluations":

    head1, head2, head3 = st.columns([1, 8, 1])

    with head2:
        st.header("Evaluations")

    selected_model=option_menu(
        menu_title=None,
        options=["Attention", "Additive Attention"],
        icons=["1-square-fill", "2-square-fill"],
        default_index=0,
        orientation="horizontal",
    )

    # Define a function to create a centered image column
    def create_centered_image_col(image_path, caption):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_path, caption=caption, use_column_width=True)

    # Display images in a centered layout for the selected model
    if selected_model == "Attention":
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 1.png', 'Beam Search n=1')
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 2.png', 'Beam Search n=2')
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 3.png', 'Beam Search n=3')
    elif selected_model == "Additive Attention":
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 1.png', 'Beam Search n=1')
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 2.png', 'Beam Search n=2')
        create_centered_image_col(f'Evaluations/{selected_model}/Beam Search 3.png', 'Beam Search n=3')
