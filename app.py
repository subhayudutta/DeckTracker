import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('card_pred.h5')

st.text("Designed by Subhayu Dutta 2023")
navbar = """
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #8EE4AF;
            padding: 20px;
            margin-bottom: 20px;
        }
        .navbar-title {
            font-size: 24px;
            font-weight: bold;
            margin-right: 10px;
        }
        .navbar-logo {
            height: 40px;
        }
    </style>
    
    <div class="navbar">
        <div class="navbar-title">DeckTracker</div>
        <img class="navbar-logo" src="https://i.pinimg.com/originals/00/49/9b/00499be3aa3343afd5ea5195f1f206b5.jpg" alt="Logo">
    </div>
    """
st.markdown(navbar, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an card image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_path = "temp_image.jpg"

    img = image.load_img(format(file_path), target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)

    if preds==0:
        preds="ace of clubs"
    elif preds==1:
        preds="ace of diamonds"
    elif preds==2:
        preds="ace of hearts"
    elif preds==3:
        preds="ace of spades"
    elif preds==4:
        preds="eight of clubs"    
    elif preds==5:
        preds="eight of diamonds"
    elif preds==6:
        preds="eight of hearts"
    elif preds==7:
        preds="eight of spades"
    elif preds==8:
        preds="five of clubs"
    elif preds==9:
        preds="five of diamonds"
    elif preds==10:
        preds="five of hearts"
    elif preds==11:
        preds="five of spades"
    elif preds==12:
        preds="four of clubs"
    elif preds==13:
        preds="four of diamonds"
    elif preds==14:
        preds="four of hearts"
    elif preds==15:
        preds="four of spades"
    elif preds==16:
        preds="jack of clubs"
    elif preds==17:
        preds="jack of diamonds"
    elif preds==18:
        preds="jack of hearts"
    elif preds==19:
        preds="jack of spades"
    elif preds==20:
        preds='Joker'
    elif preds==21:
        preds="king of clubs"
    elif preds==22:
        preds="king of diamonds"
    elif preds==23:
        preds="king of hearts"
    elif preds==24:
        preds="king of spades"
    elif preds==25:
        preds="nine of clubs"
    elif preds==26:
        preds="nine of diamonds"
    elif preds==27:
        preds="nine of hearts"
    elif preds==28:
        preds="nine of spades"
    elif preds==29:
        preds="queen of clubs"
    elif preds==30:
        preds="queen of diamonds"
    elif preds==31:
        preds="queen of hearts"
    elif preds==32:
        preds="queen of spades"
    elif preds==33:
        preds="seven of clubs"
    elif preds==34:
        preds="seven of diamonds"
    elif preds==35:
        preds="seven of hearts"
    elif preds==36:
        preds="seven of spades"
    elif preds==37:
        preds="six of clubs"
    elif preds==38:
        preds="six of diamonds"
    elif preds==39:
        preds="six of hearts"
    elif preds==40:
        preds="six of spades"
    elif preds==41:
        preds="ten of clubs"
    elif preds==42:
        preds="ten of diamonds"
    elif preds==43:
        preds="ten of hearts"
    elif preds==44:
        preds="ten of spades"
    elif preds==45:
        preds="three of clubs"
    elif preds==46:
        preds="three of diamonds"
    elif preds==47:
        preds="three of hearts"
    elif preds==48:
        preds="three of spades"
    elif preds==49:
        preds="two of clubs"
    elif preds==50:
        preds="two of diamonds"
    elif preds==51:
        preds="two of hearts"
    else:
        preds="two of spades"
    
    ans="The predicted card image is of "+preds
    st.info(ans)
    st.image(uploaded_file, caption=preds, use_column_width=True)
