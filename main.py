import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np
from skimage.feature import corner_fast

IMG_SIZE = 500

class_list = {'0': 'Canh', '1': 'Canh'}

st.title('Predict Human')

input = open('hackathon.pkl', 'rb')
model = pkl.load(input)

st.header('Upload your image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

def feature_fast(img):
  fast_image = corner_fast(img, n=8, threshold=0)
  return fast_image
    
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image')

    if st.button('Predict'):
        image = image.resize((IMG_SIZE*IMG_SIZE*3, 1))
        feature_vector = np.array(feature_fast(image[:,:,0])).reshape(1, 500*500)
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
