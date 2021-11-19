import streamlit as st
from keras.applications.vgg16 import preprocess_input
#from pillow import Image, ImageOps

import keras
from PIL import Image, ImageOps
import numpy as np
import pickle
from scipy import misc
import urllib.request
with open('breeds.pk', 'rb') as f:
	breeds = pickle.load(f)
#st.title("Image Classification Witt VGG16 transfer learning")
#st.header("Dog race imaage classification example")
#st.text("Upload an dog image for image classification ")

def main():
    # Render the readme as markdown using st.markdown.
	readme_text = st.markdown("# \n")
	st.title("Image Classification Witt VGG16 transfer learning")
	st.header("Dog race imaage classification example")
	st.text("Upload an dog image for image classification ")
	#intro_markdown = read_markdown_file("intructions for tha app.md")
    #st.markdown(intro_markdown, unsafe_allow_html=True)
	url = 'https://drive.google.com/file/d/1DKFnygreq69uPJTv882JSNUn30jOD_QL/view?usp=sharing'
	output = 'model.h5'
	gdown.download(url, output, quiet=False)
	#urllib.request.urlretrieve(
        #'https://raw.github.com/aliardouz0/MMMM/main/tl_best_model__xx8_prpr.h5', 'model.h5')
	#MODEL_PATH = './model.h5'
	#sw = keras.models.load_model(MODEL_PATH)
	#sw = keras.models.load_model(urlopen("https://raw.github.com/aliardouz0/MMMM/main/tl_best_model__xx8_prpr.h5"))
	#sw = keras.models.load_model('./tl_best_model__xx8_prpr.h5')
    #clf = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/clf.pk"))
    #tfidf_2 = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/tfidf_2.pk"))
    #F_tags = pickle.load(urlopen("https://raw.github.com/aliardouz0/IML-P05/main/F_tags.pk"))
    # Download external dependencies.
    #for filename in EXTERNAL_DEPENDENCIES.keys():
     #   download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
	st.sidebar.title("What to do")
	app_mode = st.sidebar.selectbox("Choose the app mode",
		["Show instructions", "Run the app", "Show the source code"])
	if app_mode == "Show instructions":
		st.sidebar.success('To continue select "Run the app".')
	elif app_mode == "Show the source code":

		readme_text.empty()
		st.code(get_file_content_as_string("app_first_try.py"))
	elif app_mode == "Run the app":
		readme_text.empty() 
		run_the_app()
        

def run_the_app():
	uploaded_file = st.file_uploader("Choose a image ...", type="jpg")
	if uploaded_file is not None:
		image = Image.open(uploaded_file)
		#arr = np.array(image) 
		st.image(image, caption='Uploaded image.', use_column_width=True)
		st.write("")
		st.write("Classifying...")
		label = teachable_machine_classification(image, 'tl_best_model__xx8_prpr.h5')
		st.write(label)
		
def teachable_machine_classification(img, weights_file):
	# Load the model
	model = keras.models.load_model('tl_best_model__xx8_prpr.h5')
	# Create the array of the right shape to feed into the keras model
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	image = img
	#image sizing
	size = (224, 224)
	image = ImageOps.fit(image, size)

	#turn the image into a numpy array
	image_array = np.asarray(image)
	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = image_array
	# run the inference
	prediction = model.predict(preprocess_input(data))
	breed = breeds[np.argmax(prediction)]
	return breed # return position of the highest probability
	
	
def get_file_content_as_string(path):
    url = 'https://github.com/aliardouz0/IML/blob/29f080131c22971519363acb79676a766881c9ff/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")
            
def read_markdown_file(markdown_file):
	return Path(markdown_file).read_text()

if __name__ == "__main__":
    main()
