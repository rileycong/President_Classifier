# How to send pictures from UI to back-end: 
# Use base64 encoded string 
import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

# define 2 variables for model running
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(image_base64_data, file_path=None): # if want to pass the file in directly use this variable
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)   # final is a 1-dim array for 1 picture

        result.append({
            'class': convert_number_to_name(__model.predict(final)[0]),    # return the first result
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0],    # return the similarity between faces, round to 2 de places, turn to list
            'class_dictionary': __class_name_to_number
        })
        
    return result

def load_saved_artifacts():
    print("loading saved arifacts...start")
    global __class_name_to_number 
    global __class_number_to_name

    # president name to number and number to name
    with open(r"D:\Python code\US_President_Classifier\Server\Artifacts\class_dictionary.json","r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    
    # import the model
    global __model
    if __model is None:
        with open(r"D:\Python code\US_President_Classifier\Server\Artifacts\saved_model.pkl", "rb") as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def convert_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(r'D:\Python code\US_President_Classifier\Server\opencv\opencv\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r'D:\Python code\US_President_Classifier\Server\opencv\opencv\haarcascades\haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

# get the b64 string
def get_b64_test_image():
    with open(r"D:\Python code\US_President_Classifier\Server\b64.txt") as f:
        return f.read()

if __name__ == "__main__":
    load_saved_artifacts()
    #print(classify_image(get_b64_test_image(), None))
    # print(classify_image(None, r"D:\Python code\US_President_Classifier\Server\Test_image\Barack Obama.jpg"))
    # print(classify_image(None, r"D:\Python code\US_President_Classifier\Server\Test_image\Bill Clinton.jpg"))
    # print(classify_image(None, r"D:\Python code\US_President_Classifier\Server\Test_image\Bush.jpg"))
    # print(classify_image(None, r"D:\Python code\US_President_Classifier\Server\Test_image\Donald Trump.jpg"))
    print(classify_image(None, r"D:\Python code\US_President_Classifier\Server\Test_image\Obama.jpg"))



