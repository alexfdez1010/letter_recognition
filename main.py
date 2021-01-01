from train import train_model
from tensorflow.keras.models import load_model
import gradio,os,numpy

if __name__ == "__main__":
    
    if "letter_recognition.h5" in os.listdir():
        model = load_model("letter_recognition.h5")
    else:
        model = train_model()
    
    def predict_image(img):
        img = numpy.expand_dims(img,axis = -1)
        img = numpy.expand_dims(img,axis = 0)
        prediction = model.predict(img).tolist()[0]
        return {chr(i+65) : prediction[i] for i in range(26)}
    
    label = gradio.outputs.Label(num_top_classes= 5)
    gradio.Interface(fn = predict_image, inputs = "sketchpad", outputs = label).launch()
        
    
    
    