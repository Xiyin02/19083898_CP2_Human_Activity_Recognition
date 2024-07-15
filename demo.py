import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load the model and processor
    weight_dir = 'Xiyin02/CP2_HAR_ViT_Base_16_224'
    image_processor = AutoImageProcessor.from_pretrained(weight_dir)
    model = AutoModelForImageClassification.from_pretrained(weight_dir)
    
    def predict(image):
        # Convert the image to RGB and preprocess it
        print(image)
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
    
        # Forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
    
        # Get the predicted class index
        predicted_class_idx = logits.argmax(-1).item()
        prediction = model.config.id2label[predicted_class_idx]
        
        return prediction
    
    # Gradio interface
    image_input = gr.Image(type="pil", label="Choose an image")
    output_text = gr.Textbox(label="Predicted Class")
    
    iface = gr.Interface(
        fn=predict,
        inputs=image_input,
        outputs=output_text,
        title="Human Activity Recognition",
        description="Upload an image and get the predicted class."
    )
    
    iface.launch()
