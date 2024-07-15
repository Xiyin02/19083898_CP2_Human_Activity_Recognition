import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import argparse
random.seed(8883)
warnings.filterwarnings('ignore')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Simple example of argparse')
    parser.add_argument('--mode', type=str, help='validation/test mode')
    parser.add_argument('--model', type=str, help='model directory/huggingface model name')
    args = parser.parse_args()
    
    mode = args.mode
    df = pd.read_csv(f'{mode}.csv')
    print(f'{mode} Set Size : ' + str(df.shape[0]))
    
    weight_dir = args.model
    # weight_dir = 'Xiyin02/CP2_HAR_MobileNet'
    # initialize the model
    image_processor = AutoImageProcessor.from_pretrained(weight_dir)
    model = AutoModelForImageClassification.from_pretrained(weight_dir)
    correct = 0
    gts = []
    preds = []
    for i in tqdm(df.values):
        # read image 
        image = Image.open(i[0])
        # ground truth
        gt = i[1]
        gts.append(gt)
        # convert into encodings
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
        
        # forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        
        # prediction
        predicted_class_idx = logits.argmax(-1).item()
        pred_class = model.config.id2label[predicted_class_idx]
        if pred_class.lower() == gt:
            correct += 1
        preds.append(pred_class.lower())
    
    # Sample true labels and predicted labels
    true_labels = gts
    predicted_labels = preds
    n_classes = len(np.unique(true_labels))
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    # Print accuracy, F1 score, precision, and recall
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{mode}.png')
    plt.show()
