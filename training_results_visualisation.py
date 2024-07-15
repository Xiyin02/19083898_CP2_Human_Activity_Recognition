import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    epochs = range(1, len(train_loss) + 1)
    
    name = 'CP2_HAR_mobilenet_v2_1.4_224/'
    train = json.load(open(f'{name}/trainer_state.json','r'))
    log_history= train['log_history']
    mobile_train_loss = [log["loss"] for log in log_history if "loss" in log]
    mobile_eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    mobile_acc= [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]
    
    
    name = 'CP2_HAR_vit-base-patch16-224/'
    train = json.load(open(f'{name}/trainer_state.json','r'))
    log_history= train['log_history']
    vit_train_loss = [log["loss"] for log in log_history if "loss" in log]
    vit_eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    vit_acc= [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mobile_train_loss, label='Training Loss')
    plt.plot(epochs, mobile_eval_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'MobileNet_loss_comparison.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, vit_train_loss, label='Training Loss')
    plt.plot(epochs, vit_eval_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'ViT-Base-16-224_loss_comparison.png')
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mobile_acc, label='MobileNet Validation Accuracy')
    plt.plot(epochs, vit_acc, label='ViT-Base-16-224 Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'models_accuracy_comparison.png')
    
    
    print('Plots Generated!')