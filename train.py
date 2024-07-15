import pandas as  pd
from datasets import load_dataset, Features, Image, ClassLabel
from transformers import AutoImageProcessor,AutoModelForImageClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np
from torchvision.transforms import Compose,Normalize,RandomHorizontalFlip,RandomResizedCrop,Resize,ToTensor
import torch
import argparse
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple example of argparse')
    parser.add_argument('--pretrained_model', type=str, help='choose pretrained model')
    args = parser.parse_args()
    
    #get all label_names
    df = pd.read_csv('train.csv')
    label_names = df.label.unique().tolist()
    
    # Define the path to the CSV file
    csv_file = {"train":"train.csv",'val':"val.csv"}
    
    # Define the features of the dataset
    features = Features({
        'filename': Image(),  # Define the image column
        'label': ClassLabel(names=label_names)  # Define the label column
    })
    
    # Load the dataset
    dataset = load_dataset('csv', data_files=csv_file, features=features)
    
    # pretrained model
    model_name = args.pretrained_model #google/vit-base-patch16-224 or google/mobilenet_v2_1.4_224
    
    image_processor  = AutoImageProcessor.from_pretrained(model_name)
    normalize = Normalize(mean=image_processor.image_mean,
                          std=image_processor.image_std)
    
    if 'mobile' in model_name:
    image_size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    else:
        image_size = (image_processor.size["height"], image_processor.size["width"])
        
    train_transforms = Compose([RandomResizedCrop(image_size),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                normalize])
    
    val_transforms = Compose([Resize(image_size),
                              ToTensor(),
                              normalize])
    
    
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(label_names):
        label2id[label] = i
        id2label[i] = label
        
        
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)
    
    
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True,
    )
    
    
    
    metric = load("accuracy")
    hf_model_name = "Human-Action-Recognition-VIT-Base-patch16-224"
    
    args = TrainingArguments(
        hf_model_name,
        remove_unused_columns=False,
        num_train_epochs=20,
        # max_steps=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        evaluation_strategy = "epoch",
        logging_strategy="epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    
    train_results = trainer.train()
    
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)