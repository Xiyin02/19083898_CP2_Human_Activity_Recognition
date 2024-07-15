# Human Activity Recognition Captopne Project Sunway University
## Step 1
download dataset from website and unzip it into working directory  

    ./download_dataset.sh

## Step 2
create an environment for development  
    
    python -m virtualenv HAR_env  
    HAR_env/Scripts/activate  
    python -m pip install pandas numpy matplotlib opencv-python pillow torch torchvision transformers evaluate seaborn scikit-learn tqdm gradio

## Step 3
create dataset from downloaded dataset for training  

    python create_dataset.py

## Step 4
run the training script  

### Train MobileNet
    python train.py --pretrained_model "google/mobilenet_v2_1.4_224"  

### Train ViT 
    python train.py --pretrained_model "google/vit-base-patch16-224"  

## Step 5
visualise and generate the plots for results comparison between 2 models  
    
    python training_results_visualisation.py  

## Run evaluation on the models
### MobileNet + validation set
    python evaluate.py --mode val --model "Xiyin02/CP2_HAR_MobileNet"  

### MobileNet + test set
    python evaluate.py --mode test --model "Xiyin02/CP2_HAR_MobileNet"  

### ViT + validation set
    python evaluate.py --mode val --model "Xiyin02/CP2_HAR_ViT_Base_16_224"  

### ViT + test set
    python evaluate.py --mode test --model "Xiyin02/CP2_HAR_ViT_Base_16_224"  

## Run live demo / inference with Gradio  
    python demo.py  
