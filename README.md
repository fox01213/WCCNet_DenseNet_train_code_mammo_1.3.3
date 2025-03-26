# Getting Started
This README provides instructions on how to get started with WCCNet on DenseNet to make mammogram binary (Abnormal vs. Normal) classification by using the public mammogram FFDM dataset, such as INBreast and VinDr-Mammo, as the source of training data.

## Requirements
There are a few requirements needed in order to get started with this shared Python script:
* Install Anaconda, Python and PyTorch in your AI training environment
* Recommend adopting a single FFDM as training dataset in the very beginning. Then to incorporate multiple FFDM databases for generalization
* Convert the DICOM mammogram images in your selected FFDM database to 1024*1024 png images by leveraging medical grade image preprocessing techniques that are specially developed for mammogram
* Anonymize/block the private information on DICOM images if any
* Recommend to adopt patient-aware technique to split the train set of the selected FFDM database to Train and Validate split by either 80/20 or 85/15 ratio  
* WCCNet is a deep learning model designed to enhance classification tasks by addressing class imbalance through a weighted cross-entropy loss function. This allows the network to prioritize learning from underrepresented classes, which is particularly useful in medical imaging tasks like mammogram classification. Try first with an imbalanced dataset to evaluate the effectiveness of WCCNet for imbalanced dataset 
* Augment the scarcity Abnormal images later-on to evaluate the difference by also adopt medical grade, mammogram-specific data augmentation techniques.



## Version Evolution (Note that the shared version is version 1.3.3)

##### The ver1.2 is to first try the class weighting approach for WCCNet
-------------------------------------------------------
The ver1.2.1 is to:
1. Maintains training history when resuming
2. Saves optimizer state for consistent training continuation
3. Allows specifying new total epochs when resuming
4. Preserves all metrics and best model tracking
   
How to use:
1. Train from scratch: python WCCNet_DenseNet_train_code_mammo_1.3.2.py --epochs 100
2. Resume from checkpoint: python WCCNet_DenseNet_train_code_mammo_1.3.py --resume --epochs 100
3. Continue training from a pretrained model: python WCCNet_DenseNet_train_code_mammo_1.3.py --pretrained path/to/model.pth --epochs 100
---------------------------------------------------------
The ver1.2.2 is to: 
1. Fix torch.load issue
-----------------------------------------------------------
The ver1.3 is to: 
1. Add (1) aggressive FlocalLoss and (2) medical imaging-specific data augmentation modified to DenseNet
2. Yet to implement: (1) train dataset split for train and val, and (2) find_optimal_threshold
3. Still using manual Learn Rate
4. How to adjust FocalLoss poarameters:
    * gamma_pos = 0.1 (reduced from 0.2) Further reduces the focus on positive (Abnormal) samples
Makes the loss less sensitive to positive class errors
    * gamma_neg = 4.0 (increased from 2.0) Significantly increases focus on negative (Normal) samples
Makes the model pay more attention to misclassified normal cases
    * alpha = 0.25 (reduced from 0.4) Gives more weight to negative cases (1 - alpha = 0.75)
Helps balance the class importance
    * pos_weight = 0.5 (reduced from 1.0) Reduces the importance of positive class
Helps prevent overemphasis on abnormal cases
    * Added neg_weight = 2.0 (new parameter) Explicitly increases the importance of negative class
Helps the model focus more on normal cases
--------------------------------------------------------------
The ver1.3.1 is to:
1. Loss Function and Weighting Changes:
    * Increased positive class weight (pos_weight=3.0)
    * Modified focal loss parameters to focus more on positive cases
    * Adjusted alpha and gamma values for better sensitivity
2. Training Process Modifications:
    * Added learning rate scheduler based on sensitivity
    * Implemented early stopping based on sensitivity
    * Enhanced model saving criteria to focus on sensitivity 
----------------------------------------------------------------
The ver1.3.2 is to:
1. Added mixed precision training for memory efficiency
2. Optimized memory usage and GPU utilization
3. Enhanced gradient checkpointing
4. Improved data loading efficiency
5. Modified class weights and focal loss parameters
6. Added TF32 support for RTX 4090

How to use:
1. Train from scratch: python WCCNet_DenseNet_train_code_mammo_1.3.2.py --epochs 100
2. Resume from checkpoint: python WCCNet_DenseNet_train_code_mammo_1.3.2.py --resume --epochs 100
3. Continue training from a pretrained model: python WCCNet_DenseNet_train_code_mammo_1.3.2.py --pretrained path/to/model.pth --epochs 100
-----------------------------------------------------------------
The ver1.3.2.1 is to:
1. Added gradient clipping
2. adjust FocalLoss abd class weights
-----------------------------------------------------------------
The ver1.3.3 is to:
1. Enhanced FocalLoss Parameters:
    * Phase 1 (First 20 epochs):
        - gamma_pos=0.5 (more focus on all positive samples)
        - gamma_neg=3.0 (less focus on negative samples)
        - alpha=0.75 (heavy positive class weight)
        - pos_weight=3.0 (strong positive emphasis)
    * Phase 2 (After 20 epochs):
        - gamma_pos=1.0 (balanced positive focus)
        - gamma_neg=2.0 (moderate negative focus)
        - alpha=0.6 (moderate positive weight)
        - pos_weight=2.0 (maintained positive emphasis)
2. Added Curriculum Learning:
    * Gradually increases abnormal class weight from 2.0 to 3.5
    * Helps prevent sudden drops in sensitivity
3. Added Sensitivity Loss Component:
    * Additional loss term specifically targeting sensitivity
    * Different weights for each training phase
4. Dynamic Training Adjustments:
    * Monitors sensitivity drops
    * Automatically adjusts alpha and pos_weight if sensitivity decreases
    * Reduces learning rate when sensitivity drops
5. Add proper train/validation split
6. Implement advanced learning rate scheduling

## Adopted Training Techniques and Configurations
In order to cope with the vehement oscillation of Sensitivity and Specificity metrics during training, following training techniques with its associated hyperparameters were adopted to converge both lines in early stage. The following are the techniques and their tested hyperparameter values. Reader can further be adjusted for best result against your specific training dataset.     

![image](https://github.com/user-attachments/assets/4e56210b-d290-4071-b5fb-0fd42137caec)



## Training Result
Having carefully implemented the shared script by using well-prepared training, validate and test dataset, and hyperparameters fine-turning. Readers might get training results like the one as attached below.

![image](https://github.com/user-attachments/assets/9b1a0553-9bbc-423e-aa4d-9296873f7d95)

