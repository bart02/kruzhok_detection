# kruzhok_detection
KD logo detection for KRUZHOK.PRO selection

## Data augmentation
To train the network better on a small dataset, you can use data augmentation, a technique for modifying the images in the training set and adding them to the set. Thus, the network will receive more different examples for training and, accordingly, will show better results.


Data_Augmentation.ipynb can help you with it. It can:
- make a lot of modified images
- randomly rotate images
- randomly shift images horizontally and vertically
- randomly zoom images
- randomly invert colors in images


To augment your images you need:
1. Replace images in Dataset/Dataset with yours
2. Open Data_Augmentation.ipynb
3. Tune size and number of augmented images (optionally you can tune augmentation parameters, for example, zoom, shift or rotation)
4. Run all code in file
5. Finally, you will see augmented dataset in directory "AugmentedDataset" in same directory with script


It uses:
- Python
- TensorFlow library
- os library