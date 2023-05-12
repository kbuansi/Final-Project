# Final-Project

The methodology behind the code for the CT Covid Pneumonia scan to run is what follows. This can be summarized as follows. The first step is reading and preprocessing the images. There are about 7500 images and 6800 normal or non-covid images. We chose to select 400 images from the two groups and resized them down, since the original dimensions took up a lot of RAM capacity, giving us issues when running the code. We also appended them as well as creating an array for labeling the images efficiently. Looking back, we could have probably increased the image size had we used a stronger computer to run the code.

We output random images of a normal ct scan and a covid ct scan. You can notice the visual differences between the two

We output random images of a normal ct scan and a covid ct scan. You can notice the visual differences between the two

We then split the train dataset to train and validation, and created a custom vgg16 model as we used a similar methodology in one of the previous assignments. We integrated early stopping to monitor for overfitting, and model checkpoint. We also created a means of finding the most effective amount of epochs using history.history[] to look back at which one had the best value.

Finally, we output the model accuracy and loss, to use for comparison.

We started by importing the necessary libraries and defining constants, and then added a directory for the dataset. We opted to use the validation images for visualization due to their limited quantity, which speeds up the process, and just like the ct scan, we previewed the images. 

Because of the dataset having a lot more covid pneumonia positive cases, we augmented the training data to provide more images for our model to use for training, and attempted to prevent overfitting as best as we could. We created another preview of images to check for distortion (there wasn’t much). For training we used a sequential CNN model, and we ran 50 epochs for it.


We then graphed for training accuracy, validation accuracy, as well as training and validation loss. 

For the Multimodal CNN, the ‘create_multi_modal_cnn’ function defines the architecture of the multi-modal CNN. It takes the input shapes for X-ray and CT images, as well as the number of classes, as parameters. The function constructs separate branches for processing X-ray and CT inputs using convolutional and pooling layers. The outputs of both branches are then concatenated and fed into dense layers for classification. We graphed the validation accuracy and loss for it.


