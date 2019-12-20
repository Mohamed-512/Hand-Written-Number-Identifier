# Hand-Written-Number-Identifier
This is a Python/Keras solution that identifies hand written digits using digital cameras such as normal computers webcams and the Raspberry Pi Camera. It was trained on MNIST data set and achieved more than 98% accuracy.

## Running
Run any camera test file on the proper environment, by setting the model to be used depending on what you want.

## Pretrained models
A model was trained with 1 epoch with the settings in """Main Keras.py""" on all MNIST images and it scored more than 98% accuracy.

## Notes
* If tested in bad lighting, misclassifications can occur. To solve tweak the offset in split_colors() function in the ImageProcessor.py file.
