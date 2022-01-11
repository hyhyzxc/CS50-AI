AI that can identify which traffic sign appears in a photograph after giving it photos of traffic signs to analyse and learn from

-OpenCV and TensorFlow libraries used to create a neural network for machine learning

Started off with trying the values in lecture notes. Then followed on to reduce dropout rate to 0.2 as 0.5 seems to be quite high. Used an extra convolutionary layer and changed kernel size to be 3x3 which seemed to give the best outcome. Pool-size of 2 by 2 seems to give the best outcome. Lastly, activation function 'sigmoid' is used for first cnn layer which gave a higher accuracy.

Using any other pool-size decreased accuracy significantly. My first try had accuracy of 0.06~ . Using only one hidden layer seems to give best outcome and with "sigmoid" activation instead of "relu" activation.
