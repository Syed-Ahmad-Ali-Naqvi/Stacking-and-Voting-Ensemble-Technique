﻿# Stacking-and-Voting-Ensemble-Technique

Classfication of images using two ensemble techniques Voting and Stacking using three base models Restnet50, InceptionV3 and Alexnet

ResNet50
ResNet50 is a deep convolutional neural network architecture that is 50 layers deep. It is a variant of the original Residual Network (ResNet) architecture, which was introduced by Microsoft Research in 2015. The key innovation of ResNet is the introduction of residual connections or skip connections, which help mitigate the vanishing gradient problem in deep networks by allowing the gradient to flow directly through the network's layers. ResNet50 is widely used in image classification tasks and is known for its efficiency and high performance on large-scale datasets like ImageNet.

InceptionV3
InceptionV3 is a deep convolutional neural network architecture that is part of the Inception series developed by Google. It was introduced in 2015 and builds upon the concepts of the earlier Inception networks (also known as GoogLeNet). InceptionV3 incorporates several improvements, such as factorized convolutions, label smoothing, and the use of auxiliary classifiers, to enhance its performance. The architecture is known for its use of Inception modules, which allow for the efficient computation of different convolutional operations within the same network, leading to a rich hierarchical representation of features. InceptionV3 is commonly used for image classification and transfer learning.

AlexNet
AlexNet is a pioneering deep convolutional neural network that significantly contributed to the resurgence of deep learning in the field of computer vision. Developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, AlexNet won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 by a large margin, showcasing the potential of deep learning. The architecture consists of 8 layers (5 convolutional and 3 fully connected), and it introduced innovations like ReLU activation functions, dropout regularization, and data augmentation to prevent overfitting. AlexNet was one of the first models to demonstrate the effectiveness of GPUs for training deep networks.

Stacking Ensemble Technique
Stacking (or Stacked Generalization) is an ensemble learning technique that combines multiple machine learning models to improve predictive performance. In stacking, individual models (often called base models or level-0 models) are first trained on the training data. The predictions of these base models are then used as input features for a meta-model (or level-1 model), which is trained to make the final predictions. The meta-model typically captures patterns that the base models might miss, leading to more robust and accurate predictions. Stacking is powerful because it leverages the strengths of different models and mitigates their individual weaknesses.

Voting (Soft) Ensemble Technique
Voting (Soft) Ensemble is an ensemble learning method where multiple models (often of different types) are trained independently on the same dataset. In the soft voting variant, each model outputs a probability distribution over classes, rather than a single class label. These probability distributions are then averaged (or weighted and averaged) across all models, and the class with the highest average probability is selected as the final prediction. Soft voting is particularly effective when the individual models have high confidence in different classes, as it considers the overall probability landscape, leading to more balanced and accurate predictions.
