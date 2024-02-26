## Brain Tumor Multiclass Classification :brain: - Computer Vision Model using a Convolutional Neuronal Network

This project focuses on creating a convolutional neural network designed to classify MRI scans into three categories:

***Gliomas, meningiomas and pituitary tumors.***

The purpose of this model is to assist neuroradiologists in diagnosing these brain tumor types more efficiently, potentially speeding up the diagnostic process. However, it's important to emphasize that this model should complement human expertise rather than replace it entirely.

## Prerequisites 
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- Jupyter Notebook ```pip install notebook ```
- [Required dependencies](.......) ```pip install -r requirements.txt ```
- [Brain Tumor Image Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor/data) (2)

## Data Set & Preprocessing
Total number of MRI scans: 3064
Glioma images: 1426 (46.5%), Meningioma images: 708 (23.1%), Pituitary tumor images: 930 (30.4%)
The data set was split into a **training set** (80%, 2451 records), a **validation set** (10%, 306 records) and a **test set** (10%, 307 records).

## Pre-trained Base ResNet50  
For this task, a Convolutional Neural Network (CNN) known as ResNet50 was employed as the pre-trained base. It was compared against other CNN models like VGG16 and VGG19, showing significantly superior performance. 
ResNet50 has 50 layers and makes use of a technique called residual learning. This involves using shortcuts/skip connections between layers to learn adjustments to the input data rather than attempting to directly learn the entire transformation. This approach makes the network very effective.

## Strategies to Diminish Overfitting
### Dropout
Randomly selected neurons are dropped out or ignored with a certain probability. This encourages generalization as the network does not rely too heavily on individual neurons.

### L1-Regularization
L1/Lasso regularization, adds a 'penalty' to the absolute values of the weights to the loss. In this way, the neuronal network shows an increasing sparsity by driving some weights to zero, improving the capability to generalize.

### L2-Regularization
L2/Ridge regularization adds a 'penalty' to the squared values of the weights to the loss, thereby encouraging smaller weight values and reducing the impact of individual features.

### Early Stopping
During training, the model's performance on the validation set is monitored. The training is stopped and the weights are restored when the performance on the validation set diminishes. This prevents the model from learning noise in the training data (4).

## Performance Measurements
Several performance valuations were conducted to gain a detailed perspective on the predictions made with the ***test data***:


:heavy_exclamation_mark: <img width="453" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/61b24849-5012-436f-a80e-ce089fbef9d2"> 

### Accuracy
Accuracy is a good performance parameter when all classes are equally important and there are no significant class imbalances. In this project, classes were not well-balanced.

<img width="161" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/7417c4b4-09d8-4dba-bb11-8e9e9dbebc1e">

### Precision
Precision is good when minimizing false positive predictions is important.

<img width="119" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/af1d55dd-f7a5-4633-95bf-eb81504beeb7">

### Recall
Recall is a good choice when minimizing false negative predictions is crucial.

<img width="99" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/997983e0-fab6-449d-a330-bf6c128055a6">

### F1-Score 
F1-Score is suitable for situations where both false positives and false negatives are important to consider and when precision and recall are balanced (4).

<img width="506" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/0efc1f75-eccb-4670-ace2-637573984049">

### Confusion Matrix
Matrix of actual vs predicted classes for every instance. Each cell contains a count of how many instances were classifies into a particular combination of actual and predicted classes. The diagonal represents the classes which were correctly classified.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References 
(1) Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." PloS one 10.10 (2015). [doi: 10.1371/journal.pone.0140381.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4598126/)

(2) DENIZ KAVI. Brain Tumor Image Dataset, Retrieved 2/2024 from [Kaggle](https://www.kaggle.com/datasets/denizkavi1/brain-tumor?rvi=1)

(3) Kaiming H. et al. "Deep Residual Learning for Image Recognition". 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [URL](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

(4) Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.
