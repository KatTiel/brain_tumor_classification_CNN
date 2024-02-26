## Brain Tumor Classification - Computer Vision Model using a Convolutional Neuronal Network


## How It Works
### Prerequisites 

### Data Set & Preprocessing

### Model Architecture

#### Choice of Pre-trained Base 
##### ResNet50

#### Strategies to Diminish Overfitting
##### Dropout
Randomly selected neurons are dropped out or ignored with a certain probability. This encourages generalization as the network does not rely too heavily on individual neurons.

##### L1-Regularization
##### L2-Regularization
##### Early Stopping
During training, the model's performance on the validation set is monitored. The training is stopped and the weights are restored when the performance on the validation set diminishes. This prevents the model from learning noise in the training data (....).

### Performance Measurements
Several performance valuations were conducted to gain a detailed perspective on the predictions made with the test data:



#### Accuracy
<img width="161" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/7417c4b4-09d8-4dba-bb11-8e9e9dbebc1e">

#### Precision
<img width="119" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/af1d55dd-f7a5-4633-95bf-eb81504beeb7">

#### Recall
<img width="99" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/997983e0-fab6-449d-a330-bf6c128055a6">

#### F1-Score 
<img width="506" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/0efc1f75-eccb-4670-ace2-637573984049">

#### Confusion Matrix

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References 
(1) Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." PloS one 10.10 (2015). [URL](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4598126/).

(2) DENIZ KAVI. Brain Tumor Image Dataset, Retrieved 2/2024 from [URL](https://www.kaggle.com/datasets/denizkavi1/brain-tumor?rvi=1).

(3) Kaiming H. et al. "Deep Residual Learning for Image Recognition". 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [URL](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

(4) Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.
