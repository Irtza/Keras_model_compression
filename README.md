# Keras Model Compression 

An Implementation of "Distilling the Knowledge in a Neural Network - Geoffery Hinton et. al" https://arxiv.org/abs/1503.02531


#### Usage 

python runme.py 


##### Metrics:
- Initial Accuracy on test set~= 0.99

##### Initial Model Parameters:
- Total params: 1,199,882
- Trainable params: 1,199,882

##### Evaluation of the Model of Reduction
- Compressed Model parameters:  74188
- Compression Rate :  16.2x
- Accuracy : 0.96


##### See Experiment Details in 
- report.md

![Compression Rate and Accuracy](/plots/CompressionRate_Accuracy.png){:class="img-responsive"}
![Parameter Size and Accuracy](/plots/parameterSize_Accuracy.png){:class="img-responsive"}






###### See experiments in Notebook. 

