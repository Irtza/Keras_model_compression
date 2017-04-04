import keras
import h5py
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt 


from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , Activation

plt.rcParams['figure.figsize'] = (8, 6)


# from skimage import io
# def show(img):
	# io.imshow(img)
	# io.show()
	
def softmax_c(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis] 
	return e_x / div

def prepare_softtargets(model,X):
	inp = model.input                                           # input placeholder
	outputs = []
	for layer in model.layers[:]:
		if layer.name == 'flatten_1':
			outputs.append(layer.output)
		if layer.name == 'dense_2':
			outputs.append(layer.output)
			
	functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
	layer_outs = functor([X, 1.])
	return np.array(layer_outs[0]) , np.array(layer_outs[1])


# Todo parse as cmd argments 
TRAIN_BIG = False  
TRAIN_SMALL = False 
PRPARE_TRAININPUT = True 
PRPARE_TESTINPUT = True 

# big model 
batch_size = 128
num_classes = 10
BIG_epochs = 20

# Small Model
STUDENT_epochs = 30
HiddenNeuron = 6   # found out after experimentation. 


### Setup Data 

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


### Buld Model 
model = Sequential()
model.add(Conv2D(32,3,3, activation='relu', input_shape=input_shape))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])


if TRAIN_BIG: 
	print "Training Model TRAIN_BIG FLAG set as TRUE"
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  nb_epoch=BIG_epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))

	print "Evaluating model .."
	score = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	print ("Savin Model weights")
	model.save_weights("new_stockweighs.h5")
	print ("Model Weights Saved")

else:
	print "loading weights TRAIN_BIG FLAG set as FALSE"
	model.load_weights('stockweighs.h5')



print "Evaluating Initial Model ...\n"
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print ('-'*30)

print "Parameter Size of Initial Model and Memory Footprint "
trainable_params = model.count_params()
footprint = trainable_params * 4
print ("Memory footprint per Image Feed Forward ~= " , footprint / 1024.0 /1024.0 ,"Mb") # 2x Backprop
print ('-'*30)


## Obtaining the Output of the last Convolutional Layer After Flatten. Caruana et. al 
## and Preparing the SoftTargets, (Logits), as proposed Geoffery Hinton et. al 




if PRPARE_TRAININPUT:
	print "Creating trainsfer Set"

	lastconv_out = []
	logit_out = []
	for i in range(0,60):
		print "Batch # : ",i
		l,l2 =  (prepare_softtargets(model,x_train[i*1000:(i+1)*1000]))
		lastconv_out.append(l)
		logit_out.append(l2)

	lastconv_out = np.array(lastconv_out)
	logit_out = np.array(logit_out)
	lastconv_out = lastconv_out.reshape((60000 , 9216))
	logit_out = logit_out.reshape((60000 , 10))

	print "clean up " 
	x_train = 0
	print "Write to Disk"
	h5f = h5py.File('new_lastconv_out.h5', 'w')
	h5f.create_dataset('dataset_1', data=lastconv_out)
	h5f.close()
	h5f2 = h5py.File('new_logit_out.h5', 'w')
	h5f2.create_dataset('dataset_1', data=logit_out)
	h5f2.close()

else:
	print "loading Transfer Set from lastconv_out.h5"
	h5f = h5py.File('lastconv_out.h5' , 'r')
	lastconv_out = h5f['dataset_1'][:]
	h5f.close()

	h5f2 = h5py.File('logit_out.h5' , 'r')
	logit_out = h5f2['dataset_1'][:]
	h5f2.close()



print "Building minimal Model"

student_model = Sequential()
student_model.add(Dense(HiddenNeuron,input_dim=9216,activation='relu'))
student_model.add(Dropout(0.2))
student_model.add(Dense(num_classes))

student_model.compile(loss='mse',
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])



if TRAIN_SMALL: 
	print "Training Small Model "
	student_model.fit(lastconv_out,logit_out,nb_epoch=STUDENT_epochs,verbose=1 , batch_size=batch_size)
	student_model.save_weights("new_student_weights_6_0.2dopout.h5")
else :
	print "Loading Small Model Weights"
	student_model.load_weights("student_weights_6_0.2dopout.h5")



print "Clean up small model Training and targets"
lastconv_out = 0
logit_out = 0 


############ Preparing Test Input #########

if PRPARE_TESTINPUT:
	print "creating Test data from the big Model on HeldOut data"
	
	test_lastconv_out = []
	test_logit_out = []
	for i in range(0,10):
		print "Batch # : ",i
		l,l2 =  prepare_softtargets(model,x_test[i*1000:(i+1)*1000])
		test_lastconv_out.append(l)
		test_logit_out.append(l2)    
	
	# lastconv_out.shape , logit_out.shape
	test_lastconv_out = np.array(test_lastconv_out)
	test_logit_out = np.array(test_logit_out)

	test_lastconv_out = test_lastconv_out.reshape((10000 , 9216))
	test_logit_out = test_logit_out.reshape((10000 , 10))

	print test_lastconv_out.shape
	print test_logit_out.shape

	print "Write to Disk"
	h5f = h5py.File('new_test_lastconv_out.h5', 'w')
	h5f.create_dataset('dataset_1', data=lastconv_out)
	h5f.close()
	h5f2 = h5py.File('new_test_logit_out.h5', 'w')
	h5f2.create_dataset('dataset_1', data=logit_out)
	h5f2.close()
else :
	print "Loading saved test data from .h5 . "
	h5f = h5py.File('test_lastconv_out.h5' , 'r')
	test_lastconv_out = h5f['dataset_1'][:]
	h5f.close()
	h5f2 = h5py.File('test_logit_out.h5' , 'r')
	test_logit_out = h5f2['dataset_1'][:]
	h5f2.close()
	
pred = student_model.predict(test_lastconv_out)
probs = softmax_c(pred)
pred_classes = np.argmax(probs,axis=1)

accuracy_student = metrics.accuracy_score(y_pred=pred_classes,y_true=np.argmax(y_test,axis=1))
print "Small Model Test Set Accuracy : " , accuracy_student
print "\n"
# Compression Rate from Number of Parameters Reduced

# Parameters for the first two conv layers from Bigger model
convparams = 320 + 18496

print "Evaluating COompression ... "
print "HiddenNeurons : " , HiddenNeuron
print "Initial Model Parameters : " , model.count_params()
print "Compressed Model parameters + initial feature extractor part params : ", student_model.count_params() + convparams
compressionRate = model.count_params() / np.float(student_model.count_params()  + convparams)
print "Compression Rate : " , compressionRate
print "\n\n"

# if __name__ == '__main__':
	# Set usage using flags. 
	# python runme.py --TRAIN_BIG=True --epochs=20 