import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import h5py
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

#define params
trn_file = sys.argv[1]
tst_file = sys.argv[2]
window_size = sys.argv[3]

nb_classes = 2
nb_kernels = 3
nb_pools = 2
window_size = int(window_size)

train = pd.read_csv(trn_file, header=None)
X = train.iloc[:,1:20*window_size+1]
Y = train.iloc[:,0]
#train_y = to_categorical(train_class.values)
print('Original dataset shape {}'.format(Counter(Y)))

test = pd.read_csv(tst_file, header=None)
X1 = test.iloc[:,1:20*window_size+1]
Y1 = test.iloc[:,0]

def LoadTrainingData(pathTrain):
	INPUT_SHAPE = (20, window_size, 1) # input dimensions
	#INPUT_SHAPE = (1, 20, 20) # input dimensions
	# LOAD DATA
	train = pd.read_csv(pathTrain, header=None)
	
	train_x = train.drop(train.columns[0], axis=1).values.reshape(
		-1, *INPUT_SHAPE
	).astype(float)
	#print(train_x);

	train_class = train.iloc[:,0]
	train_y = np_utils.to_categorical(train_class.values)

	from sklearn.preprocessing import LabelEncoder
	label_encoder = LabelEncoder()
	train_y = label_encoder.fit_transform(train_class)

	return train_x, train_y

def LoadTestingData(pathTest):
	INPUT_SHAPE = (20, window_size, 1) # input dimensions
	#INPUT_SHAPE = (1, 20, 20) # input dimensions
	# LOAD DATA
	test = pd.read_csv(pathTest, header=None)
	
	test_x = test.drop(test.columns[0], axis=1).values.reshape(
		-1, *INPUT_SHAPE
	).astype(float)
	test_class = test.iloc[:,0]
	test_y = np_utils.to_categorical(test_class.values)

	#print(test_x);
	return test_x, test_y

def cnn_model():
	model = Sequential()

	# model.add(Dropout(0.2, input_shape = (1,20,window_sizes)))
	model.add(ZeroPadding2D((1,1), input_shape = (1,20,window_size)))
	model.add(Convolution2D(32, nb_kernels, nb_kernels))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

	# model.add(ZeroPadding2D((1,1)))
	# model.add(Convolution2D(64, nb_kernels, nb_kernels, activation='relu'))
	# # model.add(Activation('relu'))
	# model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

	# model.add(ZeroPadding2D((1,1)))
	# model.add(Convolution2D(128, nb_kernels, nb_kernels, activation='relu'))
	# model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

	# model.add(ZeroPadding2D((1,1)))
	# model.add(Convolution2D(256, nb_kernels, nb_kernels, activation='relu'))
	# model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

	## add the model on top of the convolutional base
	#model.add(top_model)
	model.add(Flatten())
	# model.add(Dropout(0.1))
	model.add(Dense(32))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	#model.add(BatchNormalization())
	model.add(Activation('softmax'))

	# f = open('model_summary.txt','w')
	# f.write(str(model.summary()))
	# f.close()

	#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	return model

def mlp_model():
	model = Sequential() # The Keras Sequential model is a linear stack of layers.
	model.add(Dense(100, init='uniform', input_dim=400)) # Dense layer
	model.add(Activation('tanh')) # Activation layer
	model.add(Dropout(0.5)) # Dropout layer
	model.add(Dense(100, init='uniform')) # Another dense layer
	model.add(Activation('tanh')) # Another activation layer
	model.add(Dropout(0.5)) # Another dropout layer
	model.add(Dense(2, init='uniform')) # Last dense layer
	model.add(Activation('softmax')) # Softmax activation at the end
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # Using Nesterov momentum
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Using logloss
	return model	

# X, Y = LoadTrainingData(trn_file)
# X1, Y1 = LoadTestingData(tst_file)
# define 10-fold cross validation test harness
# f = open('ws09_output.txt', 'w')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
over_sample = SMOTE(kind='svm')
for train, test in kfold.split(X, Y):
	model = cnn_model()
	# Oversampling
	# train_x, train_y = over_sample.fit_sample(X.iloc[train], Y.iloc[train])
	# print('Resampled dataset shape {}'.format(Counter(train_y)))
	# trn_new = numpy.asarray(train_x)
	trn_new = numpy.asarray(X.iloc[train])
	# print('New training: ', trn_new)
	tst_new = numpy.asarray(X.iloc[test])
	# print('New test: ', tst_new)
	## evaluate the model
	model.fit(trn_new.reshape(len(trn_new),1,20,window_size), np_utils.to_categorical(Y.iloc[train],nb_classes), nb_epoch=15, batch_size=10, verbose=0)
	# evaluate the model
	# scores = model.evaluate(tst_new.reshape(len(tst_new), np_utils.to_categorical(Y.iloc[test],nb_classes), verbose=0))
	# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	# cvscores.append(scores[1] * 100)
	#prediction
	true_labels_cv = numpy.asarray(Y.iloc[test])
	predictions = model.predict_classes(tst_new.reshape(len(tst_new),1,20,window_size))
	print('\nCV: ', confusion_matrix(true_labels_cv, predictions))
	# f.write(str(confusion_matrix(true_labels_cv, predictions)))
# Fit the model
# save best weights
# model = cnn_model()
#plot_model(model, to_file='model.png')
model = cnn_model()
# X, Y = over_sample.fit_sample(X, Y)
# print('Resampled dataset shape {}'.format(Counter(Y)))
# X = X.reshape(-1,1,20,window_size)
X = numpy.asarray(X)

model.fit(X.reshape(len(X),1,20,window_size), np_utils.to_categorical(Y,nb_classes), nb_epoch=15, batch_size=10, class_weight = 'auto', verbose=0)
# evaluate the model
# scores = model.evaluate(X1.reshape(len(X1),1,20,window_size), np_utils.to_categorical(Y1,nb_classes), verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)
#prediction
#model.load_weights(filepath)
X1 = numpy.asarray(X1)
predictions = model.predict_classes(X1.reshape(len(X1),1,20,window_size))
pred_labels = numpy.asarray(predictions)
true_labels = numpy.asarray(Y1)
tst_confu = confusion_matrix(true_labels, pred_labels)
print('\nIndependent test: ', tst_confu)
