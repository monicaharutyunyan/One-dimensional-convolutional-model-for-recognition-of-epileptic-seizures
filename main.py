# multi-headed cnn model
import numpy as np
from sklearn.utils import shuffle
import random as rand
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import concatenate
import matplotlib.pyplot as plt


"""
1. Read Artefacts and Seizures
2. Shuffle data
3. Dived data: 30->Test, 70->Train
"""

# load a single file as a numpy array
def load_file(filepath):
   dataframe = read_csv(filepath, header=None, delim_whitespace=True)
   return dataframe.values


def create_np_array(data_0, data_1):
   win_sz = 1599
   data = list()
   data_len_0 = len(data_0)
   data_len_1 = len(data_1)
   chunk_cnt_0 = int(data_len_0 / win_sz)
   chunk_cnt_1 = int(data_len_1 / win_sz)


   chunk_cnt = int(chunk_cnt_0 + chunk_cnt_1)
   dt = np.empty((int(chunk_cnt), win_sz))
   lb = np.ones((int(chunk_cnt), 1), dtype=np.int8)
   i = 0
   j = 0
   while j < chunk_cnt:
       i = 0
       while i < win_sz:
           if j < chunk_cnt_0:
               dt[j, i] = data_0[i + j * win_sz][0]
           else:
               dt[j, i] = data_1[i + (j - int(chunk_cnt_0)) * win_sz][0]
           i = i + 1
       if j < chunk_cnt_0:
           lb[j, 0] = 0
       else:
           lb[j, 0] = 1
       j = j + 1
   data.append(dt)
   data = dstack(data)


   return data, lb


def load_my_dataset_group():
   read_artefacts_train = np.array(load_file('Data/Artefacts/Train_Artefacts.txt'))
   read_artefacts_test = np.array(load_file('Data/Artefacts/Test_Artefacts.txt'))
   read_seizures_train = np.array(load_file('Data/Seizures/Train_Seizures.txt'))
   read_seizures_test = np.array(load_file('Data/Seizures/Test_Seizures.txt'))
   artefacts = np.concatenate((read_artefacts_test, read_artefacts_train))
   seizures = np.concatenate((read_seizures_test, read_seizures_train))
   artefacts = shuffle(artefacts)
   seizures = shuffle(seizures)

   range_train_artefacts = int((len(artefacts) * 70) / 100) - 1
   range_test_artefacts = int((len(artefacts) * 30) / 100) - 1
   range_train_seizures = int((len(seizures) * 70) / 100) - 1
   range_test_seizures = int((len(seizures) * 30) / 100) - 1
   # print(f"Length of train artefacts: {range_train_artefacts}")
   # print(f"Length of test artefacts: {range_test_artefacts}")
   # print(f"Length of train seizures: {range_train_seizures}")
   # print(f"Length of test seizures: {range_test_seizures}")
   artefacts_test = np.empty(shape=(range_test_artefacts, 1))
   artefacts_train = np.empty(shape=(range_train_artefacts, 1))
   seizures_test = np.empty(shape=(range_test_seizures, 1))
   seizures_train = np.empty(shape=(range_train_seizures, 1))

   #Artefacts
   j = 0
   while j < range_train_artefacts:
       artefacts_train[j] = artefacts[j]
       j += 1
   for i in range(range_test_artefacts):
       artefacts_test[i] = artefacts[j]
       j += 1

   #Seizures
   j = 0
   while j < range_train_seizures:
       seizures_train[j] = seizures[j]
       j += 1
   for i in range(range_test_seizures):
       seizures_test[i] = seizures[j]
       j += 1
   trainX, trainY = create_np_array(artefacts_train, seizures_train)
   testX, testY = create_np_array(artefacts_test, seizures_test)


   return trainX, trainY, testX, testY


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
   trainX, trainy, testX, testy = load_my_dataset_group()
   # print(f"Shape of trainX: {trainX.shape}")
   # print(f"Shape of trainY: {trainy.shape}")
   # print(f"Shape of trainX: {testX.shape}")
   # print(f"Shape of trainY: {testy.shape}")
   trainy = to_categorical(trainy)
   testy = to_categorical(testy)
   print(trainX.shape, trainy.shape, testX.shape, testy.shape)
   return trainX, trainy, testX, testy


# standardize data
def scale_data(trainX, testX):
   # remove overlap
   cut = int(trainX.shape[1] / 2)
   longX = trainX[:, -cut:, :]
   # flatten windows
   longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
   # flatten train and test
   flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
   flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
   # standardize
   s = StandardScaler()
   # fit on training data
   s.fit(longX)
   # apply to training and test data
   longX = s.transform(longX)
   flatTrainX = s.transform(flatTrainX)
   flatTestX = s.transform(flatTestX)
   # reshape
   flatTrainX = flatTrainX.reshape((trainX.shape))
   flatTestX = flatTestX.reshape((testX.shape))
   return flatTrainX, flatTestX


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
   verbose, epochs, batch_size = 0, 10, 32
   n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
   # head 1
   # trainX, trainy = shuffle(trainX, trainy)
   # testX, testy = shuffle(testX, testy)
   trainX, testX = scale_data(trainX, testX)


   inputs1 = Input(shape=(n_timesteps, n_features))
   conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
   drop1 = Dropout(0.5)(conv1)
   pool1 = MaxPooling1D(pool_size=2)(drop1)
   flat1 = Flatten()(pool1)
   # head 2
   inputs2 = Input(shape=(n_timesteps, n_features))
   conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
   drop2 = Dropout(0.5)(conv2)
   pool2 = MaxPooling1D(pool_size=2)(drop2)
   flat2 = Flatten()(pool2)
   # head 3
   inputs3 = Input(shape=(n_timesteps, n_features))
   conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
   drop3 = Dropout(0.5)(conv3)
   pool3 = MaxPooling1D(pool_size=2)(drop3)
   flat3 = Flatten()(pool3)
   # merge
   merged = concatenate([flat1, flat2, flat3])
   # interpretation
   dense1 = Dense(100, activation='relu')(merged)
   outputs = Dense(n_outputs, activation='softmax')(dense1)
   model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
   # save a plot of the model
   plot_model(model, show_shapes=True, to_file='multichannel.png')
   opt = keras.optimizers.Adam(learning_rate=0.009)
   model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )
   # fit network
   model.fit([trainX, trainX, trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
   # evaluate model
   _, accuracy = model.evaluate([testX, testX, testX], testy, batch_size=batch_size, verbose=0)
   return accuracy


# summarize scores
def summarize_results(scores):
   print(scores)
   m, s = mean(scores), std(scores)
   print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
   # pyplot.boxplot(scores, labels=params)
   # pyplot.savefig('exp_cnn_standardize.png')


# run an experiment
def run_experiment(repeats=10):
   trainX, trainy, testX, testy = load_dataset()
   scores = list()


   for r in range(repeats):
       score = evaluate_model(trainX, trainy, testX, testy)
       score = score * 100.0
       print('#%d: %.3f' % (r+1, score))
       scores.append(score)
   print('Max: ', max(scores))
   summarize_results(scores)


run_experiment()




