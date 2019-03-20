#855
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split#, GridSearchCV, PredefinedSplit
from sklearn.utils import shuffle
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
# from keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('clean_data_35_notencoded.csv')

Y=df.iloc[:, 6].values

X = pd.read_csv('encoded.csv')

Y.shape

X.drop(['Unnamed: 0'], inplace = True, axis = 1)


s = pd.Series(Y)

df2 = pd.concat([X, s], axis = 1)

df2 = shuffle(df2)

Y = df2[0]

df2.drop(0, axis = 1, inplace = True)
X = df2

Z = pd.get_dummies(Y)

print("Data Prepared...")


# def createModel(optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(1000, activation='relu', input_shape = (315,)))
#     model.add(Dropout(0.3))
#     model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.0001)))
#     model.add(Dropout(0.2))
#     model.add(Dense(250, activation='relu', kernel_regularizer=l2(0.0001)))
#     model.add(Dropout(0.1))
#     model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.0001)))
#     model.add(Dense(35, activation='softmax'))
#     print("Compiling Model...")
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

model = Sequential()
model.add(Dense(500, activation='relu', kernel_initializer='normal', input_shape = (315,)))
model.add(Dropout(0.3))
model.add(Dense(250, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.0001)))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.0001)))
#model.add(Dropout(0.1))
model.add(Dense(35, activation='softmax'))

EPOCHS = 1000

sd=[]
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(losses):
    lrate=0.01*1/(1+0.1*len(history.losses))
    momentum=0.8
    decay_rate=2e-6
    return lrate

history=LossHistory()
lrate=LearningRateScheduler(step_decay)

# lrate = LearningRateScheduler(step_decay)

sgd = SGD(lr=0.1, momentum=0.0, decay=5e-5, nesterov=False)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50)

print("Compiling Model...")
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



x_train, x_test, y_train, y_test = train_test_split(X, Z, test_size = 0.2, random_state = 1)

x_train = x_train.values
x_test = x_test.values

# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# validation_idx = np.repeat(-1, Y.shape)
# np.random.seed(550)
# validation_idx[np.random.choice(validation_idx.shape[0], int(round(.4*validation_idx.shape[0])), replace = False)] = 0
# validation_split = list(PredefinedSplit(validation_idx).split())

# print(len(validation_split[0][0]))

# optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam'] #Adamax, SGD
# param_grid = dict(optimizer=optimizer)
# model = KerasClassifier(build_fn=createModel, epochs = 100, batch_size = 1000, verbose = 0)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


print("Training...")
model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[es, history, lrate], verbose=1)
# grid_result = grid.fit(X, Y)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

print("Saving...")
fname = "model.sav"
# model_json = model.to_json()
# with open(fname + ".json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights(fname + ".h5")
joblib.dump(model, fname)
print("Model Saved.")
