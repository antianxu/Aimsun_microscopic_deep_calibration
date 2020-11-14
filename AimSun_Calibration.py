import os
import tensorflow as tf
from keras import backend as K, regularizers
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, multiply, add, concatenate
from keras.optimizers import adam
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import csv_parser
import subprocess
#import evaluate

OUTPUT_INDICES = [0,1,2,3,4,5,9]

OUTPUT_RANGES = [[0.1, 1.5], [-1, 1], [1, 4], [1, 5], [0, 100], [0, 100], [0.1, 2.0]]

AIMSUM_PROGRAM_NAME = 'calibration_data_gen.py'

# Limit outputs as a set of sigmoid activation functions at the output layer.
def output_limits(x, beta, ranges):
    assert x.shape[1] == len(OUTPUT_INDICES), F"Input shape into the limit activation is {np.shape(x)}, \
                                                however the expected shape is ({np.shape(x)[0]}, {len(OUTPUT_INDICES)})"

    x = add([multiply([tf.reshape(K.sigmoid(x * beta), [len(OUTPUT_INDICES), ]), (ranges[:, 1] - ranges[:, 0])]), ranges[:, 0]])

def structure2():
    tf.compat.v1.enable_eager_execution()

    def aimsun_loss(detector_layer, interval_layer, value_layer):
        #row_index = np.where(y == y_true)[0][0]

        #detector_id = detector_input[row_index]
        #interval    = interval_input[row_index]
        #value       = value_input[row_index]
        x = [[2.]]
        m = tf.matmul(x, x)
        print("hello, {}".format(m))

        detector_id = detector_layer
        print(detector_id)
        interval    = interval_layer
        value       = value_layer

        x_true = (detector_id, interval, value)

        def loss(y_true, y_predict):
            return evaluate.evaluate(x_true, y_predict)

        return loss
      

    cwd = os.getcwd()
    x1, y1 = csv_parser.csv_read_structure2(cwd + '/../data2/dataset1')
    x2, y2 = csv_parser.csv_read_structure2(cwd + '/../data3/dataset2')
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    data_size = len(x)

    #Split inputs & select output
    #detector_input  = x.take([0], axis=1)
    interval_input  = x.take([1], axis=1)
    value_input     = x.take([2,3], axis=1)
    location_input  = x.take([4,5,6,7], axis=1)

    selected_output = y.take(OUTPUT_INDICES, axis=1)

    #Detector input
    #detector_input_layer = Input(shape=(1, ))   # No forward pass on this layer, only to keep track of data

    #Interval input
    interval_input_layer = Input(shape=(1, )) 
    interval_dense_1     = Dense(5, kernel_regularizer=regularizers.l1_l2())(interval_input_layer)

    #Value input
    value_input_layer = Input(shape=(2, )) 
    value_normalize   = BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(value_input_layer)
    value_dense_1     = Dense(10, kernel_regularizer=regularizers.l1_l2())(value_normalize)
    value_dense_2     = Dense(20, kernel_regularizer=regularizers.l1_l2())(value_dense_1)

    #Location input
    location_input_layer = Input(shape=(4, ))
    location_dense_1     = Dense(4, kernel_regularizer=regularizers.l1_l2())(location_input_layer)
    location_dense_2     = Dense(2, kernel_regularizer=regularizers.l1_l2())(location_dense_1)

    #Concatenate inputs
    combined_layer  = concatenate([interval_dense_1, value_dense_2, location_dense_2])
    combined_dense1 = Dense(50, kernel_regularizer=regularizers.l1_l2())(combined_layer)
    combined_dense2 = Dense(50, kernel_regularizer=regularizers.l1_l2())(combined_dense1)
    output          = Dense(len(OUTPUT_INDICES), kernel_regularizer=regularizers.l1_l2())(combined_dense2)
    limited_output  = Activation(output_limits(output, beta=1, ranges=tf.convert_to_tensor(OUTPUT_RANGES, dtype=tf.float32)))(output) 

    model = Model([interval_input_layer, value_input_layer, location_input_layer], limited_output)

    model.compile(loss='mean_squared_error', optimizer=adam(learning_rate = 0.0005))

    max_epoch = 500

    history = model.fit([interval_input, value_input, location_input], selected_output, batch_size=1, epochs=max_epoch, validation_split=0.2, verbose=1, shuffle=True)

    save_model(model, cwd + '/../model/prediction_model_structure2.h5')

    y_predict = model.predict([interval_input, value_input, location_input])

    r2 = r2_score(selected_output, y_predict, multioutput='raw_values')
    print(F"\nR2: {r2}")

    #csv_parser.csv_write(y, cwd + '/../output/true.csv')
    csv_parser.csv_write(y_predict, OUTPUT_INDICES, cwd + '/../output/predict_structure2.csv')

    val_loss = history.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_index = val_loss.index(min_val_loss)

    print(F"\nMinimum validation mse: {min_val_loss:.5f}\nEpoch: {min_val_loss_index}")

    # summarize history for loss
    #plot_min_epoch = 500
    
    #plt.xlim([plot_min_epoch, max_epoch])
    plt.ylim([0, 500])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(F'mean squared loss: Indices {OUTPUT_INDICES} \
                \nMinimum validation mse: {min_val_loss:.5f}(Epoch: {min_val_loss_index})')
    plt.ylabel('msl')
    plt.xlabel('epoch')
    plt.legend([F'train (final loss: {history.history["loss"][-1]:.2f})', \
                F'test (final loss: {history.history["val_loss"][-1]:.2f})'], loc='upper right')
    plt.show()

def structure1():
    #def aimsun_loss(y_true, y_predict):
    #    index = np.where(y == y_true)[0][0]
    #    x_true = x[index]
    #
    #    subprocess.run('python ' + AIMSUM_PROGRAM_NAME + F' {y_predict[0]}')
    
    cwd = os.getcwd()
    x1, y1 = csv_parser.csv_read_structure1(cwd + '/../data2/dataset1')
    x2, y2 = csv_parser.csv_read_structure1(cwd + '/../data3/dataset2')
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    data_size = len(x)

    inputs            = Input(shape=(csv_parser.INPUT_SIZE, )) #Input layer    
    normalized_inputs = BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(inputs) 
    #dense1            = Dense(50, kernel_regularizer=regularizers.l1_l2())(normalized_inputs)
    dense2            = Dense(50, kernel_regularizer=regularizers.l1_l2())(normalized_inputs)
    dense3            = Dropout(0.2)(Dense(100)(dense2))
    dense4            = Dense(50, kernel_regularizer=regularizers.l1_l2())(dense3)
    #dense5            = Dense(50, kernel_regularizer=regularizers.l1_l2())(dense4)
    output            = Dense(len(OUTPUT_INDICES), kernel_regularizer=regularizers.l1_l2())(dense4)
    limited_output    = Activation(output_limits(output, beta=1, ranges=tf.convert_to_tensor(OUTPUT_RANGES, dtype=tf.float32)))(output) #Output layer

    model = Model(inputs, limited_output)

    model.compile(loss='mean_squared_error', optimizer=adam(learning_rate = 0.0005))

    max_epoch = 5000

    y = y.take(OUTPUT_INDICES, axis=1)

    history = model.fit(x, y, batch_size=1000, epochs=max_epoch, validation_split=0.2, verbose=1, shuffle=True)

    save_model(model, cwd + '/../model/prediction_model_structure1.h5')

    y_predict = model.predict(x)

    r2 = r2_score(y, y_predict, multioutput='raw_values')
    print(F"\nR2: {r2}")

    #csv_parser.csv_write(y, cwd + '/../output/true.csv')
    csv_parser.csv_write(y_predict, OUTPUT_INDICES, cwd + '/../output/predict.csv')

    val_loss = history.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_index = val_loss.index(min_val_loss)

    print(F"\nMinimum validation mse: {min_val_loss:.5f}\nEpoch: {min_val_loss_index}")

    # summarize history for loss
    #plot_min_epoch = 500
    
    #plt.xlim([plot_min_epoch, max_epoch])
    plt.ylim([0, 500])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(F'mean squared loss: Indices {OUTPUT_INDICES} \
                \nMinimum validation mse: {min_val_loss:.5f}(Epoch: {min_val_loss_index})')
    plt.ylabel('msl')
    plt.xlabel('epoch')
    plt.legend([F'train (final loss: {history.history["loss"][-1]:.2f})', \
                F'test (final loss: {history.history["val_loss"][-1]:.2f})'], loc='upper right')
    plt.show()

structure2()





