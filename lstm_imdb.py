

import tensorflow as tf
from keras.datasets import imdb
from keras_preprocessing import sequence

from lstm_layer import buildLstmLayer

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32
model_name = 'lstm-sentiment-imdb'


def load_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test

def build_model():
    print('Build model...')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, 128, input_length=maxlen))
    model.add(tf.keras.layers.Lambda(buildLstmLayer, arguments={'num_layers': 2, 'num_units': 128}))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def extract_tflite_model():
    tf.reset_default_graph()
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5',
                                                              input_shapes={'embedding_input': [1, maxlen]})
    tflite_model = converter.convert()
    open(model_name + '.tflite', 'wb').write(tflite_model)
    print('Model converted successfully!')


def train_save_model(model, x_train, y_train, x_test, y_test):
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save(model_name + '.h5')
