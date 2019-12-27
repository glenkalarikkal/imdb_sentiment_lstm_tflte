import os

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

from lstm_imdb import load_data, build_model, train_save_model, extract_tflite_model

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    train_save_model(model, x_train, y_train, x_test, y_test)
    extract_tflite_model()
