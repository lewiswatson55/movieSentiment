import pandas as pd
import re
from tensorflow import keras
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Load Data
def load_data(train_path, test_path):
    return pd.read_csv(train_path, header=0), pd.read_csv(test_path, header=0)


# Preprocessing function
def preprocess(data):
    # Convert to lowercase
    data['review'] = data['review'].str.lower()
    # Remove HTML tags
    data['review'] = data['review'].str.replace('<br />', ' ')
    # Remove punctuation
    data['review'] = data['review'].str.replace('[^\w\s]', '')
    # Remove numbers
    data['review'] = data['review'].str.replace('\d+', '')
    # Remove stopwords
    stop = stopwords.words('english')
    data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # Remove short words
    data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
    # Remove long words
    data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if len(word) < 20]))
    # Remove extra spaces
    data['review'] = data['review'].str.replace(' +', ' ')
    # Remove leading and trailing spaces
    data['review'] = data['review'].str.strip()
    return data


# Tokenisation
def tokenise_data(X_train, X_test, num_words=10000):
    tokeniser = keras.preprocessing.text.Tokenizer(num_words=num_words)
    tokeniser.fit_on_texts(X_train['review'])
    return tokeniser.texts_to_sequences(X_train['review']), tokeniser.texts_to_sequences(X_test['review'])


# RNN model seems to be overfitting, even with double dropout - trying LSTM
def build_model(embedding_dim=16, rnn_units=16, dropout_rate=0.5):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(10000, embedding_dim))
    model.add(keras.layers.SimpleRNN(rnn_units, return_sequences=True))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.SimpleRNN(rnn_units))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(embedding_dim=16, lstm_units=16, dropout_rate=0.5):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(10000, embedding_dim))
    model.add(keras.layers.LSTM(lstm_units, return_sequences=True))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.LSTM(lstm_units))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    train, test = load_data('train.csv', 'test.csv')

    # Preprocess data
    X_train = preprocess(train)
    X_test = preprocess(test)

    # Extract labels
    Y_train = X_train['label']
    Y_test = X_test['label']

    # Remove labels from input data
    X_train = X_train.drop(['label'], axis=1)
    X_test = X_test.drop(['label'], axis=1)

    # Tokenise data
    X_train, X_test = tokenise_data(X_train, X_test)

    # Pad data
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=1000)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=1000)

    model = build_lstm_model()

    history = model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_split=0.2)

    results = model.evaluate(X_test, Y_test)
    print("Test Loss, Test Accuracy:", results)

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Additional evaluation metrics
    y_pred = model.predict(X_test)
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
    print(classification_report(Y_test, y_pred))

    # Save model
    model.save('model.keras')

    print(len(Y_train))