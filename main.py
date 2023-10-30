# Binary sentiment analysis of IMDB reviews using a simple RNN

import pandas as pd
from tensorflow import keras
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#import nltk
#nltk.download('stopwords')


# Load in train and test data, NOTE: i have added a header to the two datasets
train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)


# Preprocess data
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

#print(train['review'][0])

X_train = preprocess(train)
X_test = preprocess(test)

# Get Y values into own dataframe
Y_train = X_train['label']
Y_test = X_test['label']

# Remove Y values from X dataframes
X_train = X_train.drop(['label'], axis=1)
X_test = X_test.drop(['label'], axis=1)


# Tokenise data
tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train['review'])

X_train = tokenizer.texts_to_sequences(X_train['review'])
X_test = tokenizer.texts_to_sequences(X_test['review'])

# Pad data - ensure all vectors are same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train)
X_test = keras.preprocessing.sequence.pad_sequences(X_test)

# Create model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.SimpleRNN(16, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.SimpleRNN(16))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train model

history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate model
results = model.evaluate(X_test, Y_test)
print(results) # returns loss and accuracy

# Save model
model.save('model.h5')

