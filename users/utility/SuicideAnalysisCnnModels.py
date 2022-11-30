import re

from keras.layers import Conv1D, LSTM


class MainCNNModel:


    def startProcess(self):
        import pandas as pd
        import numpy as np
        import re
        import nltk
        from nltk.corpus import stopwords

        from numpy import array
        from keras.preprocessing.text import one_hot
        from keras.preprocessing.sequence import pad_sequences
        from keras.models import Sequential
        from keras.layers.core import Activation, Dropout, Dense
        from keras.layers import Flatten
        from keras.layers import GlobalMaxPooling1D
        from keras.layers.embeddings import Embedding
        from sklearn.model_selection import train_test_split
        from keras.preprocessing.text import Tokenizer
        from django.conf import settings
        path = settings.MEDIA_ROOT + "\\" + "suicideanalysis.csv"
        movie_reviews = pd.read_csv(path)

        movie_reviews.isnull().values.any()

        movie_reviews.shape
        movie_reviews.head()
        movie_reviews["review"][3]
        import seaborn as sns

        sns.countplot(x='sentiment', data=movie_reviews)

        def preprocess_text(sen):
            # Removing html tags
            sentence = remove_tags(sen)

            # Remove punctuations and numbers
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)

            # Single character removal
            sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence)

            return sentence

        TAG_RE = re.compile(r'<[^>]+>')

        def remove_tags(text):
            return TAG_RE.sub('', text)

        X = []
        sentences = list(movie_reviews['review'])
        for sen in sentences:
            X.append(preprocess_text(sen))

        X[3]
        y = movie_reviews['sentiment']

        y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        maxlen = 100

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        from numpy import array
        from numpy import asarray
        from numpy import zeros

        embeddings_dictionary = dict()
        glovpath = settings.MEDIA_ROOT+"\\"+"glove.6B.50d.txt"
        #glove_file = open('E:/Datasets/Word Embeddings/glove.6B.100d.txt', encoding="utf8")
        ## Get glove.6B.100D.txt file from kaggle it contain more than 300MB Data
        glove_file = open(glovpath, encoding='cp1252')

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        embedding_matrix = zeros((vocab_size, 100))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        model = Sequential()
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        print(model.summary())
        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        import matplotlib.pyplot as plt
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        model = Sequential()

        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)

        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

        score = model.evaluate(X_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        import matplotlib.pyplot as plt

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        model = Sequential()
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(128))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

        score = model.evaluate(X_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        import matplotlib.pyplot as plt

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        instance = X[57]
        print(instance)

        instance = tokenizer.texts_to_sequences(instance)

        flat_list = []
        for sublist in instance:
            for item in sublist:
                flat_list.append(item)

        flat_list = [flat_list]

        instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

        model.predict(instance)




