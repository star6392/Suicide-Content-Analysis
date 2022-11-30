from django.conf import settings
import nltk
class ProcesAndDetect:
    def preProcess(self,tweet):
        import pickle
        import re
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import nltk
        # nltk.download('stopwords')

        def preprocess_tweet(text):
            text = re.sub('<[^>]*>', '', text)
            emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
            lowercase_text = re.sub('[\W]+', ' ', text.lower())
            text = lowercase_text + ' '.join(emoticons).replace('-', '')
            return text

        tqdm.pandas()
        path = settings.MEDIA_ROOT + "\\" + "data.csv"
        df = pd.read_csv(path)
        df['tweet'] = df['tweet'].progress_apply(preprocess_tweet)

        from nltk.stem.porter import PorterStemmer
        porter = PorterStemmer()

        def tokenizer_porter(text):
            return [porter.stem(word) for word in text.split()]

        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        [w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]

        def tokenizer(text):
            text = re.sub('<[^>]*>', '', text)
            emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
            text = re.sub('[\W]+', ' ', text.lower())
            text += ' '.join(emoticons).replace('-', '')
            tokenized = [w for w in tokenizer_porter(text) if w not in stop]
            return tokenized

        from sklearn.feature_extraction.text import HashingVectorizer
        vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21,
                                 preprocessor=None, tokenizer=tokenizer)

        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log', random_state=1)

        X = df["tweet"].to_list()
        y = df['label']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.20,
                                                            random_state=0)

        X_train = vect.transform(X_train)
        X_test = vect.transform(X_test)

        classes = np.array([0, 1])
        clf.partial_fit(X_train, y_train, classes=classes)

        print('Accuracy: %.3f' % clf.score(X_test, y_test))

        clf = clf.partial_fit(X_test, y_test)

        label = {0: 'Non-suicidal', 1: 'Suicidal'}
        #example = ["I'll kill myself am tired of living depressed and alone"]
        example = [tweet]
        X = vect.transform(example)
        print('Prediction: %s\nProbability: %.2f%%'% (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))
        return label[clf.predict(X)[0]], round(np.max(clf.predict_proba(X)) * 100,2)

    def detectTypes(selfself,tweet):
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        stop_words = set(stopwords.words('english'))
        tokenized = sent_tokenize(tweet)
        for i in tokenized:
            # Word tokenizers is used to find the words
            # and punctuation in a string
            wordsList = nltk.word_tokenize(i)

            # removing stop words from wordList
            wordsList = [w for w in wordsList if not w in stop_words]

            #  Using a Tagger. Which is part-of-speech
            # tagger or POS-tagger.
            tagged = nltk.pos_tag(wordsList)

            print(tagged)