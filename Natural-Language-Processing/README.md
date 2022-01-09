# Natural Language Processing

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import string
import pickle

from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

```

**Bag-of-Words**

```
rng = np.random.RandomState(42)
``` 

**Newsgroups**
```
categories = [
    'alt.atheism',
    'talk.politics.guns',
    'comp.graphics',
    'sci.space',
]
```
**Load train and test data from fetch_20newsgroups**
```
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=rng)

test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=rng)
```
**Print out keys and data for the first file in the train set**
```
train.keys()

print(f"Data: {train.data[0][:40]}\n")
print(f"Filename: {train.filenames[0]}\n")
print(f"Category: {train.target_names[0]}\n")
print(f"Category, numeric label: {train.target[0]}")
```

**Summary of train and test datasets**

**Create a counter object from targets (category) of train and test sets**
```
train_counter = Counter(train.target)
test_counter =  Counter(test.target)
```

**Create dataframe with counted n.o. files belonging to a certain category**
```
cl = pd.DataFrame(data={
    'Train': { **{ train.target_names[index]: count for index, count in train_counter.items()}, 'Total': len(train.target)},
    'Test':  { **{test.target_names[index]: count for index, count in test_counter.items()},  'Total': len(test.target)},
})

cl.columns = pd.MultiIndex.from_product([["Class distribution"], cl.columns])
cl
```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Natural-Language-Processing/1.png?raw=true)

**Transform text to a TF-IDF-weighted document-term matrix**

```
# Text document
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']

# Vocabulary
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']

# Create tokens from text given vocabulary (i.e. create the count vectorizer)
token_matrix = CountVectorizer(vocabulary=vocabulary)

# Convert count matrix to TF-IDF format (i.e.create the tfi-df trasformer)
tfid_transform = TfidfTransformer()

# Chain steps
pipe = Pipeline([('count', token_matrix),
                 ('tfid', tfid_transform)])

# Fit data
pipe.fit(corpus)

# Display tokenized text
pipe['count'].transform(corpus).toarray()

```

### A pre-processing pipeline to convert documents to vectors using bag-of-words and TF-IDF

``` 
# Remove numbers, lines like - or _ or combinations like "/|", "||/" or "////"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)      
    text = re.sub(r'[-_]+', '', text)     
    text = re.sub(r'\/*\|+|\/+', '', text)  
    return text

'''
Corpus pre-processing pipeline

The inputs to the function are:
- list of strings (one element is a document)
- maximum number of features (size of the vocabulary) to use  
Returns a Pipeline object   
'''

```
def text_processing_pipeline(features=None):
	vectorizer = CountVectorizer(preprocessor=preprocess_text, analyzer='word', stop_words='english', max_features=features)
    tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
    pipeline = Pipeline([('count', vectorizer),('tfid', tfidf)]).fit(corpus)
    return pipeline

```

CountVectorizer

Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term or token counts. 

TfidfTransformer.

Transform a count matrix to a normalized tf or tf-idf representation.

```
tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
```

**Use text processing pipeline to create training and test datasets**

```
pipeline = text_processing_pipeline(features=10000)
X_train = pipeline.fit_transform(train.data)
y_train = train.target

X_test = pipeline.transform(test.data)
y_test = test.target
```

**multi-class classification**


**define classifier with sklearn LogisticRegression**
```
clf = LogisticRegression(random_state=0)
```

**fit classifier to training set**
```
clf = clf.fit(X_train, y_train)
```

**get predictions for test set**
```
pred = clf.predict(X_test)

score = accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score)

f1 = f1_score(y_test, pred, average='weighted')
print("      F1:   %0.3f" % f1)
```

**The confusion matrix**

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Natural-Language-Processing/2.png?raw=true)

**Document classification with word embeddings**

```
import pickle
from pathlib import Path
```

```
embeddings_path = Path().cwd() / '..' / '..' / '..' / 'coursedata' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p'

with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)
    vocabulary = list(embeddings.keys())
    
print(f'The vocabulary has a total of {len(vocabulary)} words')
```

**Extract a training and validation split**
```
validation_split = 0.2
num_validation_samples = int(validation_split * len(train.data))

train_samples = train.data[:-num_validation_samples]
val_samples = train.data[-num_validation_samples:]

train_labels = train.target[:-num_validation_samples]
val_labels = train.target[-num_validation_samples:]

test_samples = test.data
test_labels = test.target
```

## Using CNN to perform document classification

```
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val   = vectorizer(np.array([[s] for s in val_samples])).numpy()
x_test  = vectorizer(np.array([[s] for s in test_samples])).numpy()

y_train = train_labels
y_val   = val_labels
y_test  = test_labels
```

The vocabulary size, or the number of words in the vocabulary built by Tokenizer

```
vocabulary_size = 20000

print(vocabulary_size)
```

The vocabulary has a total of 20000 words

Each document is of length 500 tokens.

The embedding vectors of length 300.

```
print(f"Training set shape: {x_train.shape}")
print(f"Validation set shape: {x_val.shape}")
print(f"Test set shape: {x_test.shape}")

x_train_emb = embedding_layer(x_train)
```

**The number of categories for classification**
```
m = len(categories)

print(m)

print(x_train_emb.shape)

model = keras.Sequential([
    embedding_layer,
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv1"),
    layers.MaxPool1D(pool_size=2, name="maxpool1"),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv2"),
    layers.MaxPool1D(pool_size=2, name="maxpool2"),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv3"),
    layers.GlobalMaxPool1D(name="globalmaxpool"),
    layers.Dense(128, activation="relu", name="dense"),
    layers.Dropout(0.5),
    layers.Dense(m, activation='softmax', name="output")
])

model.summary()
```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Natural-Language-Processing/3.png?raw=true)

```
training=True
```
**Compile the model**

``` 
model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
```
**Training the model**

```
if training:
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20, verbose=1)
    model.save('model.h5')
else: 
    model = tf.keras.models.load_model("model.h5")

model = tf.keras.models.load_model("model.h5")

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy {:.2f}".format(test_acc))
```

**An end-to-end model**

**keras layer that takes a string as an input**
```
string_input = keras.Input(shape=(1,), dtype="string")
```
**vectorize string input**
```
x = vectorizer(string_input)
```
**pass the vector to main model**

```
preds = model(x)

end_to_end_model = keras.Model(string_input, preds)
end_to_end_model.summary()

y_pred = [np.argmax(prob) for prob in end_to_end_model.predict(test.data)]

score = accuracy_score(y_test, y_pred)
print("Accuracy:   %0.3f" % score)

f1 = f1_score(y_test, y_pred, average='weighted')
print("      F1:   %0.3f" % f1)

from sklearn.metrics import multilabel_confusion_matrix
cfs_matrix = multilabel_confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    
for axes, cfs, label in zip(ax.flatten(), cfs_matrix, train.target_names):
    print_confusion_matrix(cfs, axes, label, ["N", "P"])

fig.tight_layout()
plt.show()
```

![alt text](https://github.com/jylhakos/Deep-Learning-with-Python/blob/main/Natural-Language-Processing/4.png?raw=true)
