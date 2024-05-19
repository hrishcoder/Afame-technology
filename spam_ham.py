import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess(text):
    messages=text.apply(word_tokenize)
    corpus=[]
    lemmatizer=WordNetLemmatizer()
    for tokens in messages:
         # Initialize list to store stemmed words for each message
        lemmatize_words = []
        # Iterate over each word in the tokenized message
        for word in tokens:
            # Apply stemming and remove stopwords
            if word.lower() not in stopwords.words('english') and re.match('[a-zA-Z]+', word):
                lemmatized_word = lemmatizer.lemmatize(word.lower())
                lemmatize_words.append(lemmatized_word)
        # Join the stemmed words back into a string and append to the corpus
        corpus.append(' '.join(lemmatize_words))

    # Print the stemmed corpus
    return corpus


    

def vectorize(messages,max_features=5000):
    #we get the corpus
    pre_text=preprocess(messages)
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    vect_text=tfidf_vectorizer.fit_transform(pre_text)
    joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.joblib')
    return vect_text


df = pd.read_csv(r'C:\Users\HRISHAB\Documents\A.I_projects\spam ham\data\spam.csv', encoding='ISO-8859-1')

def map_sentiments(results):
    if results == 'ham':
        return 0
    elif results == 'spam':
        return 1
   

# Check for null values in each column
null_columns = df.columns[df.isnull().any()]


# Delete columns with null values
df = df.drop(null_columns, axis=1)

#rename columns
df.rename(columns={'v1': 'results', 'v2': 'messages'}, inplace=True)
df['results']=df['results'].map(map_sentiments)

print(df['messages'])
X=vectorize(df['messages'])
y=df['results']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Apply SMOTETomek to the training data
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts())

# Define the SVM classifier
svm_classifier = SVC()

# Define the classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

# # Define the hyperparameters grid for GridSearchCV
# param_dist = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
#     'degree': [2, 3, 4, 5]  # for poly kernel
# }
# # Create GridSearchCV object
# Random_search = RandomizedSearchCV(svm_classifier, param_dist, cv=5, scoring='accuracy')
# # Train the model on the balanced training data
# Random_search.fit(X_train_resampled, y_train_resampled)
rf_classifier.fit(X_train_resampled, y_train_resampled)
y_pred = rf_classifier.predict(X_test)


# Predictions on test data
# y_pred = Random_search.predict(X_test)
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#save the model
joblib.dump(rf_classifier,'SVC_classifier.pkl')




