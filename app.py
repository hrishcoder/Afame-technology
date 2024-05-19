from flask import Flask, render_template,request
import joblib

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app=Flask(__name__)
# Load the pickled TF-IDF vectorizer
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the model from the previous file
model = joblib.load('SVC_classifier.pkl')

nltk.download('punkt')
nltk.download('stopwords')


#function for preprocessing
def preprocessing(message):

    messages=word_tokenize(message)
    corpus=[]
    lemmatizer=WordNetLemmatizer()
    
    lemmatize_words = []
     # Iterate over each word in the tokenized message
    for word in messages:
            # Apply stemming and remove stopwords
            if word.lower() not in stopwords.words('english') and re.match('[a-zA-Z]+', word):
                lemmatized_word = lemmatizer.lemmatize(word.lower())
                lemmatize_words.append(lemmatized_word)
        # Join the stemmed words back into a string and append to the corpus
    corpus.append(' '.join(lemmatize_words))

    # Print the stemmed corpus
    return corpus




def predict_data(user_data,model,loaded_tfidf_vectorizer):
    clean_text=preprocessing(user_data)
    print(clean_text)
    clean_text = " ".join(clean_text)
    #vectorize the cleaned text
    vectorized_input=loaded_tfidf_vectorizer.transform([clean_text])
    #give the vectorized text to the model
    prediction=model.predict(vectorized_input)
    #return the result
    return prediction[0]

    

@app.route('/',methods=['GET','POST'])
def index():
    #if post
    if request.method=='POST':

        #get the data from the input box
        user_input=request.form.get('user_text')

        #give the data and the model to  predict_data function
        result=predict_data(user_input,model,loaded_tfidf_vectorizer)

        #print the result
        print("Result:", result)

        #display that result in index.html
        return render_template('index.html',Result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)