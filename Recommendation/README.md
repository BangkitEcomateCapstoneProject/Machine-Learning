# Article Recommendation
This recommendation model uses TF-IDF (Term Frequency Inverse Document Frequency) calculations and Cosine Similarity to compare the vectors between user input and the content of pre-processed articles. For preprocessing, such as stemming and stopwords removal, the NLTK library version 3.8.1 is used, as the articles are in English. The model is then implemented using the Flask library for API calls.

# Keypoints
- The model uses TF-IDF to convert the text data into numerical vectors.
- Cosine similarity is used to calculate the similarity between the user input and the article content.
- The model recommends the top 5 most similar articles based on the user input.
- The articles used for recommendations are English articles stored in a dataframe with an attribute 'content-preprocessed', which contains the article content that has undergone cleaning, stemming, and stopwords removal to minimize execution time.
- The NLTK library is used for preprocessing, such as stemming and stopwords removal.

# Step by steps on building the model
1. **Import Libraries** : Import libraries that are needed
```python 
import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
``` 
2. **Download Necessary NLTK data** : Download nltk data that used for stopwords and stemming method 
```python 
nltk.download('punkt')
nltk.download('stopwords')
```
3. **Create Python Class**:  Create the python class that have the necessary method for preprocessing and recommendation (detailed class documentation can be found in Recommendation.py).
    ```python 
    class ArticleRecommendation:
        def __init__(self):
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self.vectorizer = TfidfVectorizer(stop_words=list(self.stopwords),analyzer='word',lowercase=True,ngram_range(1,3),min_df=1,use_idf=True)
    
        def stopwords_remove(self,text):
            words = nltk.word_tokenize(text)
            stopword_removed = " ".join(word for word in words if word.lower() not in self.stopwords)
            return stopword_removed
    
        def stem(self,text):
            words = nltk.word_tokenize(text)
            stemmed_text = " ".join([self.stemmer.stem(word) for word in words])
            return stemmed_text
    
        def preprocessed(self,text):
            text = text.strip()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = self.stopwords_remove(text)
            text = self.stem(text)
            text = text.strip()
            return text
    
        def recommendation(self,dataframe_path,user_input):
            df = pd.read_csv(dataframe_path)
            user_input = self.preprocessed(user_input)
            data_vector = self.vectorizer.fit_transform(df['content_preprocessed']).toarray()
            user_vector = self.vectorizer.transform([user_input]).toarray()
            cos_sim = cosine_similarity(data_vector, user_vector)
            df['cos_sim'] = cos_sim[:, 0]  # Assigning the similarity score to a new column
            result = df.sort_values("cos_sim", ascending=False).head(5)
            return result[['Kode', 'title', 'raw_content', 'clean_content', 'date_created', 'author', 'articleLink','imageSrc']]
    ```
4. **Model Testing** : Run the model by importing classes and call the recommendation method. (Demonstrated recommendation.py)
    ```python 
    from Recommendation import ArticleRecommendation
    model = ArticleRecommendation()
    result = model.recommendation(data_path,'how to recycle plastics i have many plastics')
    ```

5. **API Implementation**: Implement the model using the Flask library to create an API that accepts user
input and returns the top 5 recommended articles based on the cosine similarity between the user input and the
article content. (detailed information can be found in app.py)

    ```python 
    from flask import Flask, request, jsonify
    from Recommendation import ArticleRecommendation
    from flask import Response
    from dotenv import load_dotenv
    import os

    load_dotenv()
    app = Flask(__name__)


    @app.route("/")
    def index():
        return "Hello World"


    @app.route("/recommend", methods=["POST"])
    def recommend():
        input_user = request.get_json(force=True)
        if "text" not in input_user:
            return (
                jsonify(error="Bad Request", message="JSON must contain 'text' attribute"),
                400,
        )
        text = input_user["text"]
        model = ArticleRecommendation()
        result = model.recommendation(
        os.environ.get("DATA_PATH"),
        text,
    )
    return result.to_json(orient="records", lines=True)


    if __name__ == "__main__":
    app.run(debug=True)
    ```


