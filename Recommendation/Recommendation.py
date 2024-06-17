import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class ArticleRecommendation:
    def __init__(self):
        """
        Initializes the ArticleRecommendation class with necessary tools for text preprocessing and vectorization.
        """
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=list(self.stopwords), analyzer='word', lowercase=True, ngram_range=(1, 3), min_df=1, use_idf=True)

    def stopwords_remove(self, text):
        """
        Removes stopwords from the given text.

        Parameters:
        text (str): The input text from which stopwords are to be removed.

        Returns:
        str: The text after stopwords have been removed.
        """
        words = nltk.word_tokenize(text)
        stopword_removed = " ".join(word for word in words if word.lower() not in self.stopwords)
        return stopword_removed

    def stem(self, text):
        """
        Stems the words in the given text.

        Parameters:
        text (str): The input text to be stemmed.

        Returns:
        str: The text after stemming.
        """
        words = nltk.word_tokenize(text)
        stemmed_text = " ".join([self.stemmer.stem(word) for word in words])
        return stemmed_text

    def preprocessed(self, text):
        """
        Preprocesses the given text by removing non-alphabetic characters, stopwords, and stemming the words.

        Parameters:
        text (str): The input text to be preprocessed.

        Returns:
        str: The preprocessed text.
        """
        text = text.strip()
        text = re.sub("[^a-zA-Z\s]", "", text)  # Allow spaces and letters
        text = self.stopwords_remove(text)
        text = self.stem(text)
        text = text.strip()
        return text

    def recommendation(self, dataframe_path, user_input):
        """
        Generates article recommendations based on the similarity between user input and preprocessed articles in the dataset.

        Parameters:
        dataframe_path (str): The file path to the CSV file containing the dataset.
        user_input (str): The user's input text for which article recommendations are to be generated.

        Returns:
        DataFrame: A DataFrame containing the top 5 recommended articles, sorted by cosine similarity. 
                   The DataFrame includes the following columns: 'Kode', 'title', 'raw_content', 'clean_content', 
                   'date_created', 'author', 'articleLink', and 'imageSrc'.
        """
        # Read the dataset
        df = pd.read_csv(dataframe_path)
        
        # Preprocess the user input
        user_input = self.preprocessed(user_input)
        
        #Vectorize the preprocessed dataset content
        data_vector = self.vectorizer.fit_transform(df['content_preprocessed']).toarray()
        
        #Vectorize the user input
        user_vector = self.vectorizer.transform([user_input]).toarray()
        
        # Compute cosine similarity between the user input and dataset content
        cos_sim = cosine_similarity(data_vector, user_vector)
        
        # Assign the similarity score to a new column in the DataFrame
        df['cos_sim'] = cos_sim[:, 0]
        
        # Sort the DataFrame by the similarity score in descending order and select the top 5 results
        result = df.sort_values("cos_sim", ascending=False).head(5)
        
        # Return the relevant columns of the top 5 recommended articles
        return result[['Kode', 'title', 'raw_content', 'clean_content', 'date_created', 'author', 'articleLink','imageSrc']]

# Usage example:
# recommendation_system = ArticleRecommendation()
# result = recommendation_system.recommendation("path_to_your_dataframe.csv", "sample user input text")
# print(result)