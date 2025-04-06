# model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding="latin1")

        # Fill missing text fields
        self.df['Relevant Job Roles'] = self.df['Relevant Job Roles'].fillna('')
        self.df['Knowledge, Skills, Abilities'] = self.df['Knowledge, Skills, Abilities'].fillna('')

        # Create combined text for TF-IDF
        self.df['combined'] = self.df['Relevant Job Roles'] + " " + self.df['Knowledge, Skills, Abilities']

        # Fit the vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined'])

    def recommend(self, user_input, top_n=6):
        user_vec = self.vectorizer.transform([user_input])
        cosine_sim = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[::-1][:top_n]
        
        # Return the important columns like in notebook
        cols_to_return = [
            'Test', 'time(min)', 'Test Type',
            'Remote Testing', 'Adaptive/IRT Support'
        ]
        results = self.df.iloc[top_indices][cols_to_return].to_dict(orient="records")
        return results
