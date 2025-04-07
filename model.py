import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    # Simple clean_text function; you can enhance it as needed.
    return str(text).lower().strip()

class Recommender:
    def __init__(self, csv_path):
        try:
            # Load CSV with ISO-8859-1 encoding to handle non-UTF-8 characters.
            self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            self.df = pd.DataFrame()  # Safe fallback in case of error

        if not self.df.empty:
            # Clean the text in the specified columns if they exist.
            for col in ['Knowledge, Skills, Abilities', 'Relevant Job Roles']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('').apply(clean_text)
                else:
                    print(f"[WARNING] Column '{col}' not found in CSV.")

            # Combine the text from both columns for TF-IDF processing.
            self.df['combined'] = self.df['Relevant Job Roles'] + " " + self.df['Knowledge, Skills, Abilities']

            # Initialize and fit the TF-IDF vectorizer on the combined text.
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined'])

    def recommend(self, user_input, top_n=6):
        if self.df.empty:
            print("[ERROR] DataFrame is empty. Cannot perform recommendation.")
            return []
        
        # Transform the user input into the same vector space.
        user_vec = self.vectorizer.transform([user_input])
        cosine_sim = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        
        # Get the indices of the top N matching rows.
        top_indices = cosine_sim.argsort()[::-1][:top_n]
        
        # Define the columns you want to return.
        cols_to_return = ['Test', 'time(min)', 'Test Type', 'Remote Testing', 'Adaptive/IRT Support']
        missing_cols = [col for col in cols_to_return if col not in self.df.columns]
        
        # If any of the specified columns are missing, return the full row data.
        if missing_cols:
            print(f"[WARNING] Columns {missing_cols} not found in CSV. Returning complete rows.")
            results = self.df.iloc[top_indices].to_dict(orient="records")
        else:
            results = self.df.iloc[top_indices][cols_to_return].to_dict(orient="records")
        
        return results
