import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Converts textual data into a numeric representation using TF-IDF.
    """
    print("Generating TF-IDF embeddings...")
    # Initialize the vectorizer. We limit to 5000 features to keep it efficient.
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Combine the Ticket Summary and Interaction Content for richer context
    combined_text = df[Config.TICKET_SUMMARY].astype(str) + " " + df[Config.INTERACTION_CONTENT].astype(str)
    
    # Learn the vocabulary and return the document-term matrix
    X = vectorizer.fit_transform(combined_text).toarray()
    
    return X#Methods related to converting text in into numeric representation and then returning numeric representation may go here