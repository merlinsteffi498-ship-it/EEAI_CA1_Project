import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame, target_col: str = Config.CLASS_COL) -> None:
        """
        Encapsulates the data and performs the train-test split.
        """
        # Store the full embeddings and target variable
        self.embeddings = X
        self.y = df[target_col].values

        # Split the data into 80% training and 20% testing 
        indices = np.arange(len(df))
        self.X_train, self.X_test, self.y_train, self.y_test, idx_train, idx_test = train_test_split(
            self.embeddings, self.y, indices, test_size=0.2, random_state=seed
        )
        
        # 3. Store the split dataframes
        self.train_df = df.iloc[idx_train]
        self.test_df = df.iloc[idx_test]


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df


