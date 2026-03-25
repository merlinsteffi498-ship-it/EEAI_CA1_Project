#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    df = get_input_data()
    return  df

def preprocess_data(df):
    df =  de_duplication(df)                                                             # De-duplicate input data
    df = noise_remover(df)                                                               # remove noise in input data
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())      # translate data to english
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame, target_col: str):
    return Data(X, df, target_col)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

# Code will start executing from following line
if __name__ == '__main__':
    
    # Data pre-processing
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Data transformation
    X, group_df = get_embeddings(df)
    
    
    # --- DESIGN CHOICE 1: CHAINED MULTI-OUTPUTS ---
    
    # 1. Create the chained target variables in the dataframe
    df['chain_1'] = df['y2'].astype(str)
    df['chain_2'] = df['y2'].astype(str) + "_" + df['y3'].astype(str)
    df['chain_3'] = df['y2'].astype(str) + "_" + df['y3'].astype(str) + "_" + df['y4'].astype(str)
    
    # 2. Define the chains we want to iterate through
    target_chains = ['chain_1', 'chain_2', 'chain_3']
    
    # 3. Loop through each chain, encapsulate the data, and run the model
    for target in target_chains:
        print(f"\n{'='*40}")
        print(f"Executing Pipeline for Target: {target}")
        print(f"{'='*40}")
        
        # Encapsulate data specifically for this chained target
        data = get_data_object(X, df, target)
        
        # Execute modelling
        perform_modelling(data, df, 'RandomForest')

