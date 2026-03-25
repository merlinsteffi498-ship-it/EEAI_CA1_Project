from model.randomforest import RandomForest

def model_predict(data, df, name):
    """
    Instantiates the model, trains it, and generates predictions.
    """
    # Instantiate the model using the methods defined in your base.py
    print(f"Initializing {name}...")
    model = RandomForest(name, data.get_embeddings(), data.get_type())
    
    model.train(data)
    model.predict(data.get_X_test())
    model_evaluate(model, data)


def model_evaluate(model, data):
    """
    Prints the classification report.
    """
    model.print_results(data)