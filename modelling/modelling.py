from model.randomforest import RandomForest

def model_predict(data, df, name):
    """
    Instantiates the model, trains it, and generates predictions.
    """
    print(f"Initializing {name}...")
    # 1. Instantiate the model (using the methods defined in your base.py)
    model = RandomForest(name, data.get_embeddings(), data.get_type())
    
    # 2. Train the model
    model.train(data)
    
    # 3. Predict on the test set
    model.predict(data.get_X_test())
    
    # 4. Evaluate the results
    model_evaluate(model, data)

def model_evaluate(model, data):
    """
    Prints the classification report.
    """
    model.print_results(data)