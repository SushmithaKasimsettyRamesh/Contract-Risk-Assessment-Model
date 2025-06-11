import json
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def init():
    """
    This function is called when the container is initialized/revised.
    """
    global model, encoders
    # Load the saved model package
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'risk_score_model.pkl')
    package = joblib.load(model_path)
    
    # Extract model and encoders
    model = package['model']
    encoders = package['encoders']

def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame for easier manipulation
        import pandas as pd
        input_df = pd.DataFrame(data)
        
        # Apply the same transformations as during training
        if 'AGENT_CLEAN' in input_df.columns and 'AGENT_CLEAN' in encoders:
            input_df['AGENT_CLEAN'] = encoders['AGENT_CLEAN'].transform(input_df['AGENT_CLEAN'].astype(str))

        # Make prediction
        result = model.predict(input_df)
        
        # Return the prediction as JSON
        return json.dumps({"risk_score": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
