import pickle
import numpy as np
from flask import Flask, request
import pandas as pd

# Load the pre-trained model
with open('./rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict')
def predict_iris():
    try:
        # Retrieve the features from the query parameters and convert them to float
        s_length = float(request.args.get("s_length"))
        s_width = float(request.args.get("s_width"))
        p_length = float(request.args.get("p_length"))
        p_width = float(request.args.get("p_width"))

        # Make a prediction
        prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))

        # Return the prediction as a string
        return str(prediction[0])
    except Exception as e:
        return f"Error: {str(e)}", 400


@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    try:
        # Retrieve the uploaded CSV file and read it into a DataFrame
        input_file = request.files.get("input_file")
        input_data = pd.read_csv(input_file, header=None)

        # Ensure that the data has the right shape for prediction
        if input_data.shape[1] != 4:
            return "Error: Input file must have 4 columns.", 400

        # Make a prediction
        prediction = model.predict(input_data)

        # Convert numpy.int64 to Python int
        prediction = [int(p) for p in prediction]

        # Return the prediction as a string
        return str(prediction)
    except Exception as e:
        return f"Error: {str(e)}", 400


if __name__ == '__main__':
    app.run(port=7000)



#sample url: http://127.0.0.1:7000/predict?s_length=5.1&s_width=3.5&p_length=1.4&p_width=6