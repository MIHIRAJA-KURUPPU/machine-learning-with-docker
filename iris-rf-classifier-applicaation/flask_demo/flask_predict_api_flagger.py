import pickle
import numpy as np
from flask import Flask, request
from flasgger import Swagger
import pandas as pd

# Load the pre-trained model
with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """
    Predict Iris flower type from sepal/petal measurements.
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
        description: Sepal length
      - name: s_width
        in: query
        type: number
        required: true
        description: Sepal width
      - name: p_length
        in: query
        type: number
        required: true
        description: Petal length
      - name: p_width
        in: query
        type: number
        required: true
        description: Petal width
    responses:
      200:
        description: The predicted class
    """
    s_length = float(request.args.get("s_length"))
    s_width = float(request.args.get("s_width"))
    p_length = float(request.args.get("p_length"))
    p_width = float(request.args.get("p_width"))

    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction[0])

@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """
    Predict iris flower type from CSV file input.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
        description: CSV file with 4 features per row (no header)
    responses:
      200:
        description: The predicted classes
    """
    input_file = request.files.get("input_file")
    input_data = pd.read_csv(input_file, header=None)

    if input_data.shape[1] != 4:
        return "Error: Input file must have 4 columns.", 400

    prediction = model.predict(input_data)
    prediction = [int(p) for p in prediction]
    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)


#sample url: http://127.0.0.1:7000/predict?s_length=5.1&s_width=3.5&p_length=1.4&p_width=6
#sample url: http://127.0.0.1:7000/apidocs