# Iris Prediction API using Flask

This Flask app allows you to make predictions using a pre-trained Random Forest model on the Iris dataset. You can use two methods to make predictions:

1. **Predict Iris Class Using Query Parameters (GET method)**
2. **Predict Iris Class Using CSV File Upload (POST method)**

## How to Use

### 1. **Predict Iris Class Using Query Parameters (GET Method)**

Make a prediction by passing the feature values as query parameters in the URL.

#### Example Request:

```bash
http://127.0.0.1:7000/predict?s_length=5.1&s_width=3.5&p_length=1.4&p_width=0.2
```

#### Parameters:

- `s_length`: Sepal length
- `s_width`: Sepal width
- `p_length`: Petal length
- `p_width`: Petal width

#### Example Response:

```json
"0"
```

---

### 2. **Predict Iris Class Using CSV File Upload (POST Method)**

Upload a CSV file containing feature values to make predictions for multiple samples at once.

#### Example Request (using `curl`):

```bash
curl -X POST -F "input_file=@iris_data.csv" http://127.0.0.1:7000/predict_file
```

#### CSV File Format:

Your CSV file should contain rows of feature values (sepal length, sepal width, petal length, petal width) without a header.

#### Example Response:

```json
[0, 0, 1, 2]
```
