from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

knn = joblib.load('knn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sample = pd.DataFrame([data])
    for col, le in label_encoders.items():
        if col != 'class':
            sample[col] = le.transform(sample[col])
    # Certifique-se de que a amostra tem a forma correta
    if len(sample.shape) == 1:
        sample = pd.DataFrame([sample])
    else:
        sample = pd.DataFrame(sample)

    prediction = knn.predict(sample)
    result = label_encoders['class'].inverse_transform(prediction)
    if result[0] == 'p':
        resposta='venenoso'
    elif result[0] == 'e':
        resposta='comestivel'
    return jsonify({'class': resposta})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
