import pickle

from flask import Flask, request, jsonify

model_file = 'model1.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

dv_file = 'dv.bin'
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)   

app = Flask('marketing')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    marketing = y_pred >= 0.5

    result = {
        'Probability': float(y_pred.round(3)),
        'Marketing': bool(marketing)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


