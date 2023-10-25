from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import PolynomialFeatures
import joblib
import pandas as pd

poly = PolynomialFeatures(degree=2)
model_names = ['thigh', 'knee', 'ankle', 'biceps', 'forearm', 'wrist']
LR_models = {}
for model_name in model_names:
    model_filename = f'model_predict/{model_name}_model.joblib'
    model = joblib.load(model_filename)
    LR_models[model_name] = model

# Init Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# http://127.0.0.1/add

@app.route('/predict', methods=['POST'])
@cross_origin(origins='*', headers=['Content-Type', 'Authorization'])

def predict():
  data = request.get_json(force=True)
  feature = data['feature']
  input = poly.fit_transform(pd.DataFrame.from_dict(feature, orient='index').T)

  predictions = {}

  for model_name, model in LR_models.items():
    prediction = model.predict(input)
    predictions[model_name] = prediction[0]

  print(predictions)
  res = {
     'status': 'success',
     'data': {
        **feature,
        **predictions
     }
  }

  return jsonify({**res})

# Start Backend
if __name__ == '__main__':
  app.run(host='0.0.0.0', port='6868')