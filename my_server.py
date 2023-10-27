from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import PolynomialFeatures
import joblib
import pandas as pd
import cv2
import numpy as np
import os

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

# http://127.0.0.1/image
@app.route('/image', methods=['POST'])
@cross_origin(origins='*', headers=['Content-Type', 'Authorization'])

def upload_image():
    image_file = request.files['image']
    height = request.form['height']
    weight = request.form['weight']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', gray_image)
    # cv2.imshow('Gray Image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if os.path.exists('gray_image.jpg'):
      try:
          os.remove('gray_image.jpg')
      except Exception as e:
          print(f"Lỗi xảy ra khi xóa tệp: {e}")
    return jsonify({
       'status': 'success'
    })

# http://127.0.0.1/predict

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
    #  'data': {
    #     **feature,
    #     **predictions
    #  }
    'data': {
      'volumetric': {
        'Bust grith': 94.5,
        'Upper chest girth': 96.3,
        'Waist girth': 88.9
       }, 
      'linear': {
        'Neck to upper hip length': 52.9,
        'Outside leg length': 116.7
      }
    }
  }

  return make_response(jsonify(res), 200)

# Start Backend
if __name__ == '__main__':
  app.run(host='0.0.0.0', port='6868')