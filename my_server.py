from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import PolynomialFeatures
import joblib
import pandas as pd
from io import BytesIO
import base64
from PIL import Image
import json
from human_detector import *

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
   # Handle image from base64 to .jpg
   image = request.form.get('imageUser')
   info =  json.loads(request.form.get('infoUser'))
   w = float(info.get('Weight'))
   h = float(info.get('Height'))

   base64_data = image.split(",")[1]
   # image_data = base64.b64decode(base64_data)
   # image = Image.open(BytesIO(image_data))
   # image.show()


   linear, volumetric = predict2D(base64_data, h, w)
   res_linear = []
   res_volumetric = []
   for key, value in linear.items():
      res_linear.append({
         'title': key,
         'value': value
      })
   for key, value in volumetric.items():
      res_volumetric.append({
         'title': key,
         'value': value
      })
   # Config res
   res = {
      'status': 200,
      'data': [
         {
            'type': 'linear',
            'statistics': res_linear
         },
         {
            'type': 'volumetric',
            'statistics': res_volumetric
         }
      ]
    }

   return jsonify(
      res
   )

# # http://127.0.0.1/predict

# @app.route('/predict', methods=['POST'])
# @cross_origin(origins='*', headers=['Content-Type', 'Authorization'])

# def predict():
#   data = request.get_json(force=True)
#   feature = data['feature']
#   input = poly.fit_transform(pd.DataFrame.from_dict(feature, orient='index').T)

#   predictions = {}

#   for model_name, model in LR_models.items():
#     prediction = model.predict(input)
#     predictions[model_name] = prediction[0]

#   print(predictions)
#   res = {
#     'status': 'success',
#     #  'data': {
#     #     **feature,
#     #     **predictions
#     #  }
#     'data': {
#       'volumetric': {
#         'Bust grith': 94.5,
#         'Upper chest girth': 96.3,
#         'Waist girth': 88.9
#        }, 
#       'linear': {
#         'Neck to upper hip length': 52.9,
#         'Outside leg length': 116.7
#       }
#     }
#   }

#   return make_response(jsonify(res), 200)

# Start Backend
if __name__ == '__main__':
  app.run(host='0.0.0.0', port='6868')