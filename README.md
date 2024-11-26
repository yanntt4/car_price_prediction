[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)
![Tensorflow 2.17.0](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white]

# Description
Estimate car price following fake data from Kaggle competition
Data can be found on Kaggle website : https://www.kaggle.com/competitions/playground-series-s4e9


# Results
<img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/preparation.JPG" alt="Alt Text" width="300" height="200"><img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/completion.JPG" alt="Alt Text" width="300" height="200"><img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/result.JPG" alt="Alt Text" width="300" height="200">


# Usage
Application can be used for two main usages :
- Analyse data to create machine learning/deep learning model to estimate car price. 6 algorithms can be used to create model : Random Forest, Gradient Boosting, Neural Network, Extrem Gradient Boosting, Cat Boosting and Light Boosting
- Allow user to make prediction from a webpage


# Requirements
**pandas** : 2.2.2

**numpy** : 1.26.4

**matplotlib** : 3.9.1

**openpyxl** : 3.1.5


# Follow up
With the current version, the program uses `BDD.xlsx` to calculate emission. A method has been found to calculate main emission using any origin and destination.
It has not been impletemented yet.

For PRE and POST transport using ROAD or RAIL, no method is currently available to calculate emission from any origin or destination. 
