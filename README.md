[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)

# Description
Estimate car price following fake data from Kaggle competition
Data can be found on Kaggle website : https://www.kaggle.com/competitions/playground-series-s4e9


# Results
![image](https://github.com/user-attachments/assets/3c3061f8-3327-4fef-b75a-8e8e27ab7bbd)
![image](https://github.com/user-attachments/assets/8f1770f3-bbf4-4440-a0ea-3c9cdbc51a02)
![image](https://github.com/user-attachments/assets/cbc3e9fc-1354-459b-9b34-73dad002ceac)

<img src="![image](https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/completion.JPG)" alt="Alt Text" width="300" height="200">


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
