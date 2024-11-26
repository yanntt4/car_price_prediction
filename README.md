![Python 3.12](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Tensorflow 2.17.0](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![Conda](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)
![Flask 3.0.3](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Javascript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![Numpy 1.26.4](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas 2.2.2](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn 1.5.1](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Description
Estimate car price following fake data from Kaggle competition
Data can be found on Kaggle website : https://www.kaggle.com/competitions/playground-series-s4e9


# Results
<img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/preparation.JPG" alt="Alt Text" width="200" height="133"><img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/completion.JPG" alt="Alt Text" width="200" height="133"><img src="https://github.com/yanntt4/car_price_prediction/blob/main/readme_photo/result.JPG" alt="Alt Text" width="200" height="133">


# Usage
Application can be used for two main usages :
- Analyse data to create machine learning/deep learning model to estimate car price. 6 algorithms can be used to create model : Random Forest, Gradient Boosting, Neural Network, Extrem Gradient Boosting, Cat Boosting and Light Boosting. A spcific github page is available to present the model
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
