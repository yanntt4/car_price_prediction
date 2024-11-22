[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)

# Description
Estimate car price following fake data from Kaggle competition
Data can be found on Kaggle website : https://www.kaggle.com/competitions/playground-series-s4e9


# Results
![image](https://github.com/user-attachments/assets/3c3061f8-3327-4fef-b75a-8e8e27ab7bbd)
![image](https://github.com/user-attachments/assets/8f1770f3-bbf4-4440-a0ea-3c9cdbc51a02)
![image](https://github.com/user-attachments/assets/cbc3e9fc-1354-459b-9b34-73dad002ceac)


# Usage
This application was designed specifically with CARGOWISE data extract. 
Nevertheless, it can be used with an excel table containing the following columns :
- "Shipment ID" (Shipment identification number)
- "Consol ID" (Container identificatio number)
- "Type" (Container type)
- "Master/Lead" (The master shipment can regroup several shipments)
- "Trans." (Transport method)
- "Origin" (Code for origin place)
- "Dest." (Code for destination place)
- "ETD" (Estimated time of departure)
- "ETA" (Estimated time of arrival)
- "Weight" (Ware weight)
- "UW" (Weight unit)
- "Volume" (Ware volume)
- "UV" (Volume unit)
- "Pic. Trn." (Company pickup transport name)
- "Dlv. Trn." (Company delivery transport name)
- "Pickup From Address" (adress for pickup)
- "Delivery To Address" (adress for delivery)
- "1st Load" (1st place where main transport starts)
- "Last Disc" (last place where main transport ends)

The table must be set in the folder `X/2024/source`

The program can be launched using the script *main.py*. The parameters can be modified using the class `Parameters()`.
The year inside the class `Parameters()` needs to correspond to the name of the folder. 

During execution, some messages are printed, giving indication on the execution. 
If data are missing, it must be added to the file `BDD.xlsx`


# Requirements
**pandas** : 2.2.2

**numpy** : 1.26.4

**matplotlib** : 3.9.1

**openpyxl** : 3.1.5


# Follow up
With the current version, the program uses `BDD.xlsx` to calculate emission. A method has been found to calculate main emission using any origin and destination.
It has not been impletemented yet.

For PRE and POST transport using ROAD or RAIL, no method is currently available to calculate emission from any origin or destination. 
