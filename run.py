from sql_funcs import etl
import pandas as pd
import numpy as np 
from tqdm import tqdm
import sqlite3
from model import Modeling
etl = etl()
cnxn = sqlite3.connect('clv.db')
query = "SELECT * FROM df1 LIMIT 100"
df = pd.read_sql(query, cnxn)
cnxn.close()

dtype_sorted = {
    'IMEI': 'int',
    'Total_Duration': 'int',
    'MaxDuration': 'int',
    'SameDeviceQty': 'int',
    'ID': 'float',
    'Purchasing_Behaviour': 'int',
    'CAMERA_MATRIX': 'int',
    'FRONTCAM_MATRIX': 'int',
    'PIXEL_DENSITY': 'int',
    'OS': 'str',
    'PhoneType': 'str',
    'BRAND': 'str',
    'MODEL': 'str',
    'SUPPORTS_VOLTE': 'bool',
    'SUPPORTS_VOWIFI': 'bool',
    'SUPPORTS_NFC': 'bool',
    'SUPPORTS_HTML5': 'str',
    'BATTERY_CAPACITY': 'str',
    'DATA_ONLY': 'bool',
    'SCREEN_RESOLUTION': 'str',
    'SUPPORTS_ESIM': 'str',
    'Min_END_DATE': 'str',  
    'Max_END_DATE': 'str',  
    'START_DATE': 'str',     
    'id0' : 'int',
    'recency': 'int',
    'recency_score': 'int',
    'frequency_score': 'int',
    'monetary_score': 'int'
}


modeler = Modeling(df)
modeler.run_all_models()