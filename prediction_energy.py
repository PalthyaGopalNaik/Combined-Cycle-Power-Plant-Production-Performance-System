# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:35:57 2024

@author: palth
"""

import numpy as np
import pickle

load_model=pickle.load(open('D:/project_3_deployment/trained_model.sav','rb'))


input_data=[[12.04,42.34,1019.72,94.67]]
predicted_output=load_model.predict(input_data)
print('Predicted Energy Production:',predicted_output)