# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:05:03 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

delivery = pd.read_csv("F:\\R\\files\\delivery_time.csv")

# y is delivery time, x is sorting time

delivery.columns

delivery = delivery.rename(columns = {"Delivery Time" : "Dtime", "Sorting Time": "Stime"})
delivery.columns

np.corrcoef(delivery.Dtime, delivery.Stime) #coefficient is 82 percentage

#building a model

import statsmodels.formula.api as smf

model1 = smf.ols("Dtime~Stime", data = delivery).fit()

model1.summary() #0.682

model1.conf_int(0.05)
pred1 = model1.predict(delivery)

pred1.corr(delivery.Dtime) #predicted correlation is 82.5%

#model2

model2 = smf.ols("Dtime~ np.log(Stime)", data = delivery).fit()

model2.summary() #0.85
pred2 = model2.predict(delivery)
np.corrcoef(pred2, delivery.Dtime) # 83%

#model3
Stimesqr = delivery.Stime*delivery.Stime
model3 = smf.ols("Dtime ~ Stime+Stimesqr", data= delivery).fit()
model3.summary() #0.693

pred3 = model3.predict(delivery)
pred3.corr(delivery.Dtime) #83 %

finalmodel = model2

#inferences

