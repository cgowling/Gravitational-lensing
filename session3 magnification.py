# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:57:00 2016

@author: ppycago
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
from scipy import sparse
import numpy as np

plt.close()
N = 10
data = np.array([[1,3,5,7,9,11], [2,4,6,8,10], [2,4,6,8,10],[3,5,7,9],[3,5,7,9],[4,6,8],[4,6,8],[5,7][5,7][6,][6]])
diags = np.array([0, -1, 1,-2,2,-3,3,-4,4,-5,5,-6,6])
sparse.spdiags(data, diags, 4, 4).toarray()