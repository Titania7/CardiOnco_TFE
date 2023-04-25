# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:39:21 2023

@author: Tatiana Dehon
"""

import numpy as np


#%%
class SingleGraph:
    
    # Protected variables
    _title = None
    _x = None
    _y = None
    _samplerate = None
    _step = None
    _cali = None
    _xlim = None
    _ylim = None
    
    # Class initialization
    def __init__(self, x, y, title = 'New Graph', samplerate = 0, step = 0, calibrage = [0,0], xlim = [0,0], ylim = [0,0]):
        self._title = title
        self._x = x
        self._y = y
        self._samplerate = samplerate
        self._step = step
        self._cali = calibrage
        self._xlim = xlim
        self._ylim = ylim
    
    def __str__(self):
        return f"Title : {self._title}\nSamplerate(or step) : {self._samplerate}({self._step})\nx-limit values : {self._xlim}\ny-limit values : {self._ylim}\nCalibration [A, B] (Ay+B) : {self._cali}"
    
    def set_title(self, myTitle):
        self._title = myTitle
    
    def set_x(self, xarray):
        self._x = xarray
        self._xlim = [min(xarray), max(xarray)]
    
    def set_y(self, ylist):
        self._y = ylist
        self._ylim = [min(ylist), max(ylist)]

#%%

"""
x = np.arange(0,10, 1)
y = [0,1,2,3,4,5,6,7,8,9]


a = SingleGraph(x, y)
#a = SingleGraph(x, y, 'Mon graphique', 100, 1, [4,3], [0,1], [-2,2])
"""
