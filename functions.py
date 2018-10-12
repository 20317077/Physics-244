# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:12:18 2018

@author: 20317077
"""

import numpy as np


def trig(tsample, params):
    terms = np.size(params)
    y = np.transpose([None] * terms)
    y[0] = params[0] * np.ones(np.size(tsample))
    for i in range(1, terms, 2):
        y[i] = params[i] * np.sin(i * tsample)
        y[i + 1] = params[i + 1] * np.cos(i * tsample)
    return y


def poly(tsample, params):
    terms = np.size(params)
    y = [None] * terms
    y[0] = params[0] * tsample**0
    for i in range(1, terms, 2):
        y[i] = params[i] * tsample**i
        y[i + 1] = params[i + 1] * tsample**(i + 1)
    return y


def custom_trig(tsample, params):
    terms = np.size(params)
    y = [None] * terms
    y[0] = params[0] * np.ones(np.size(tsample))
    for i in range(1, terms, 2):
        y[i] = params[i] * np.sin((1 - 1.0 / i) * tsample)
        y[i + 1] = params[i + 1] * np.cos((1 - 1.0 / i) * tsample)
    return y
