# -*- coding: utf-8 -*-
"""testsuite1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eijSUQYBqgRSsGD_m08VTgXU2MEVfNnB

# Test spot detection

This is a test suite.
"""

import mlflow
import numpy
import sys
import os

"""Prepare for generating inputs."""
step3inputval1 = int(sys.argv[1]) if len(sys.argv) > 1 else 3

"""Set physical parameters."""

with mlflow.start_run():
    
    mlflow.log_param("step3inputval1", step3inputval1)

    mlflow.log_metric("sub3", step3inputval1-3)
