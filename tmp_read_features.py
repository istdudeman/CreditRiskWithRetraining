import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('__file__')))
from preprocessing.feature_selection import WoETransformer

woe = joblib.load("ModelTraining/woe_transformer.pkl")
print("Features:")
print(woe.get_selected_features())
