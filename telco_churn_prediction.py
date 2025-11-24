# [Beginner Note] These are essential libraries for data analysis and working with machine learning models.
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd  # For data frames, like Excel tables in Python
import numpy as np   # For numbers and math, helps with arrays and calculations
import matplotlib.pyplot as plt  # For drawing charts/plots
import seaborn as sns            # For nicer statistical plots

# [Beginner Note] These help with splitting data, scaling numbers, training and testing models.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import joblib  # For saving models and reusable objects

# [Beginner Note] TensorFlow is a popular library for building neural networks (models that learn by finding patterns).
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Try to import SMOTE, which helps balance uneven datasets (optional)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

RANDOM_STATE = 42  # [Beginner Note] Ensures results are the same each time you run the notebook ("random seed").
TEST_SIZE = 0.2    # 20% of data will be saved for testing later

print('Libraries imported!')
