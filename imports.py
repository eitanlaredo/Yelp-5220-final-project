# Data management imports
import sqlite3
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix 

# Visualization and graphing
import matplotlib as plt

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Lasso Import
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sentiment imports (excluding sklearn)
import nltk
from textblob import TextBlob # Textblob
from transformers import pipeline # Bert package
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA # VADER
nltk.download('vader_lexicon')

