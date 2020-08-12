import pandas as pd 
import pickle


clusters = pd.read_csv('/home/mahdi/Downloads/result.csv')

with open('features.pkl','rb') as f:
    features = pickle.load(f)
