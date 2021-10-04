from GGO import *
import pickle


sp.init_printing()

with open('./template.pickle', 'rb') as f:
    template = pickle.load(f)
