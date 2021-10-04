from GGO import *
import pickle


sp.init_printing()

with open('./template.pickle', 'rb') as f:
    template = pickle.load(f)


cs = ConfigurationSpace(3, template)

internal_coordinate_chart = {
    'distance_functions' : [(0,2),(1,2)],
    'angle_functions' : [(1,0,2)],
    'moving_frame' : (0,1,2)
    }

point = {
    0 : [1,0,0],
    1 : [0,1,0],
    2 : [0,0,1/2]
    }
