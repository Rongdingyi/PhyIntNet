import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
# print(sys.path)
# print(dir_name)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from sklearn.metrics import mean_squared_error
import numpy as np

def read_spar(filename):

    with open(filename, 'r') as f:
        line_ = f.readline()
        line = line_.strip()
        spl = line.split(',')
        arr = []
        for i in spl:
            arr.append(float(i))

    arr = np.array(arr)

    # print(arr)

    return arr

class Engine:
    def __init__(self):

        self.data_dir = '/data1/rong/code/Eyes/data'
        self.train_txt = '/data1/rong/code/Eyes/data/txts/train/main.txt'
        self.val_txt = '/data1/rong/code/Eyes/data/txts/val/main.txt'

    @classmethod
    def load_data(self, csv_dir, txt):

        baseline_files = []

        f = open(os.path.join(csv_dir, txt) , 'r')
        for line in f:
            line = line.strip()
            line = line.split(',')
            baseline_files.append([line[0],line[1]])   

        AL_filenames = [os.path.join(csv_dir, 'after', x[0], x[1], 'Len_paras', 'AL.txt') for x in baseline_files]
        spar_filenames = [os.path.join(csv_dir, 'spar', x[0], x[1]+'.txt') for x in baseline_files]

        sixALs = []
        spars = []

        for tar_index in range(len(AL_filenames)):
    
            AL_filename = AL_filenames[tar_index]
            with open(AL_filename, 'r') as fin:
                sixAL = float(fin.readline())
                sixAL = np.array(np.float32(sixAL))[np.newaxis]
                
            spar_filename = spar_filenames[tar_index]
            spar_arr = read_spar(spar_filename)
            
            sixALs.append(sixAL)
            spars.append(spar_arr)

        return np.array(spars), np.array(sixALs)

    def train_val(self):
        x_train, y_train = self.load_data(self.data_dir, self.train_txt)
        x_val, y_val = self.load_data(self.data_dir, self.val_txt)

        max_features=int(x_train.shape[1])
        model1 = RandomForestRegressor(n_estimators=50, max_features=max_features, random_state=0)
        model2 = LinearRegression()
        model3 = LinearSVR(epsilon=0, random_state=0, max_iter=5000)

        model1.fit(x_train, y_train[:, 0])
        model2.fit(x_train, y_train[:, 0])
        model3.fit(x_train, y_train[:, 0])

        y_pred1 = model1.predict(x_val)
        metric1 = mean_squared_error(y_val,y_pred1)
        print('RandomForestRegressor:', metric1)

        y_pred2 = model2.predict(x_val)
        metric2 = mean_squared_error(y_val,y_pred2)
        print('LinearRegression:', metric2)

        y_pred3 = model3.predict(x_val)
        metric3 = mean_squared_error(y_val,y_pred3)
        print('LinearSVR:', metric3)

        # print(y_pred3)

        # return metric


if __name__ == '__main__':

    # x, y = Engine.load_data('/data1/rong/code/Eyes/data', '/data1/rong/code/Eyes/data/txts/val/main.txt')
    # print(x.shape, y.shape)
    engine = Engine()
    engine.train_val()

