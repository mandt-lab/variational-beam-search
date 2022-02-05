import os
import sys
import numpy as np

def get_malware_dataset(valid=False):

    def log_odds_vec(p):
        # convert a probability to log-odds. Feel free to ignore the "divide by
        # zero" warning since we deal with it manually. The extreme values are
        # determined by looking at the histogram of the first-month data such
        # that they do not deviate too far from the others
        ind_0 = np.where(p == 0)[0]
        ind_1 = np.where(p == 1)[0]
        logodds = np.log(p) - np.log(1.-p)
        logodds[ind_0] = -5
        logodds[ind_1] = 4
        return logodds

    def read_data(valid, folder_name='./malware-dataset/'):
        if valid:
            # first month
            xs = np.load(folder_name + '/X_val.npy')
            ys = np.load(folder_name + '/y_val.npy')
        else:
            xs = np.load(folder_name + '/X.npy')
            ys = np.load(folder_name + '/y.npy')
        x_train_set, y_train_set = xs[:-1], ys[:-1]
        x_test_set, y_test_set = xs[1:], ys[1:]

        y_train_set = log_odds_vec(y_train_set)

        print('data size:', len(xs))
        return x_train_set, y_train_set, x_test_set, y_test_set

    return read_data(valid)

def get_elec2_dataset(valid=False):

    def get_all_data(file_path='./electricity-price-dataset/electricity-normalized.csv', 
                     num_feature=15):
        ''' 15 features in total:
        - The first seven features are indicator of week days;
        - The eighth feature is time
        - The ninth feature is date
        - The remaining five features: NSWprice, NSWdemand, VICprice, VICdemand, transfer
        - The bias
        '''
        X, y, _y = [], [], []
        with open(file_path, 'r') as datafile:
            header = datafile.readline()
            for line in datafile.readlines():
                feature = [0] * num_feature
                feature[-1] = 1 # bias term
                
                items = line.split(',')
                feature[int(items[1])-1] = 1 # day
                feature[7] = float(items[2]) # time
                feature[8] = float(items[0]) # date
                fid = 9 # others
                for item in items[3:-1]:
                    feature[fid] = float(item)
                    fid += 1
                    
                X.append(feature)
                
                # y.append(float(items[3])) # target
                # print(np.mean(y[-49:-1])<y[-1], items[-1])
                
                _y.append(float(items[3]))
                y_prob = np.sum(np.array(_y[-49:-1]) < _y[-1]) / len(_y[-49:-1])
                y.append(y_prob)
                # print(y_prob, items[-1])
                
            
        # make it predict the future
        X = X[49:]
        y = y[49:]
        num_instance = len(X)
        print(f'Number of samples: {num_instance}')
        return np.array(X), np.array(y)

    def log_odds_vec(p):
        # convert a probability to log-odds. Feel free to ignore the "divide by
        # zero" warning since we deal with it manually. The extreme values are
        # determined by looking at the histogram of the first-month data such
        # that they do not deviate too far from the others
        ind_0 = np.where(p == 0)[0]
        ind_1 = np.where(p == 1)[0]
        logodds = np.log(p) - np.log(1.-p)
        logodds[ind_0] = -4
        logodds[ind_1] = 4
        return logodds

    X, y = get_all_data()
    log_odds = log_odds_vec(y)

    val_size = 4000

    if valid:
        X = X[:val_size]
        y = y[:val_size]
        log_odds = log_odds[:val_size]
        
    else:
        X = X[val_size:]
        y = y[val_size:]
        log_odds = log_odds[val_size:]

    x_train_set, y_train_set, x_test_set, y_test_set = X[:-1], log_odds[:-1], X[1:], y[1:]
    return x_train_set, y_train_set, x_test_set, y_test_set

def get_sensordrift_dataset(valid=False):
    
    def get_batch_data(file_path, num_feature=129):
        '''`gas_class` - dict; args in {1,...,6}
        `gas_class[i]` - dict; args in {'X', 'y'}
        `gas_class[i][j]` - list
        
        e.g., gas_class[2]['X']
        '''
        gas_class = {}
        for i in range(1, 7):
            gas_class[i] = {}
            gas_class[i]['X'] = []
            gas_class[i]['y'] = []
        with open(file_path, 'r') as datafile:
            for line in datafile:
                feature = [0] * num_feature
                feature[-1] = 1 # bias term
                
                class_items = line.split(';')
                X = gas_class[int(class_items[0])]['X']
                y = gas_class[int(class_items[0])]['y']
                
                items = class_items[1].strip().split()
                y.append(float(items[0])) # concentration
                for item in items[1:]:
                    k, v = item.split(':')
                    feature[int(k)-1] = float(v)
                X.append(np.array(feature))
                
        # summary
        print(file_path)
        for i in range(1, 7):
            assert len(gas_class[i]['X']) == len(gas_class[i]['y'])
            num_instance = len(gas_class[i]['X'])
            print(f'class{i}: {num_instance} samples')
            
        return gas_class

    class_id = 2
    gas_class = get_batch_data('./sensor-drift-dataset/batch1.dat') # validation
    X, y = gas_class[class_id]['X'], gas_class[class_id]['y']

    mu_x = np.mean(X, axis=0, keepdims=True)
    mu_x[0, -1] = 0
    scale_x = np.std(X, axis=0, keepdims=True)
    scale_x[0, -1] = 1
    mu_y = np.mean(y)
    scale_y = np.std(y)

    def scaling_x(x):
        return (x-mu_x)/scale_x

    def scaling_y(y):
        return (y-mu_y)/scale_y

    if valid:
        X = scaling_x(X)
        y = scaling_y(y)
    else:
        file_names = ['batch2.dat', 
                      'batch3.dat', 
                      'batch4.dat', 
                      'batch5.dat', 
                      'batch6.dat', 
                      'batch7.dat', 
                      'batch8.dat', 
                      'batch9.dat', 
                      'batch10.dat']

        X, y = [], []
        for file_name in file_names:
            gas_class = get_batch_data('./sensor-drift-dataset/'+file_name)
            X.append(gas_class[class_id]['X'])
            y.append(gas_class[class_id]['y'])
            
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        X = scaling_x(X)
        y = scaling_y(y)

    x_train_set, y_train_set, x_test_set, y_test_set = X[:-1], y[:-1], X[1:], y[1:]
    return x_train_set, y_train_set, x_test_set, y_test_set