import tables
import numpy as np
np.random.seed(0)

class IODataPoint(tables.IsDescription):
    x = tables.Int64Col(3)
    y = tables.Int64Col(3)

def generateData(n,current_y,table):
    ptr = table.row
    input_indexes = np.random.poisson(50, (int(2.1*10**(n-2)),3) )
    for i in range(1,input_indexes.shape[0]):
        input_indexes[i,:] = input_indexes[i,:]+input_indexes[i-1,:]
    j=np.array([0,0,0])
    prev_ind = 0
    while prev_ind<10**n:
        next_ind = min(input_indexes[j,np.array([0,1,2])])
        for i in range(next_ind-prev_ind):
            ptr['x'] = [0,0,0]
            ptr['y'] = current_y
            ptr.append()
        mod_ind = np.equal(input_indexes[j,np.array([0,1,2])],next_ind)
        new_val = np.random.choice((-1,1),size=3)
        ptr['x'] = np.multiply(new_val,mod_ind)
        j += mod_ind
        current_y[np.where(mod_ind)] = new_val[np.where(mod_ind)]
        ptr['y'] = current_y
        ptr.append()
        prev_ind = next_ind
    table.flush
    return current_y

#h5file = tables.open_file('/opt/DATA/train_test_arrays.h5', mode='w', title='data for training and testing 3-bit flop')
h5file = tables.open_file('/opt/DATA/train_test_arrays.h5', mode='a')

#table = h5file.create_table('/', 'test_array', IODataPoint, 'array for testing 3-bit flop')
#table = h5file.root.test_array
#current_y = generateData(5,np.array([-1,-1,-1]),table)

#table = h5file.create_table('/', 'train_array', IODataPoint, 'array for training 3-bit flop')
table = h5file.root.train_array
current_y = np.array([1,1,1])
for i in range(5):
    print(i)
    current_y = generateData(7,current_y,table)

h5file.close()