import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    #print(torch.__version__)
    temp_x = np.loadtxt("qsort_qsort.py.csv",dtype=np.float32, delimiter=",")
    temp_y = np.loadtxt("qsort_result.txt",dtype=np.float32, delimiter=",")
    print(type(temp_x[0][0]))
    x_train, x_test, y_train, y_test = train_test_split(temp_x, temp_y, test_size=0.25, random_state=0)

    n_in, n_h, n_out, batch_size = len(x_train[0]), 3, 1, len(x_train)

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train.T)

    '''
    n_in ,n_h, n_out, batch_size = 9,3,1,7
    x = torch.tensor([[1.,1.,1.,1.,0.,1.,0.,0.,1.],
                     [1.,0.,0.,0.,1.,1.,1.,1.,0],
                     [0.,0.,0.,0.,0.,1.,1.,0.,0],
                     [1.,1.,0.,0.,1.,0.,1.,1.,1],
                     [1.,1.,1.,0.,1.,1.,1.,1.,1],
                     [0.,0.,1.,0.,0.,1.,1.,1.,0],
                     [1.,1.,1.,1.,0.,1.,0.,1.,1]])
    y = torch.tensor([[0.],
                      [0.],
                      [0.],
                      [0.],
                      [0.],
                      [1.],
                      [1.]])
    
    '''
    #print(x_train)
    #print(y_train)

    model = nn.Sequential(nn.Linear(n_in,n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h,n_out),
                          nn.Sigmoid())
    #print(model)
    # loss
    criterion = torch.nn.MSELoss()
    #print(criterion)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #print(optimizer)
    loss_list =[]
    for i in range(5000):
        # forward prapagation
        y_pred = model(x_train)

        # loss
        loss = criterion(y_pred,y_train)
        #print(y_pred)
        loss_list.append(float(loss))

        #print(model.named_parameters())

        optimizer.zero_grad()
        loss.backward()
        #print(model.state_dict())
        optimizer.step()
        #
        if i%100 == 0:
            print('i: ', i, ' loss: ', loss.item())
            #params = list(model.named_parameters())
            #for (name, param) in params:
            #    print(name, param.grad)
        #print("\n-----------------------------\n")


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('step')
    plt.ylabel('loss')
    ax1.scatter([i for i in range(5000)],loss_list,c = 'b',marker = '.')
    plt.show()


