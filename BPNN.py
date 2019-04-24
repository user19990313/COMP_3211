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
    temp_x = np.loadtxt("buggy_sort_buggy.py.csv",dtype=np.float32, delimiter=",")
    temp_y = np.loadtxt("buggy_sort_result.txt",dtype=np.float32, delimiter=",")
    # process data [>0 => 1, 0 => 0]
    x_train = np.float32(temp_x>0)
    y_train = temp_y # false 1 ; true 0
    # x_train, x_test, y_train, y_test = train_test_split(temp_x, temp_y, test_size=0.25, random_state=0)

    # set size
    n_in, n_h, n_out, batch_size = len(x_train[0]), 3, 1, len(x_train)

    x_train = torch.tensor(x_train).cuda()
    y_train = torch.tensor(y_train.T).cuda()

    inp_x_train = np.float32(temp_x > 0)
    inp_y_train = temp_y
    inp_x_train = torch.tensor(inp_x_train).cuda()
    inp_y_train = torch.tensor(inp_y_train.T).cuda()
    #debug case
    '''
    n_in ,n_h, n_out, batch_size = 9,3,1,7
    x_train = np.float32([  [1.,1.,1.,1.,0.,1.,0.,0.,1.],
                            [1.,0.,0.,0.,1.,1.,1.,1.,0],
                            [0.,0.,0.,0.,0.,1.,1.,0.,0],
                            [1.,1.,0.,0.,1.,0.,1.,1.,1],
                            [1.,1.,1.,0.,1.,1.,1.,1.,1],
                            [0.,0.,1.,0.,0.,1.,1.,1.,0],
                            [1.,1.,1.,1.,0.,1.,0.,1.,1]])
    y_train = np.float32([[0.],
                      [0.],
                      [0.],
                      [0.],
                      [0.],
                      [1.],
                      [1.]])
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train.T)
    n_in, n_h, n_out, batch_size = len(x_train[0]), 3, 1, len(x_train)
    '''

    # design model
    model = nn.Sequential(nn.Linear(n_in,n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h,n_out),
                          nn.Sigmoid()).cuda()

    # loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay = 1e-6)

    # try to improve
    inp_model = nn.Sequential(nn.Linear(n_in,n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h,n_out)
                          ).cuda()
    inp_criterion = torch.nn.MSELoss()
    inp_optimizer = torch.optim.SGD(inp_model.parameters(), lr=0.01,weight_decay = 1e-6)

    # use for plot
    loss_list =[]
    inp_loss_list=[]
    # train
    for i in range(1, 5001):
        # forward prapagation
        y_pred = model(x_train)

        # loss
        loss = criterion(y_pred,y_train)
        loss_list.append(float(loss))

        #print(model.named_parameters())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #################
        inp_y_pred = inp_model(inp_x_train)

        # loss
        inp_loss = inp_criterion(inp_y_pred, inp_y_train)
        inp_loss_list.append(float(inp_loss))

        # print(model.named_parameters())

        inp_optimizer.zero_grad()
        inp_loss.backward()
        inp_optimizer.step()

        if i % 1000 == 0:
            print('i: ', i, ' loss: ', loss.item(),' inp_loss: ', inp_loss.item())

    x_train = x_train.cpu()
    model = model.cpu()
    inp_model = inp_model.cpu()
    # print loss result

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot([i for i in range(5000)], loss_list, c='r')
    plt.plot([i for i in range(5000)], inp_loss_list, c='b')
    plt.show()

    # get fail line set
    s_f = [1]*n_in
    for i in range(batch_size):
        if(y_train[i]==1):
            s_f = np.multiply(s_f,x_train[i])

    #print(s_f)

    #test fail line
    model.eval()
    inp_model.eval()
    result = []
    inp_result = []
    for i in range(n_in):
        if(s_f[i]==1):
            test = torch.tensor(np.float32([0]*n_in))
            test[i] = 1
            #print(test,float(model(test)[0]))
            result.append((i+1,float(model(test)[0])))
            inp_result.append((i + 1, float(inp_model(test)[0])))
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)
    inp_sorted_result = sorted(inp_result, key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_result)):
        print(sorted_result[i],inp_sorted_result[i])

