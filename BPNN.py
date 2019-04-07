import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BPNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(BPNN,self).__init__()

        # set sigmoid function as transfer function
        self.tfc = nn.Sigmoid()



if __name__ == '__main__':
    print(torch.__version__)
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
    print(x)
    print(y)
    model = nn.Sequential(nn.Linear(n_in,n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h,n_out),
                          nn.Sigmoid())
    print(model)
    # loss
    criterion = torch.nn.MSELoss()
    print(criterion)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print(optimizer)

    for i in range(50):
        # forward prapagation
        y_pred = model(x)

        # loss
        loss = criterion(y_pred,y)
        print('i: ', i, ' loss: ', loss.item())
        #print(y_pred)


        #print(model.named_parameters())

        optimizer.zero_grad()
        loss.backward()
        #print(model.state_dict())
        optimizer.step()
        #
        '''params = list(model.named_parameters())
        for (name, param) in params:
            print(name, param.grad)'''
        #print("\n-----------------------------\n")

