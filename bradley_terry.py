import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


class BradleyTerryModel(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        self.num_players = num_players
        self.betas = nn.Parameter(torch.ones(num_players))

    def forward(self, win_matrix):
        assert self.num_players == win_matrix.shape[0], ''
        alpha = torch.exp(self.betas)
        loss = 0 
        for i in range(self.num_players):
            for j in range(self.num_players):
                if i != j:
                    loss += win_matrix[i, j] * torch.log(alpha[i] / (alpha[i] + alpha[j]) )

        return - loss 

    def get_rank(self,):
        return torch.exp(self.betas).data.numpy()

    def get_win_rate(self, ):
        alpha = torch.exp(self.betas)
        win_rate = torch.zeros(self.num_players, self.num_players)
        for i in range(self.num_players):
           for j in range(self.num_players):
               if i != j:
                   win_rate[i, j] = alpha[i] / (alpha[i] + alpha[j]) 
        return win_rate.data.numpy()


def bradley_terry_inference(win_matrix, lr=0.01, iters=500):
    model = BradleyTerryModel(win_matrix.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in tqdm(range(iters)):
        optimizer.zero_grad()
        loss = model(win_matrix)
        loss.backward()
        optimizer.step()
    return model.get_rank(), model.get_win_rate()

if __name__ == '__main__':
    win_matrix = torch.tensor([
            [0, 8, 3],
            [4, 0, 2],
            [5, 3, 0],
            ])

    #  win_matrix = torch.tensor([
            #  [0, 7, 9, 7, 7, 9, 11],
            #  [6, 0, 7, 5, 11, 9, 9],
            #  [4, 6, 0, 7, 7, 8, 12],
            #  [6, 8, 6, 0, 6, 7, 10],
            #  [6, 2, 6, 7, 0, 7, 12],
            #  [4, 4, 5, 6, 6, 0, 6],
            #  [2, 4, 1, 3, 1, 7, 0]
            #  ])

    rank, win_rate = bradley_terry_inference(win_matrix)
    print(rank, '\n', win_rate)








