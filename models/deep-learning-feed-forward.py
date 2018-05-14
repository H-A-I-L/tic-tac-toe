import torch

USE_CUDA = False
HIDDEN_LAYER_SIZE = 100
WINNING_COMBINATIONS = [[1,1,1, 0,0,0, 0,0,0],
                        [0,0,0, 1,1,1, 0,0,0],
                        [0,0,0, 0,0,0, 1,1,1],
                        [1,0,0, 1,0,0, 1,0,0],
                        [0,1,0, 0,1,0, 0,1,0],
                        [0,0,1, 0,0,1, 0,0,1],
                        [1,0,0, 0,1,0, 0,0,1],
                        [0,0,1, 0,1,0, 1,0,0]]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(
            in_features = 9,
            out_features = HIDDEN_LAYER_SIZE,
            bias = False)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.hidden_layer = torch.nn.Linear(
            in_features = HIDDEN_LAYER_SIZE,
            out_features = HIDDEN_LAYER_SIZE,
            bias = False)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.output_layer = torch.nn.Linear(
            in_features = HIDDEN_LAYER_SIZE,
            out_features = 9,
            bias = False)
        self.bn3 = torch.nn.BatchNorm1d(1)

    def forward(self, in_features):
        x = self.input_layer(in_features)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace = True)
        x = self.hidden_layer(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x, inplace = True)
        x = self.output_layer(x)
        x = self.bn3(x)
        x = torch.nn.functional.softmax(x)
        return x

def judge(grid):
    '''
judges if the player indexed by 2 in the grid is the winner.

input:
grid - a list with 9 elements. Each element can be either 0, 1 or 2

returns:
won? - Returns True if the player indexed by 2 in the grid is a winner, else returns False.
'''
    #check the columns and rows
    for i in range(3):
        if grid[i] == 2 and grid[i+3] == 2 and grid[i+6] == 2:
            return True
        if grid[i*3] == 2 and \
           grid[i*3+1] == 2 and \
           grid[i*3+2 == 2]:
            return True
    #check the two diagonals
    if grid[0] == 2 and grid[4] == 2 and grid[8]:
        return True
    if grid[2] == 2 and grid[4] == 2 and grid[6]:
        return True

def judge_batch(grid, winning_combinations):
    grid_reshaped = grid.view(-1, 1, 3, 3)
    print(grid_reshaped)
    out = winning_combinations(grid_reshaped).eq(6)
    out = out.sum(1).squeeze(2)
    print(out, "*"*20)
    return out

    
def convert_state_to_relative(grid, current_player):
    '''
    Convert the grid relative the current player.
    The grid relative to the current player would be as follows:
    if a position is the current player's it will have the value 2
    if a position is the opposite player's it will have the value 1
    if a position is not used it will have the value 0
    
    input:
    grid - A flat tensor that represents the input grid, should have the following convention:
         if a position on the grid belongs to player 1, it should have the value 1
         if a position on the grid belongs to player 2, it should have the value 2
         if a position on the grid belongs to neither player, it should have the value 0
    current_player - The player relative to whome the grid needs to be converted (alowed valued: 1, 2)
    
    return:
    grid - as described above.
    '''
    assert current_player != 1 or current_player != 2, "allowed values are 1 and 2"
    # grid_for_current_player = [2 if i == current_player else 0 for i in grid]
    # grid_for_opposing_player = [1 if i != current_player and i != 0 else 0 for i in grid]
    # return [grid_for_current_player[i] + grid_for_opposing_player[i] for i in range(len(grid))]
    output_grid = torch.zeros_like(grid)
    output_grid[grid == current_player] = 2
    output_grid[((grid != current_player) + (grid != 0)).eq(2)] = 1
    return output_grid

def inv_convert_state_to_relative(grid, current_player):
    '''
    The inverse of the function `convert_state_to_relative`

    input:
    grid: A flat tensor that represents the input grid, should have the following convention:
           if a position belongs the to current player, it should have the value 2
           if a position belongs the to opposing player, it should have the value 1
           if a position belongs the to neither, it should have the value 0
    current_player - The player relative to whome the grid is set. (Allowed values: 1,2)
    '''
    assert current_player != 1 or current_player != 2, "allowed values are 1 and 2"
    if current_player == 2:
        opposing_player = 1
    else:
        opposing_player = 2
        
    # grid_for_current_player = [current_player if i == 2 else 0 for i in grid]
    # grid_for_opposing_player = [opposing_player if i == 1 else 0 for i in grid]
    # return [grid_for_current_player[i] + grid_for_opposing_player[i] for i in range(len(grid))]
    output_grid = torch.zeros_like(grid)
    output_grid[grid == 2] = current_player
    output_grid[grid == 1] = opposing_player
    return output_grid

def init_grid_batch(batch_size = 20):
    #return [0]*9
    return (torch.rand(9*batch_size)*3).int().view(-1, 1, 9)

def main():
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    winning_combinations = torch.nn.Conv2d(1, len(WINNING_COMBINATIONS), kernel_size = 3, stride = 3, bias=False)
    winning_combinations.weight = torch.nn.Parameter(torch.Tensor(
        WINNING_COMBINATIONS).view(len(WINNING_COMBINATIONS), 1, 3, 3))
    
    if USE_CUDA and torch.cuda.is_available():
        model.cuda()
        optimizer.cuda()
        
    loss_avg = 0
    for iteration in range(1):
        grid = init_grid_batch(20)
        grid_var = torch.autograd.Variable(grid.float())
        x = model(grid_var)
        
        max_values, max_indices = x.max(2)
        max_indices = max_indices.unsqueeze(1).data

        #print(grid.gather(2, max_indices), grid.gather(2, max_indices) == 0, grid)
        #if the predicted position has a non-zero value in the grid, incur a loss,
        #also maximize the probability of predicting position that is non-zero in grid
        target = torch.autograd.Variable(grid.gather(2, max_indices) == 0).float().squeeze()
        loss = target - max_values.squeeze()
        loss = loss.pow(2).sum()/grid.size(0)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.data[0]
        if iteration%500 == 0:
            print("loss: {:.4f} iteration: {}".format(loss_avg/500, iteration))
            loss_avg = 0

    #Save this model
    torch.save({"model":model.state_dict()}, "model.out")
    prev_model_state_dict = model.state_dict()
    for iteration in range(1):
        grid = torch.autograd.Variable(init_grid_batch(1).float())
        current_player = 1
        loss = None
        for turn in range(9):
            prev_model = Model()
            prev_model.load_state_dict(prev_model_state_dict)
            grid_for_current_player = convert_state_to_relative(grid, current_player)
            if current_player == 1:
                grid_move = model(grid_for_current_player)
                judgement = judge_batch(grid_move, winning_combinations)
                # Checking graph 
                # gf = judgement.grad_fn
                # while gf is not None:
                #     print(gf)
                #     print(gf.__dir__())
                #     print("*"*20)
                #     try:
                #         [print(g[0].next_functions) for g in gf.next_functions]
                #     except:
                #         pass
                #     gf = gf.next_functions[0][0]
                # return
                if judgement.squeeze().data[0] == 1: 
                    loss = judgement
                    break
            else:
                grid_move = prev_model(grid_for_current_player)
                judgement = judge_batch(grid_move, winning_combinations)
                if judgement.squeeze().data[0] == 1: 
                    loss = judgement
                    break
            grid = inv_convert_state_to_relative(grid_for_current_player, current_player)

            if current_player == 1:
                current_player = 2
            else:
                current_player = 1

        if loss is not None:
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prev_model_state_dict = model.state_dict()
    torch.save({"model":model.state_dict()}, "model_RL.out")
        
if __name__ == "__main__":
    main()
