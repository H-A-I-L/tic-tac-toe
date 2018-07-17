import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="Pass this option if you want to run the model with cuda enabled. Note that the model will not run if pytorch fails to recognize a cuda enabled device.", action='store_true')
parser.add_argument("-i","--iterations", help="The number of iterations to train each phase. (Default is 20000)", type=int, default=20000)

USE_CUDA = True
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
        x = torch.nn.functional.softmax(x, dim=2)
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
    '''
    Produces a result based on the winning_combinations.
    If the current player (indexed by `2` in the `grid`) has made a winning combinations, 
    return 1 for that grid
    input:
    grid - a batch of game states
    winning_combinations - a nn.Conv2d object with weights set to the winning combinations
    '''
    grid_reshaped = grid.view(-1, 1, 3, 3)
    out = winning_combinations(grid_reshaped).eq(6)
    out = out.sum(1).squeeze(2) > 0
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
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.cuda:
        USE_CUDA = False

    iterations = args.iterations
    
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())

    # Converting the winning combinations into a convolusional layer.
    winning_combinations = torch.nn.Conv2d(1, len(WINNING_COMBINATIONS), kernel_size = 3, stride = 3, bias=False)
    winning_combinations.weight = torch.nn.Parameter(torch.Tensor(
        WINNING_COMBINATIONS).view(len(WINNING_COMBINATIONS), 1, 3, 3))
    winning_combinations.requires_grad = False
    if USE_CUDA:
        model.cuda()
        winning_combinations.cuda()
        
    loss_avg = 0
    # This loop trains the model to predict valid states
    for iteration in range(iterations):
        # Create a batch of different states the board can be in
        grid = init_grid_batch(20)
        grid_var = torch.autograd.Variable(grid.float())
        if USE_CUDA:
            grid_var = grid_var.cuda()
            
        # Use model to obtain proposed moves
        x = model(grid_var)

        # Get the move proposed by the model (The position with the highest probability)
        max_values, max_indices = x.max(2)
        max_indices = max_indices.unsqueeze(1).data

        # if the predicted position has a non-zero value (not empty) in the grid,
        # incur a loss, also maximize the probability of predicting position
        # that is non-zero in grid
        target = torch.autograd.Variable(torch.gather(grid_var,2, max_indices) == 0).float().squeeze()
        if USE_CUDA:
            target = target.cuda()
        loss = target - max_values.squeeze()
        loss = loss.pow(2).sum()/grid.size(0)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging the loss information
        loss_avg += loss.item()
        if iteration%500 == 0:
            print("loss: {:.4f} iteration: {}".format(loss_avg/500, iteration))
            loss_avg = 0

            
    #Save this model
    torch.save({"model":model.state_dict()}, "model.out")
    prev_model_state_dict = model.state_dict()

    # This loop will train the model to propose moves that will result in a win
    # We'll be trying to maximise the winning chances of player 1
    loss_avg = 0
    loss_count = 0
    player_1_win_count = 0
    player_2_win_count = 0

    prev_model = Model()
    if USE_CUDA:
        prev_model.cuda()

    for iteration in range(iterations):
        # Each iteration initialize the game with a random state.
        while True:
            grid = torch.autograd.Variable(init_grid_batch(1).float())
            if USE_CUDA:
                grid = grid.cuda()
            if judge_batch(grid, winning_combinations).squeeze().item() != 1 and \
               judge_batch(
                   convert_state_to_relative(grid, 1),
                   winning_combinations).squeeze().item() != 1:
                break
        current_player = 1
        loss = None
        
        # As there can be a maximum of 9 turns, we'll loop 9 times, and break if the game ends
        for turn in range(9):
            if iteration%500 == 0:
                print("player {}:".format(current_player))
                print(grid.view(3,3))

            # Initialize the opponent with the model from the previous iteration
            # TODO: select from a pool of previous models
            prev_model.load_state_dict(prev_model_state_dict)
            grid_for_current_player = convert_state_to_relative(grid, current_player)

            # The current player makes his move
            if current_player == 1:
                grid_score = model(grid_for_current_player)
                model_out = grid_score
            else:
                grid_score = prev_model(grid_for_current_player)
                #todo: check i valid, else move to next most probabale
            sorted_grid_scores, sorted_grid_index =  torch.sort(grid_score, dim = 2,descending = True)
            sorted_grid_index = sorted_grid_index.squeeze()
            sorted_grid_scores = sorted_grid_scores.squeeze()

            # var to store the grid after a move
            updated_grid_for_current_player = grid_for_current_player.clone()

            # Even when trained to predict valid moves, it is possible the proposed move is invlid
            # Hence, this work around.
            # TODO: incur a loss here?
            for state in range(9):
                if grid_for_current_player.squeeze()[sorted_grid_index[state]] != 0:
                    continue
                candidate_grid = grid_score == sorted_grid_scores[state]
                updated_grid_for_current_player[candidate_grid] = 2
                break

            # Thou shall be judged now
            judgement = judge_batch(updated_grid_for_current_player, winning_combinations)

            # If the judgement says the current player wins,
            # decide on the loss and terminate this round
            if judgement.squeeze().item() == 1:
                if iteration%500 == 0:
                    print("Winning move for player {}:".format(current_player))
                    print(inv_convert_state_to_relative(updated_grid_for_current_player,
                                                        current_player).view(3,3))
                judgement = judgement.float()
                if current_player == 2:
                    loss = torch.sum(judgement*model_out)
                else:
                    loss = torch.sum((-judgement)*model_out)
                break

            grid_for_current_player = updated_grid_for_current_player
            # If no one won, the game shall go on
            grid = inv_convert_state_to_relative(grid_for_current_player, current_player)
            if current_player == 1:
                current_player = 2
            else:
                current_player = 1

        # Store the previous model
        prev_model_state_dict = model.state_dict()

        
        if loss is not None:
            loss_avg += loss.item()
            loss_count += 1
            if iteration%500 == 0:
                print("loss:", loss_avg/loss_count,
                      "  count:", loss_count,
                      "  iteration: ", iteration)
                loss_avg = 0
                loss_count = 0
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_player == 1:
                player_1_win_count += 1
            else:
                player_2_win_count += 1

        if iteration%500 == 0:
            print("  Winning ratio (500 rounds): p1: {:.4f}   p2: {:.4f}"
              .format(player_1_win_count/500,
                      player_2_win_count/500))
            player_1_win_count = 0
            player_2_win_count = 0
        
    torch.save({"model":model.state_dict()}, "model_RL.out")
        
if __name__ == "__main__":
    main()
