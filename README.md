# A simple demonstration to train a DL model to play tic-tac-toe

- To run the server:
  - Execute the simple_server.py
```sh
python3 simple_server.py
```
  - *Note*: If pytorch is installed and a model has been trained, the script will automatically load the model and you can play the game against the model by navigating to smart_player.html

- To train the model:
  - Install pytorch [pytorch](http://www.pytorch.org)
  - Run the traning script:
```sh
python3 models/deep_learning_feed_forward.py --cuda -i 20000
```
  - Pass arguments as you see fit

Read here for more details: [Teaching a computer to play Tic-Tac-Toe.](https://ahmed-shariff.github.io/2018/07/17/tic-tac-toe/)
