import tkinter as tk
from tkinter import messagebox
import numpy as np
print(np.__version__)
import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import random
from tictactoe import TickTacToe, ResNet

device = "cpu"
print(device)

tictactoe = TickTacToe()

model = ResNet(tictactoe, 4, 64, device)
model.load_state_dict(torch.load("model_2.pt"))
model.eval()

# Create the main game class
class TicTacToeGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        
        #self.current_player = "X"
        #self.board = [""] * 9
        self.buttons = []
        
        # Game logic
        self.player = 1
        self.state = tictactoe.get_initial_state()
        
        self.create_board()
        self.window.mainloop()

    def create_board(self):
        for i in range(9):
            button = tk.Button(self.window, text="", font=("Arial", 24), width=5, height=2,
                               command=lambda i=i: self.on_click(i))
            button.grid(row=i//3, column=i%3)
            self.buttons.append(button)

    def on_click(self, index):

        if np.sum(tictactoe.get_valid_moves(self.state)) == 0:
            messagebox.showinfo("Game Over", f"Player {self.player} wins!")
            self.reset_board()
            return

        #self.board[index] = self.player
        self.buttons[index].config(text="X")
        print(self.player)
        print(index)
        
        self.state = tictactoe.get_next_state(self.state, index, player=self.player)
        print(self.state)
        # Check if the game is over
        if tictactoe.check_win(self.state, index):
            messagebox.showinfo("Game Over", f"Player {"X" if self.player == 1 else "O"} wins!")
            self.reset_board()
            return
        if np.sum(tictactoe.get_valid_moves(self.state)) == 0:
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_board()
            return

        encoded_state = tictactoe.get_encoded_state(self.state)
        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
        policy, value = model(tensor_state)
        value = value.item()
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        print(value, policy)

        move_index = np.argmax(policy)

        self.buttons[move_index].config(text="O")
        
        self.player = tictactoe.get_opponent(self.player)
        self.state = tictactoe.get_next_state(self.state, move_index, player=self.player)
        print(self.state)
        
        # Check if the game is over
        if tictactoe.check_win(self.state, move_index):
            messagebox.showinfo("Game Over", f"Player {"O" if self.player == -1 else "X"} wins!")
            self.reset_board()
            return
        if np.sum(tictactoe.get_valid_moves(self.state)) == 0:
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_board()
            return

        self.player = tictactoe.get_opponent(self.player)

    def reset_board(self):
        print("Resetting")
        for button in self.buttons:
            button.config(text="")
        self.state = tictactoe.get_initial_state()
        self.player = 1

# Run the game
TicTacToeGame()
