import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
from train_game import ResNet, ConnectFour


device = "cpu"
connectfour = ConnectFour()

model = ResNet(connectfour, 9, 128, device)
model.load_state_dict(torch.load("models/connectfour/model_7_ConnectFour.pt",map_location=device))
model.eval()

ROWS = 6
COLS = 7
'''
class ConnectFourGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four")
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        self.buttons = []
        self.canvas = tk.Canvas(root, width=COLS*100, height=(ROWS+1)*100, bg="blue")
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.drop_piece)

    def draw_board(self):
        self.canvas.delete("all")
        for c in range(COLS):
            for r in range(ROWS):
                x1, y1 = c*100, (r+1)*100
                x2, y2 = (c+1)*100, (r+2)*100
                self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, fill="white", outline="black")
        self.update_pieces()

    def update_pieces(self):
        colors = {0: "white", 1: "red", 2: "yellow"}
        for c in range(COLS):
            for r in range(ROWS):
                x1, y1 = c*100, (ROWS-r)*100
                x2, y2 = (c+1)*100, (ROWS-r+1)*100
                self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, fill=colors[self.board[r, c]], outline="black")

    def drop_piece(self, event):
        col = event.x // 100
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.board[row, col] = self.current_player
            self.draw_board()
            if self.check_win(self.current_player):
                self.canvas.create_text(COLS*50, 50, text=f"Player {self.current_player} wins!", font=("Arial", 24), fill="black")
                self.canvas.unbind("<Button-1>")
                return
            self.current_player = 3 - self.current_player  # Switch player
    
    def is_valid_location(self, col):
        return self.board[ROWS-1, col] == 0

    def get_next_open_row(self, col):
        for r in range(ROWS):
            if self.board[r, col] == 0:
                return r

    def check_win(self, player):
        # Check horizontal locations
        for r in range(ROWS):
            for c in range(COLS-3):
                if all(self.board[r, c+i] == player for i in range(4)):
                    return True
        # Check vertical locations
        for c in range(COLS):
            for r in range(ROWS-3):
                if all(self.board[r+i, c] == player for i in range(4)):
                    return True
        # Check positive diagonal
        for r in range(ROWS-3):
            for c in range(COLS-3):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True
        # Check negative diagonal
        for r in range(3, ROWS):
            for c in range(COLS-3):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True
        return False
'''

class ConnectFourGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four")
        self.board = np.zeros((ROWS, COLS), dtype=int)
        #self.current_player = 1
        self.buttons = []
        self.canvas = tk.Canvas(root, width=COLS*100, height=(ROWS+1)*100, bg="blue")
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.drop_piece)

        # Game logic
        self.player = 1
        self.state = connectfour.get_initial_state()

    def draw_board(self):
        self.canvas.delete("all")
        for c in range(COLS):
            for r in range(ROWS):
                x1, y1 = c*100, (r+1)*100
                x2, y2 = (c+1)*100, (r+2)*100
                self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, fill="white", outline="black")
        self.update_pieces()

    def update_pieces(self):
        colors = {0: "white", 1: "red", -1: "yellow"}
        for c in range(COLS):
            for r in range(ROWS):
                x1, y1 = c*100, (ROWS-r)*100
                x2, y2 = (c+1)*100, (ROWS-r+1)*100
                self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, fill=colors[self.board[r, c]], outline="black")

    def drop_piece(self, event):

        if np.sum(connectfour.get_valid_moves(self.state)) == 0:
            messagebox.showinfo("Game Over", f"Player {self.player} wins!")
            #self.reset_board()
            print("Game Over")
            return
        
        col = event.x // 100
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            print("col")
            print(col)
            self.board[row, col] = self.player
            self.state = connectfour.get_next_state(self.state, col, player=self.player)
            print(self.state)
            
            if self.check_win(self.player):
                self.canvas.create_text(COLS*50, 50, text=f"Player {self.player} wins!", font=("Arial", 24), fill="black")
                self.canvas.unbind("<Button-1>")
                return
            # Use model to predict the next move
            #self.current_player = 3 - self.current_player  # Switch player
            self.player = connectfour.get_opponent(self.player)
            encoded_state = connectfour.get_encoded_state(self.state)
            tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
            policy, value = model(tensor_state)
            value = value.item()
            policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

            print(value, policy)

            move_index = np.argmax(policy)
            print("move_index")
            print(move_index)
            row = self.get_next_open_row(move_index)

            self.board[row, move_index] = self.player
            
            self.state = connectfour.get_next_state(self.state, move_index, player=self.player)
            print(self.state)
            print(self.board)
            
            self.draw_board()
            if self.check_win(self.player):
                self.canvas.create_text(COLS*50, 50, text=f"Player {self.player} wins!", font=("Arial", 24), fill="black")
                self.canvas.unbind("<Button-1>")
                return
            self.player = connectfour.get_opponent(self.player)
            #self.current_player = 3 - self.current_player  # Switch player

    
    def is_valid_location(self, col):
        return self.board[ROWS-1, col] == 0

    def get_next_open_row(self, col):
        for r in range(ROWS):
            if self.board[r, col] == 0:
                return r

    def check_win(self, player):
        # Check horizontal locations
        for r in range(ROWS):
            for c in range(COLS-3):
                if all(self.board[r, c+i] == player for i in range(4)):
                    return True
        # Check vertical locations
        for c in range(COLS):
            for r in range(ROWS-3):
                if all(self.board[r+i, c] == player for i in range(4)):
                    return True
        # Check positive diagonal
        for r in range(ROWS-3):
            for c in range(COLS-3):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True
        # Check negative diagonal
        for r in range(3, ROWS):
            for c in range(COLS-3):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True
        return False
    
if __name__ == "__main__":
    root = tk.Tk()
    game = ConnectFourGame(root)
    root.mainloop()


'''


# Create the main game class
class TicTacToeGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")

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
        if self.buttons[index]["text"] != "":
            return
        if np.sum(tictactoe.get_valid_moves(self.state)) == 0:
            messagebox.showinfo("Game Over", f"Player {self.player} wins!")
            self.reset_board()
            return

        self.buttons[index].config(text="X")
        self.state = tictactoe.get_next_state(self.state, index, player=self.player)

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
        for button in self.buttons:
            button.config(text="")
        self.state = tictactoe.get_initial_state()
        self.player = 1

# Run the game
TicTacToeGame()
'''
