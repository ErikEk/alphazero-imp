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

class ConnectFourGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four")
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.buttons = []
        self.canvas = tk.Canvas(root, width=COLS*100, height=(ROWS+1)*100, bg="blue")
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.drop_piece)

        # prediction model logic
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
            return
        
        col = event.x // 100
        if self.is_valid_location(col):
            self.board[self.get_next_open_row(col), col] = self.player
            self.state = connectfour.get_next_state(self.state, col, player=self.player)
            
            if self.check_win(self.player):
                self.canvas.create_text(COLS*50, 50, text=f"Player {self.player} wins!", font=("Arial", 24), fill="black")
                self.canvas.unbind("<Button-1>")
                return

            # Use model to predict the next move
            self.player = connectfour.get_opponent(self.player)
            encoded_state = connectfour.get_encoded_state(self.state)
            tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
            policy, value = model(tensor_state)
            value = value.item()
            policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

            print(value, policy)

            move_index = np.argmax(policy)
            self.board[self.get_next_open_row(move_index), move_index] = self.player

            self.state = connectfour.get_next_state(self.state, move_index, player=self.player)
            print(self.state)

            self.draw_board()
            if self.check_win(self.player):
                self.canvas.create_text(COLS*50, 50, text=f"Player {self.player} wins!", font=("Arial", 24), fill="black")
                self.canvas.unbind("<Button-1>")
                return
            self.player = connectfour.get_opponent(self.player)
    
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
