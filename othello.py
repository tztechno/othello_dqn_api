# othello.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import requests
import os
import random

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class OthelloGame:
    def __init__(self):
        self.board_size = 8
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        center = self.board_size // 2
        self.board[center-1:center+1, center-1:center+1] = [[-1, 1], [1, -1]]
        self.current_player = 1
        return self.get_state()
    
    def get_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_valid_move(self, row, col):
        if self.board[row, col] != 0:
            return False
            
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc):
                return True
        return False
    
    def _would_flip(self, row, col, dr, dc):
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == self.current_player:
                return len(to_flip) > 0
            to_flip.append((r, c))
            r, c = r + dr, c + dc
        return False
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        pieces_flipped = 0
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == 0:
                    break
                if self.board[r, c] == self.current_player:
                    for flip_r, flip_c in to_flip:
                        self.board[flip_r, flip_c] = self.current_player
                        pieces_flipped += 1
                    break
                to_flip.append((r, c))
                r, c = r + dr, c + dc
        
        self.current_player *= -1
        if not self.get_valid_moves():
            self.current_player *= -1
        
        return True
    
    def get_state(self):
        return self.board.copy()
    
    def get_winner(self):
        if self.get_valid_moves():
            return None
        
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN_Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def get_action(self, state, valid_moves, training=True):
        if not valid_moves:
            return None
            
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        valid_q_values = [(move, q_values[0][move[0] * 8 + move[1]].item()) 
                         for move in valid_moves]
        return max(valid_q_values, key=lambda x: x[1])[0]

class OthelloAI:
    def __init__(self):
        self.game = OthelloGame()
        self.ai = DQN_Agent(learning_rate=0.001, gamma=0.99)
        self.trained = False
    
    def load_model(self, model_path):
        if model_path.startswith("http://") or model_path.startswith("https://"):
            try:
                response = requests.get(model_path, stream=True)
                response.raise_for_status()
    
                filename = os.path.basename(model_path)
                filepath = filename
    
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
    
                model_path = filepath
    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error downloading model from {model_path}: {e}")
    
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
    
        checkpoint = torch.load(model_path, map_location=self.ai.device)
    
        self.ai.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.ai.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ai.epsilon = checkpoint['epsilon']
        self.ai.memory = deque(checkpoint['memory'], maxlen=10000)
        self.trained = True
    
        print(f"Model loaded from {model_path}")

# models.py
from pydantic import BaseModel
from typing import List, Optional, Tuple

class GameState(BaseModel):
    board: List[List[int]]
    valid_moves: List[Tuple[int, int]]
    winner: Optional[int] = None

class MoveRequest(BaseModel):
    row: int
    col: int

# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import requests
import os
from models import GameState, MoveRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_instance = None
ai_instance = None

@app.on_event("startup")
async def startup_event():
    global game_instance, ai_instance
    game_instance = OthelloGame()
    ai_instance = OthelloAI()
    
    try:
        ai_instance.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/new-game", response_model=GameState)
async def new_game():
    global game_instance
    state = game_instance.reset()
    return GameState(
        board=state.tolist(),
        valid_moves=game_instance.get_valid_moves(),
        winner=game_instance.get_winner()
    )

@app.post("/make-move", response_model=GameState)
async def make_move(move: MoveRequest):
    global game_instance, ai_instance
    
    if not game_instance.is_valid_move(move.row, move.col):
        raise HTTPException(status_code=400, detail="Invalid move")
    
    game_instance.make_move(move.row, move.col)
    
    valid_moves = game_instance.get_valid_moves()
    if valid_moves:
        state = game_instance.get_state()
        ai_move = ai_instance.ai.get_action(state, valid_moves, training=False)
        if ai_move:
            game_instance.make_move(*ai_move)
    
    return GameState(
        board=game_instance.get_state().tolist(),
        valid_moves=game_instance.get_valid_moves(),
        winner=game_instance.get_winner()
    )

@app.get("/valid-moves", response_model=List[Tuple[int, int]])
async def get_valid_moves():
    global game_instance
    return game_instance.get_valid_moves()