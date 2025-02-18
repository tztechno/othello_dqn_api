from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import requests
import os
from models import GameState, MoveRequest

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the game and AI instances
game_instance = None
ai_instance = None

@app.on_event("startup")
async def startup_event():
    global game_instance, ai_instance
    from othello import OthelloGame, OthelloAI
    
    game_instance = OthelloGame()
    ai_instance = OthelloAI()
    
    # Load the model on startup
    try:
        ai_instance.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue running even if model loading fails
        pass

@app.get("/new-game", response_model=GameState)
async def new_game():
    """Start a new game and return the initial state"""
    global game_instance
    state = game_instance.reset()
    return GameState(
        board=state.tolist(),
        valid_moves=game_instance.get_valid_moves(),
        winner=game_instance.get_winner()
    )

@app.post("/make-move", response_model=GameState)
async def make_move(move: MoveRequest):
    """Make a move and return the new game state"""
    global game_instance, ai_instance
    
    if not game_instance.is_valid_move(move.row, move.col):
        raise HTTPException(status_code=400, detail="Invalid move")
    
    # Make player's move
    game_instance.make_move(move.row, move.col)
    
    # Make AI's move if the game isn't over
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

from typing import List

@app.get("/valid-moves", response_model=List[List[int]])
async def get_valid_moves():
    """Get the list of valid moves for the current game state"""
    global game_instance
    return [list(move) for move in game_instance.get_valid_moves()]
