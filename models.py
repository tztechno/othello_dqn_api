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
