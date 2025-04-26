# Chess-AI-Alpha-Beta-and-MinMax-Implementation
# Chess AI: Alpha-Beta vs. MinMax

This repository contains two chess AI implementations: one using the **Alpha-Beta pruning algorithm** (`chess_alphabeta.py`) and another using the **MinMax algorithm** (`chess_minmax.py`). Both agents play chess against themselves, demonstrating the differences in performance and strategy between the two algorithms.

## Features

### AlphaBetaChessAI (`chess_alphabeta.py`)
- **Alpha-Beta Pruning**: Optimizes search efficiency by pruning irrelevant branches.
- **Move Ordering**: Prioritizes captures, promotions, checks, and castling using heuristic bonuses.
- **Transposition Table**: Caches previously evaluated positions to avoid redundant calculations.
- **Advanced Evaluation Function**:
  - Material balance
  - Piece mobility
  - Pawn structure (isolated/doubled/passed pawns)
  - King safety (pawn shields, open file penalties)
- **Depth-Limited Search**: Configurable search depth (default: 3).

### ChessMinMaxAgent (`chess_minmax.py`)
- **Basic MinMax Algorithm**: Exhaustive search without pruning.
- **Simpler Evaluation Function**:
  - Material balance
  - Piece mobility
- **Depth-Limited Search**: Configurable search depth (default: 3).

## Dependencies
- Python 3.x
- `chess` (Python Chess Library)
- `gym`
- `gym-chess`

Install dependencies with:
```bash
pip install chess gym gym_chess
