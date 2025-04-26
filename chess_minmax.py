import chess
import gym
import gym_chess
import time
from typing import List, Optional

class ChessMinMaxAgent:
    """
    A chess agent that uses the MinMax algorithm to select moves.
    """
    
    def __init__(self, depth: int = 3):
        """
        Initialize the agent with a specific search depth.
        
        Args:
            depth: The depth of the MinMax search tree
        """
        self.depth = depth
        self.nodes_evaluated = 0
        
    def get_action(self, env: gym.Env) -> chess.Move:
        """
        Select the best move using MinMax algorithm.
        
        Args:
            env: The chess environment
            
        Returns:
            The best move according to MinMax algorithm
        """
        self.nodes_evaluated = 0
        start_time = time.time()
        
        board = env._board.copy()
        best_move = None
        best_value = -float('inf')
        
        # Iterate through all legal moves to find the best one
        for move in env.legal_moves:
            board.push(move)
            move_value = self._minimax(board, self.depth - 1, False)
            board.pop()
            
            if move_value > best_value or best_move is None:
                best_value = move_value
                best_move = move
                
        elapsed = time.time() - start_time
        print(f"Evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s")
        return best_move
    
    def _minimax(self, board: chess.Board, depth: int, maximizing_player: bool) -> float:
        """
        The recursive MinMax function.
        
        Args:
            board: The current board position
            depth: Remaining search depth
            maximizing_player: Whether the current player is maximizing
            
        Returns:
            The evaluated value of the position
        """
        self.nodes_evaluated += 1
        
        # Base case: reached max depth or game over
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)
            
        if maximizing_player:
            value = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self._minimax(board, depth - 1, False))
                board.pop()
            return value
        else:
            value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                value = min(value, self._minimax(board, depth - 1, True))
                board.pop()
            return value
    
    def _evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluate the current board position.
        
        Args:
            board: The board to evaluate
            
        Returns:
            A score representing how favorable the position is for white
        """
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                return 1000  # White wins
            elif result == '0-1':
                return -1000  # Black wins
            else:
                return 0  # Draw
        
        # Piece values (standard chess values)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.25,  # Slightly better than knight in open positions
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Kings are not captured
        }
        
        score = 0
        
        # Evaluate material
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        
        # Add small bonuses for piece mobility
        score += 0.1 * len(list(board.legal_moves))
        
        # Perspective: positive is good for white, negative for black
        return score if board.turn == chess.WHITE else -score

def play_game(agent_depth: int = 3):
    """
    Play a game of chess using the MinMax agent against itself.
    
    Args:
        agent_depth: The search depth for the MinMax algorithm
    """
    env = gym.make('Chess-v0')
    agent = ChessMinMaxAgent(depth=agent_depth)
    
    state = env.reset()
    done = False
    
    print("Initial board:")
    print(env.render(mode='unicode'))
    
    while not done:
        action = agent.get_action(env)
        state, reward, done, _ = env.step(action)
        
        print(f"\nMove: {action}")
        print(env.render(mode='unicode'))
        
        if done:
            print("\nGame over!")
            print(f"Result: {env._board.result()}")
            print(f"Final reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    play_game(agent_depth=3)