import chess
import gym
import gym_chess
import time
from typing import List, Tuple, Optional

class AlphaBetaChessAI:
    """
    Chess AI using Alpha-Beta pruning with move ordering optimizations.
    """
    
    def __init__(self, depth: int = 3):
        """
        Initialize the AI with search depth.
        
        Args:
            depth: Depth of the search tree
        """
        self.depth = depth
        self.nodes_evaluated = 0
        self.cutoffs = 0
        self.transposition_table = {}
        
        # Piece values for evaluation
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.25,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Move ordering bonuses
        self.move_order_bonuses = {
            'capture': 10,
            'promotion': 8,
            'check': 6,
            'castle': 4,
            'killer': 2
        }

    def get_action(self, env: gym.Env) -> chess.Move:
        """
        Get the best move using alpha-beta pruning.
        """
        self.nodes_evaluated = 0
        self.cutoffs = 0
        start_time = time.time()
        
        board = env._board.copy()
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        # Get moves ordered by heuristic
        legal_moves = self._order_moves(board, list(board.legal_moves))
        
        for move in legal_moves:
            board.push(move)
            value = self._alpha_beta(board, self.depth - 1, alpha, beta, False)
            board.pop()
            
            if value > alpha:
                alpha = value
                best_move = move
        
        elapsed = time.time() - start_time
        print(f"Alpha-Beta: Evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s")
        print(f"Cutoffs: {self.cutoffs} ({self.cutoffs/self.nodes_evaluated:.1%} of nodes)")
        return best_move
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                   maximizing_player: bool) -> float:
        """
        Alpha-Beta pruning algorithm with transposition table.
        """
        self.nodes_evaluated += 1
        
        # Check transposition table
        board_key = board.fen()
        if board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry['depth'] >= depth:
                return entry['value']
        
        # Terminal node or max depth
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)
        
        # Generate and order moves
        legal_moves = self._order_moves(board, list(board.legal_moves))
        
        if maximizing_player:
            value = -float('inf')
            for move in legal_moves:
                board.push(move)
                value = max(value, self._alpha_beta(board, depth - 1, alpha, beta, False))
                board.pop()
                
                alpha = max(alpha, value)
                if beta <= alpha:
                    self.cutoffs += 1
                    break  # Beta cutoff
                    
            # Store in transposition table
            self.transposition_table[board_key] = {
                'depth': depth,
                'value': value
            }
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                board.push(move)
                value = min(value, self._alpha_beta(board, depth - 1, alpha, beta, True))
                board.pop()
                
                beta = min(beta, value)
                if beta <= alpha:
                    self.cutoffs += 1
                    break  # Alpha cutoff
                    
            # Store in transposition table
            self.transposition_table[board_key] = {
                'depth': depth,
                'value': value
            }
            return value
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves to improve alpha-beta pruning efficiency.
        """
        scored_moves = []
        for move in moves:
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                score += self.move_order_bonuses['capture']
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                victim = board.piece_at(move.to_square)
                if victim:
                    score += self.piece_values[victim.piece_type] * 2
            
            # Prioritize promotions
            if move.promotion:
                score += self.move_order_bonuses['promotion']
            
            # Prioritize checks
            if board.gives_check(move):
                score += self.move_order_bonuses['check']
            
            # Prioritize castling
            if board.is_castling(move):
                score += self.move_order_bonuses['castle']
            
            scored_moves.append((score, move))
        
        # Sort by score (highest first)
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for (score, move) in scored_moves]
    
    def _evaluate_board(self, board: chess.Board) -> float:
        """
        Comprehensive board evaluation function.
        """
        # Terminal state evaluation
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                return 1000
            elif result == '0-1':
                return -1000
            else:
                return 0
        
        # Material evaluation
        score = 0
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        
        # Mobility bonus
        score += 0.1 * len(list(board.legal_moves))
        
        # Pawn structure evaluation
        score += self._evaluate_pawn_structure(board)
        
        # King safety
        score += self._evaluate_king_safety(board)
        
        # Perspective: positive is good for white, negative for black
        return score if board.turn == chess.WHITE else -score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """
        Evaluate pawn structure (isolated, doubled, passed pawns).
        """
        pawn_structure_score = 0
        
        # White pawns
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        for square in white_pawns:
            file = chess.square_file(square)
            
            # Isolated pawn penalty
            neighbors = False
            for f in [file - 1, file + 1]:
                if 0 <= f <= 7:
                    if any(chess.square(f, r) in white_pawns for r in range(8)):
                        neighbors = True
                        break
            if not neighbors:
                pawn_structure_score -= 0.5
            
            # Passed pawn bonus
            passed = True
            for r in range(chess.square_rank(square) + 1, 8):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f <= 7:
                        if chess.square(f, r) in board.pieces(chess.PAWN, chess.BLACK):
                            passed = False
                            break
                if not passed:
                    break
            if passed:
                pawn_structure_score += 1.0
        
        # Black pawns (same logic but inverted)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        for square in black_pawns:
            file = chess.square_file(square)
            
            # Isolated pawn penalty
            neighbors = False
            for f in [file - 1, file + 1]:
                if 0 <= f <= 7:
                    if any(chess.square(f, r) in black_pawns for r in range(8)):
                        neighbors = True
                        break
            if not neighbors:
                pawn_structure_score += 0.5
            
            # Passed pawn bonus
            passed = True
            for r in range(chess.square_rank(square) - 1, -1, -1):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f <= 7:
                        if chess.square(f, r) in board.pieces(chess.PAWN, chess.WHITE):
                            passed = False
                            break
                if not passed:
                    break
            if passed:
                pawn_structure_score -= 1.0
        
        return pawn_structure_score
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """
        Evaluate king safety based on pawn shield and open files.
        """
        king_safety = 0
        
        # White king safety
        white_king_square = board.king(chess.WHITE)
        if white_king_square:
            king_file = chess.square_file(white_king_square)
            king_rank = chess.square_rank(white_king_square)
            
            # Pawn shield
            shield = 0
            for f in [king_file - 1, king_file, king_file + 1]:
                if 0 <= f <= 7:
                    for r in [king_rank + 1, king_rank + 2]:
                        if 0 <= r <= 7:
                            if chess.square(f, r) in board.pieces(chess.PAWN, chess.WHITE):
                                shield += 0.5
            king_safety += shield
            
            # Open file penalty
            if not any(chess.square(king_file, r) in board.pieces(chess.PAWN, chess.WHITE) for r in range(8)):
                king_safety -= 0.75
        
        # Black king safety (inverted)
        black_king_square = board.king(chess.BLACK)
        if black_king_square:
            king_file = chess.square_file(black_king_square)
            king_rank = chess.square_rank(black_king_square)
            
            # Pawn shield
            shield = 0
            for f in [king_file - 1, king_file, king_file + 1]:
                if 0 <= f <= 7:
                    for r in [king_rank - 1, king_rank - 2]:
                        if 0 <= r <= 7:
                            if chess.square(f, r) in board.pieces(chess.PAWN, chess.BLACK):
                                shield += 0.5
            king_safety -= shield
            
            # Open file penalty
            if not any(chess.square(king_file, r) in board.pieces(chess.PAWN, chess.BLACK) for r in range(8)):
                king_safety += 0.75
        
        return king_safety

def play_game(depth: int = 3):
    """
    Play a game using the Alpha-Beta AI.
    """
    env = gym.make('Chess-v0')
    ai = AlphaBetaChessAI(depth=depth)
    
    state = env.reset()
    done = False
    
    print("Initial board:")
    print(env.render(mode='unicode'))
    
    while not done:
        action = ai.get_action(env)
        state, reward, done, _ = env.step(action)
        
        print(f"\nMove: {action}")
        print(env.render(mode='unicode'))
        
        if done:
            print("\nGame over!")
            print(f"Result: {env._board.result()}")
            print(f"Final reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    play_game(depth=3)