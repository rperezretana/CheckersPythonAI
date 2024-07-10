import numpy as np
import cupy as cp
import time
import random
from NeuralNetworkGPU import NeuralNetworkMultiGPU

class CheckersGame:
    def __init__(self):
        self.board = self.initialize_board()
        self.valid_moves_memo = {}  # Memo dictionary for generate_valid_moves
        self.transition_memo = {}  # Memo dictionary for is_valid_transition
        self.player1_moves = 0
        self.player2_moves = 0
        self.player1_score = 0
        self.player2_score = 0
        self.previous_boards = []  # List to track the last N board states
        self.loop_counter = 0
        self.loop_threshold = 3  # Threshold to detect a loop
        self.recent_boards_limit = 5  # Number of recent boards to track
        self.total_moves = 0  # Counter for the total number of moves
        self.move_limit = 200  # Hard limit for the total number of moves

    def initialize_board(self):
        # Define the initial checkers board setup
        board = np.zeros((8, 8), dtype=int)
        # Place initial pieces (1 for player, -1 for opponent)
        board[0:3:2, 1::2] = -1
        board[1:3:2, ::2] = -1
        board[5:8:2, 0::2] = 1
        board[6:8:2, 1::2] = 1
        # Mark non-playable tiles with 3
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    board[row, col] = 3
        return board

    def get_board_state(self):
        # Flatten the board to a 1D array for the neural network input
        return self.board.flatten()

    def is_valid_transition(self, current_board, new_board, from_pos, to_pos):
        """
        Check if the transition from current_board to new_board is valid.
        
        Parameters:
        current_board (np.ndarray): The current board state.
        new_board (np.ndarray): The proposed new board state.
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        
        Returns:
        bool: True if the transition is valid, False otherwise.
        """
        current_board = current_board.reshape((8, 8))
        new_board = new_board.reshape((8, 8))
        
        piece = current_board[from_pos[0], from_pos[1]]
        expected_piece_at_to_pos = piece

        # Check if the piece should be crowned
        if (piece == 1 and to_pos[0] == 0) or (piece == -1 and to_pos[0] == 7):
            expected_piece_at_to_pos = 2 * piece
        
        # Check if the piece at the from position has moved to the to position
        if not self.check_move_to_position(new_board, from_pos, to_pos, expected_piece_at_to_pos):
            print("Invalid move: piece at from_pos did not move to to_pos correctly")
            return False
        
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        # Simple move
        if abs(row_diff) == 1 and abs(col_diff) == 1:
            return self.check_simple_move(new_board, from_pos, to_pos, expected_piece_at_to_pos)
        
        # Capturing move
        if abs(row_diff) == 2 and abs(col_diff) == 2:
            return self.check_capturing_move(current_board, new_board, from_pos, to_pos, piece)
        
        # Multiple captures
        if abs(row_diff) > 2 or abs(col_diff) > 2:
            return self.check_multiple_captures(current_board, new_board, from_pos, to_pos, piece)
        
        print("Invalid move: not a valid single, capturing, or multiple capture move")
        return False

    def check_move_to_position(self, new_board, from_pos, to_pos, expected_piece_at_to_pos):
        """
        Check if the piece at the from position has moved to the to position.
        
        Parameters:
        new_board (np.ndarray): The proposed new board state.
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        expected_piece_at_to_pos (int): The expected piece at the to position.
        
        Returns:
        bool: True if the move is valid, False otherwise.
        """
        return new_board[to_pos[0], to_pos[1]] == expected_piece_at_to_pos and new_board[from_pos[0], from_pos[1]] == 0

    def check_simple_move(self, new_board, from_pos, to_pos, expected_piece_at_to_pos):
        """
        Check if the simple move is valid.
        
        Parameters:
        new_board (np.ndarray): The proposed new board state.
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        expected_piece_at_to_pos (int): The expected piece at the to position.
        
        Returns:
        bool: True if the simple move is valid, False otherwise.
        """
        if new_board[to_pos[0], to_pos[1]] == expected_piece_at_to_pos and new_board[from_pos[0], from_pos[1]] == 0:
            return True
        print("Invalid simple move")
        return False

    def check_capturing_move(self, current_board, new_board, from_pos, to_pos, piece):
        """
        Check if the capturing move is valid.
        
        Parameters:
        current_board (np.ndarray): The current board state.
        new_board (np.ndarray): The proposed new board state.
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        piece (int): The piece being moved.
        
        Returns:
        bool: True if the capturing move is valid, False otherwise.
        """
        mid_row = (from_pos[0] + to_pos[0]) // 2
        mid_col = (from_pos[1] + to_pos[1]) // 2
        
        # used to get the potential opponents piece:
        oponents_piece = {
            1:{ -1, -2},
            2:{ -1, -2},
            -1:{ 1, 2},
            -2:{ 1, 2},
        }

        # Check if the middle position has an opponent's piece (regular or crowned)
        if current_board[mid_row, mid_col] in oponents_piece[piece] and new_board[mid_row, mid_col] == 0:
            return True
        print(f"Invalid capturing move from {from_pos} to {to_pos} with middle {mid_row, mid_col}")
        return False



    def check_multiple_captures(self, current_board, new_board, from_pos, to_pos, piece):
        """
        Check if the multiple capture move is valid.
        
        Parameters:
        current_board (np.ndarray): The current board state.
        new_board (np.ndarray): The proposed new board state.
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        piece (int): The piece being moved.
        
        Returns:
        bool: True if the multiple capture move is valid, False otherwise.
        """
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        step_row = int(row_diff / abs(row_diff))
        step_col = int(col_diff / abs(col_diff))
        current_row, current_col = from_pos
        captures = 0
        while (current_row, current_col) != (to_pos[0], to_pos[1]):
            current_row += step_row
            current_col += step_col
            if current_board[current_row, current_col] == -piece and new_board[current_row, current_col] == 0:
                captures += 1
            elif current_board[current_row, current_col] != 0 or new_board[current_row, current_col] != 0:
                print("Invalid move during multiple captures")
                return False
        if captures > 0:
            return True
        print("No captures during multiple capture move")
        return False

    def generate_valid_moves(self, board, player):
        """
        Generate all potential valid moves for a given player.
        
        Parameters:
        board (np.ndarray): The current board state.
        player (int): The player number (1 for player 1, -1 for player -1).
        
        Returns:
        list: A list of board states representing all valid moves.
        """
        # Use the board and player as a key for memoization
        board_key = (board.tobytes(), player)
        
        if board_key in self.valid_moves_memo:
            return self.valid_moves_memo[board_key]
        
        board = board.reshape((8, 8))
        valid_moves = []
        capturing_moves = []
        
        regular_directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        crowned_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece == player or piece == 2 * player:
                    directions = crowned_directions if piece == 2 * player else regular_directions
                    for direction in directions:
                        new_row, new_col = row + direction[0], col + direction[1]
                        if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row, new_col] == 0:
                            new_board = board.copy()
                            new_board[row, col] = 0
                            # Crown the piece if it reaches the opposite end
                            if (player == 1 and new_row == 0) or (player == -1 and new_row == 7):
                                new_board[new_row, new_col] = 2 * player
                            else:
                                new_board[new_row, new_col] = piece
                            if self.is_valid_transition(board, new_board, (row, col), (new_row, new_col)):
                                valid_moves.append(new_board.flatten())
                    
                    # Check capturing moves
                    for direction in directions:
                        new_row, new_col = row + 2 * direction[0], col + 2 * direction[1]
                        mid_row, mid_col = row + direction[0], col + direction[1]
                        if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row, new_col] == 0 and (board[mid_row, mid_col] == -player or board[mid_row, mid_col] == -2 * player):
                            new_board = board.copy()
                            new_board[row, col] = 0
                            new_board[mid_row, mid_col] = 0
                            # Crown the piece if it reaches the opposite end
                            if (player == 1 and new_row == 0) or (player == -1 and new_row == 7):
                                new_board[new_row, new_col] = 2 * player
                            else:
                                new_board[new_row, new_col] = piece
                            if self.is_valid_transition(board, new_board, (row, col), (new_row, new_col)):
                                capturing_moves.append(new_board.flatten())
        
        # If capturing moves are available, they must be taken
        if capturing_moves:
            valid_moves = capturing_moves

        # Memoize the result
        self.valid_moves_memo[board_key] = valid_moves
        return valid_moves


    def print_board(self, board=None):
        """
        Print the current board with colors in the console.
        
        Parameters:
        board (np.ndarray): The board state to print. If None, the current board is printed.
        """
        if board is None:
            board = self.board
        else:
            board = board.reshape((8, 8))
        
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece == 3:
                    char = '\033[40m \033[40m  \033[0m'  # Dark square
                elif piece == 0:
                    char = '\033[47m \033[47m  \033[0m'  # White square
                elif piece == 1:
                    char = '\033[42m \033[42mO \033[0m'  # Green square (player 1)
                elif piece == 2:
                    char = '\033[42m♛ \033[42m \033[0m'  # Crowned piece (player 1)
                elif piece == -1:
                    char = '\033[41m \033[41mV \033[0m'  # Red square (player -1)
                elif piece == -2:
                    char = '\033[41m♛ \033[41m \033[0m'  # Crowned piece (player -1)
                print(char, end=" ")
            print()  # Newline after each row




    def run_simulation(self):
        """
        Run a slow simulation of the game where players make random valid moves.
        """
        player = 1  # Start with player 1
        print("Current Board:")
        self.print_board()  # Print the board for debugging
        while True:
            valid_moves = self.generate_valid_moves(self.get_board_state(), player)
            if not valid_moves:
                print(f"Player {player} has no valid moves. Game over.")
                print(f"Player 1 moves: {self.player1_moves}")
                print(f"Player -1 moves: {self.player2_moves}")
                print(f"Player 1 score: {self.player1_score}")
                print(f"Player -1 score: {self.player2_score}")
                print(f"Total moves: {self.total_moves}")
                break

            chosen_move = random.choice(valid_moves)
            from_pos, to_pos = self.get_move_positions(self.board, chosen_move.reshape((8, 8)))
            self.update_score_and_board(from_pos, to_pos, player)
            self.board = chosen_move.reshape((8, 8))
            
            # Check for loop
            if self.detect_loop():
                print("Loop detected. Game ends in a tie.")
                print(f"Total moves: {self.total_moves}")
                break

            self.print_board()
            print(f"Player 1 score: {self.player1_score}")
            print(f"Player -1 score: {self.player2_score}")
            print(f"Total moves: {self.total_moves}")
            time.sleep(1)  # Wait for 2 seconds

            if player == 1:
                self.player1_moves += 1
            else:
                self.player2_moves += 1

            player = -player  # Switch player

            self.total_moves += 1
            if self.total_moves >= self.move_limit:
                print("Move limit reached. Game ends in a tie.")
                print(f"Total moves: {self.total_moves}")
                break

    def get_move_positions(self, current_board, new_board):
        """
        Get the from and to positions for a move.
        
        Parameters:
        current_board (np.ndarray): The current board state.
        new_board (np.ndarray): The new board state.
        
        Returns:
        tuple: The from and to positions of the move.
        """
        current_board = current_board.reshape((8, 8))
        new_board = new_board.reshape((8, 8))
        
        changes = np.argwhere(current_board != new_board)
        
        from_pos = changes[0] if current_board[changes[0][0], changes[0][1]] != 0 else changes[1]
        to_pos = changes[1] if from_pos is changes[0] else changes[0]
        
        return (from_pos[0], from_pos[1]), (to_pos[0], to_pos[1])


    def update_score_and_board(self, from_pos, to_pos, player):
        """
        Update the score and board state after a move.
        
        Parameters:
        from_pos (tuple): The starting position of the piece.
        to_pos (tuple): The ending position of the piece.
        player (int): The player making the move.
        """
        # Move the piece
        self.board[to_pos[0], to_pos[1]] = self.board[from_pos[0], from_pos[1]]
        self.board[from_pos[0], from_pos[1]] = 0

        # Check for captures and remove the captured pieces
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]

        if abs(row_diff) == 2 and abs(col_diff) == 2:
            mid_row = (from_pos[0] + to_pos[0]) // 2
            mid_col = (from_pos[1] + to_pos[1]) // 2
            self.board[mid_row, mid_col] = 0
        elif abs(row_diff) > 2 or abs(col_diff) > 2:
            step_row = int(row_diff / abs(row_diff))
            step_col = int(col_diff / abs(col_diff))
            current_row, current_col = from_pos
            while (current_row, current_col) != (to_pos[0], to_pos[1]):
                current_row += step_row
                current_col += step_col
                if self.board[current_row, current_col] == -player or self.board[current_row, current_col] == -2 * player:
                    self.board[current_row, current_col] = 0

        # Check if the piece should be crowned
        if (player == 1 and to_pos[0] == 0) or (player == -1 and to_pos[0] == 7):
            self.board[to_pos[0], to_pos[1]] = 2 * player

        # Update the scores based on the current board state
        self.update_game_scores()

    
    def get_scores(self, board):
        """
        This functions determines the points of each player based on the count of the chips.
        This can be used to tell the points on any of the games.
        maximum score of a player is 12 for now.
        """
        player1_pieces = np.count_nonzero((board == 1) | (board == 2))
        player2_pieces = np.count_nonzero((board == -1) | (board == -2))
        return (12 - player2_pieces), (12 - player1_pieces)

    def update_game_scores(self):
        """
        Update the scores based on the current board state.
        """
        self.player1_score, self.player2_score = self.get_scores(self.board)


    def detect_loop(self):
        """
        Detect if a loop is occurring in the game by checking if the same board state 
        has been repeated in the last few moves.
        
        Returns:
        bool: True if a loop is detected, False otherwise.
        """
        board_state = self.get_board_state().tobytes()
        
        if board_state in self.previous_boards:
            self.loop_counter += 1
        else:
            self.loop_counter = 0  # Reset the loop counter if the state is not repeated
        
        self.previous_boards.append(board_state)
        
        if len(self.previous_boards) > self.recent_boards_limit:
            self.previous_boards.pop(0)  # Keep only the last N board states
        
        return self.loop_counter >= self.loop_threshold

# Quick run:
if __name__ == "__main__":
    game = CheckersGame()
    game.run_simulation()
