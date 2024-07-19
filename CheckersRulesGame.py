
import numpy as np
from GameBoard import GameBoard
from SimpleConfig import debug_print

class CheckersRulesGame(GameBoard):
    def __init__(self):
        super().__init__()


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
        key = f"{current_board} - {new_board} - {from_pos}-{to_pos}"
        if key in self.transition_memo:
            return self.transition_memo[key]

        current_board = current_board.reshape((8, 8))
        new_board = new_board.reshape((8, 8))
        
        piece = current_board[from_pos[0], from_pos[1]]
        expected_piece_at_to_pos = piece

        # Check if the piece should be crowned
        if (piece == 1 and to_pos[0] == 0) or (piece == -1 and to_pos[0] == 7):
            expected_piece_at_to_pos = 2 * piece
        
        # Check if the piece at the from position has moved to the to position
        if not self.check_move_to_position(new_board, from_pos, to_pos, expected_piece_at_to_pos):
            debug_print("Invalid move: piece at from_pos did not move to to_pos correctly")
            self.transition_memo[key] = False
            return False
        
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        # Simple move
        if abs(row_diff) == 1 and abs(col_diff) == 1:
            res = self.check_simple_move(new_board, from_pos, to_pos, expected_piece_at_to_pos)
            self.transition_memo[key] = res
            return res
        
        # Capturing move
        if abs(row_diff) == 2 and abs(col_diff) == 2:
            res = self.check_capturing_move(current_board, new_board, from_pos, to_pos, piece)
            self.transition_memo[key] = res
            return res
        
        # Multiple captures
        if abs(row_diff) > 2 or abs(col_diff) > 2:
            res = self.check_multiple_captures(current_board, new_board, from_pos, to_pos, piece)
            self.transition_memo[key] = res
            return res
        
        debug_print("Invalid move: not a valid single, capturing, or multiple capture move")
        self.transition_memo[key] = False
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
        debug_print("Invalid simple move")
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
        
        # Check if the middle position has an opponent's piece (regular or crowned)
        if self.are_coordenates_an_opponents_piece(current_board, mid_row, mid_col, piece) and new_board[mid_row, mid_col] == 0:
            return True
        debug_print(f"Invalid capturing move from {from_pos} to {to_pos} with middle {mid_row, mid_col}")
        return False

    def are_coordenates_an_opponents_piece(self, board, row, col, piece_or_player):
        """
            piece_moving is the piece of the current player that is moving
        """
        # used to get the potential opponents piece:
        oponents_piece = {
            1:{ -1, -2},
            2:{ -1, -2},
            -1:{ 1, 2},
            -2:{ 1, 2},
        }
        return board[row, col] in oponents_piece[piece_or_player]

    def are_coordenates_valid(self, board, row, col):
        # returns true if the cordenates are in the bounds of the board and it is not a tile 3, or umplayable tile.
        return 0 <= row < 8 and 0 <= col < 8 and board[row, col] != 3
    
    def are_coordenates_empty_and_playable(self, board, row, col):
        return self.are_coordenates_valid(board, row, col) and board[row, col] == 0

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
            if self.are_coordenates_an_opponents_piece(current_board, current_row, current_col, piece) and new_board[current_row, current_col] == 0:
                captures += 1
            elif current_board[current_row, current_col] != 0 or new_board[current_row, current_col] != 0:
                debug_print("Invalid move during multiple captures")
                return False
        if captures > 0:
            return True
        debug_print("No captures during multiple capture move")
        return False

    def generate_valid_moves(self, board, player):
        key = f"{player} - {self.board}"
        if key in self.valid_moves_memo:
            return self.valid_moves_memo[key]
        capturing_moves = []
        non_capturing_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece == player or piece == 2 * player:
                    self.find_all_capturing_moves(board, row, col, player, capturing_moves)
                    if not capturing_moves:
                        self.find_all_non_capturing_moves(board, row, col, player, non_capturing_moves)
        
        res = capturing_moves if capturing_moves else non_capturing_moves
        self.valid_moves_memo[key] = res
        return res

    def get_directions_for_piece_during_capture(self, board, row, col, player):
        chip_type = board[row, col]
        directions = []
        if player == 1 or chip_type in (2, -2):
            directions = directions + [(-1, -1), (-1, 1), (-2, -2), (-2, 2)]
        if player == -1 or chip_type in (2, -2):
            directions = directions + [(1, -1), (1, 1), (2, -2), (2, 2)]
        return directions

    def get_directions_for_piece_non_capture(self, board, row, col, player):
        chip_type = board[row, col]
        directions = []
        if player == 1 or chip_type in (2, -2):
            directions = directions + [(-1, -1), (-1, 1)]
        if player == -1 or chip_type in (2, -2):
            directions = directions + [(1, -1), (1, 1)]
        return directions

    def find_all_capturing_moves(self, board, row, col, player, capturing_moves, path=[]):
        directions = self.get_directions_for_piece_during_capture(board, row, col, player)
        found = False
        for dr, dc in directions:
            mid_row, mid_col = row + dr // 2, col + dc // 2
            new_row, new_col = row + dr, col + dc
            if self.are_coordenates_valid(board, new_row, new_col) and self.are_coordenates_empty_and_playable(board, new_row, new_col):
                if self.are_coordenates_an_opponents_piece(board, mid_row, mid_col, player):
                    new_board = board.copy()
                    new_board[row, col] = 0
                    new_board[mid_row, mid_col] = 0
                    new_board[new_row, new_col] = board[row, col]
                    path.append((row, col, new_row, new_col))
                    self.find_all_capturing_moves(new_board, new_row, new_col, player, capturing_moves, path)
                    found = True
                    path.pop()
        
        if not found and path:
            capturing_moves.append(path.copy())

    def find_all_non_capturing_moves(self, board, row, col, player, non_capturing_moves):
        directions = self.get_directions_for_piece_non_capture(board, row, col, player)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.are_coordenates_valid(board, new_row, new_col) and self.are_coordenates_empty_and_playable(board, new_row, new_col):
                non_capturing_moves.append([(row, col, new_row, new_col)])
