import numpy as np

DEBUG_ON = True # set to false if not interested on the ouputs
TRAINING = True # set False if not interested on training the nn
RANDOM_PLAY = True # This activates the random play, so the NN learns from randomness first
    
def debug_print(*args, end=None):
    if DEBUG_ON:
        if end:
            print(*args, end=end)
        else:
            print(*args, end=end)

class CheckersGame:
    def __init__(self):
        self.valid_moves_memo = {}  # Memo dictionary for generate_valid_moves
        self.transition_memo = {}  # Memo dictionary for is_valid_transition
        self.blank_board = None
        
        # Define the initial checkers board setup
        self.blank_board = np.zeros((8, 8), dtype=int)
        # Mark non-playable tiles with 3
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    self.blank_board[row, col] = 3
        self.board = self.initialize_board()
        

    def initialize_board(self):
        self.bodies_of_captures = set() # used to represent recently captured positions on the UI.
        self.player1_moves = 0
        self.player2_moves = 0
        self.player1_score = 0
        self.player2_score = 0
        self.loop_counter = 0
        self.loop_threshold = 3  # Threshold to detect a loop
        self.recent_boards_limit = 5  # Number of recent boards to track
        self.total_moves = 0  # Counter for the total number of moves
        self.move_limit = 200  # Hard limit for the total number of moves
        self.previous_boards = []  # List to track the last N board states
        self.total_moves += 1
        """
        Visualization of the matrix after this step:
        [
            [3, 0, 3, 0, 3, 0, 3, 0],  # 0
            [0, 3, 0, 3, 0, 3, 0, 3],  # 1
            [3, 0, 3, 0, 3, 0, 3, 0],  # 2
            [0, 3, 0, 3, 0, 3, 0, 3],  # 3
            [3, 0, 3, 0, 3, 0, 3, 0],  # 4
            [0, 3, 0, 3, 0, 3, 0, 3],  # 5
            [3, 0, 3, 0, 3, 0, 3, 0],  # 6
            [0, 3, 0, 3, 0, 3, 0, 3],  # 7
        ]
        """
        new_boad = self.blank_board.copy()
        self.place_players_chips(new_boad)
        return new_boad
    
    def place_players_chips(self, board):
        # Place initial pieces (1 for player, -1 for opponent)
        board[0:3:2, 1::2] = -1
        board[1:3:2, ::2] = -1
        board[5:8:2, 0::2] = 1
        board[6:8:2, 1::2] = 1


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
            debug_print("Invalid move: piece at from_pos did not move to to_pos correctly")
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
        
        debug_print("Invalid move: not a valid single, capturing, or multiple capture move")
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
        capturing_moves = []
        non_capturing_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece == player or piece == 2 * player:
                    self.find_all_capturing_moves(board, row, col, player, capturing_moves)
                    if not capturing_moves:
                        self.find_all_non_capturing_moves(board, row, col, player, non_capturing_moves)
        
        return capturing_moves if capturing_moves else non_capturing_moves


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
                    c = ' '
                    if f"{row}_{col}" in self.bodies_of_captures:
                        c = 'X'
                    char = f'\033[47m \033[47m{c} \033[0m'  # White square
                elif piece == 1:
                    char = '\033[42m \033[42mO \033[0m'  # Green square (player 1)
                elif piece == 2:
                    char = '\033[42m♛ \033[42m \033[0m'  # Crowned piece (player 1)
                elif piece == -1:
                    char = '\033[41m \033[41mV \033[0m'  # Red square (player -1)
                elif piece == -2:
                    char = '\033[41m♛ \033[41m \033[0m'  # Crowned piece (player -1)
                debug_print(char, end=" ")
            debug_print()  # Newline after each row



    def update_score_and_board(self, move_positions, player, board=None):
        """
        Update the score and board state after a move sequence.

        Parameters:
        move_positions (list): A list of tuples representing the from and to positions for each jump in the sequence.
        ie on multiple routes: [(7, 0, 5, 2), (5, 2, 3, 4)]
        ie on single move: [(7, 0, 5, 2)]
        player (int): The player making the move.
        """
        if board is None or not board.any():
            board = self.board

        # bodies are the recently captured positions
        self.bodies_of_captures = set() # reset bodies
        for move in move_positions:
            from_pos, to_pos = (move[0], move[1]), (move[-2], move[-1])
            # Move the piece
            board[to_pos[0], to_pos[1]] = board[from_pos[0], from_pos[1]]
            board[from_pos[0], from_pos[1]] = 0

            # Check for captures and remove the captured pieces
            row_diff = to_pos[0] - from_pos[0]
            col_diff = to_pos[1] - from_pos[1]

            if abs(row_diff) == 2 and abs(col_diff) == 2:
                mid_row = (from_pos[0] + to_pos[0]) // 2
                mid_col = (from_pos[1] + to_pos[1]) // 2
                self.bodies_of_captures.add(f"{mid_row}_{mid_col}")
                board[mid_row, mid_col] = 0
            elif abs(row_diff) > 2 or abs(col_diff) > 2:
                step_row = int(row_diff / abs(row_diff))
                step_col = int(col_diff / abs(col_diff))
                current_row, current_col = from_pos
                while (current_row, current_col) != (to_pos[0], to_pos[1]):
                    current_row += step_row
                    current_col += step_col
                    if board[current_row, current_col] == -player or board[current_row, current_col] == -2 * player:
                        self.bodies_of_captures.add(f"{current_row}_{current_col}")
                        board[current_row, current_col] = 0

            # Check if the piece should be crowned
            if (player == 1 and to_pos[0] == 0) or (player == -1 and to_pos[0] == 7):
                board[to_pos[0], to_pos[1]] = 2 * player

        # Update the scores based on the current board state
        # but only if the main scoreboard is updated
        if board is None or not board.any():
            self.update_game_scores()
        return board
    
    def get_scores(self, board):
        """
        This functions determines the points of each player based on the count of the chips.
        This can be used to tell the points on any of the games.
        A player gets one point per chip the enemy is missing,
        player also gets 3 points per crown/super that the plauyer gets
        """
        player1_pieces = np.count_nonzero((board == 1) | (board == 2))
        player2_pieces = np.count_nonzero((board == -1) | (board == -2))

        player1_c_pieces = np.count_nonzero(board == 2)
        player2_c_pieces = np.count_nonzero(board == -2)

        return (12 - player2_pieces) + player1_c_pieces * 3, (12 - player1_pieces) + player2_c_pieces * 3

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
        board_state = f"{self.board}"
        
        if board_state in self.previous_boards:
            self.loop_counter += 1
        else:
            self.loop_counter = 0  # Reset the loop counter if the state is not repeated
        
        self.previous_boards.append(board_state)
        
        if len(self.previous_boards) > self.recent_boards_limit:
            self.previous_boards.pop(0)  # Keep only the last N board states
        
        return self.loop_counter >= self.loop_threshold
