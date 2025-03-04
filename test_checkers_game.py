import unittest
import numpy as np
from CheckersTraining import CheckersTraining
from MathTooling import transform_dict_keys_base4_to_base72, clean_string

class TestCheckersGame(unittest.TestCase):
    def setUp(self):
        self.game = CheckersTraining()

    def test_generate_valid_moves_crowning(self):
        # Initialize the board with a single chip for player 1 one move away from being crowned
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 1, 3],  # Chip in a playable position near the opposite end
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        self.assertEqual(len(valid_moves), 2, "There should be 2 valid move for a single chip one move away from being crowned.")

        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        
        expected_moves = [[(1, 6, 0, 5)], [(1, 6, 0, 7)]]
        self.assertEqual(expected_moves, valid_moves, "Expected moves dont match.")



    def test_generate_valid_moves_single_chip(self):
        # Initialize the board with a single chip for player 1 in a playable position near the middle
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],  # target at (3, 2) and (3, 4)
            [3, 0, 3, 1, 3, 0, 3, 0],  # Chip in a playable position near the middle
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        self.assertEqual(len(valid_moves), 2, "There should be 2 valid moves for a single chip in a playable position near the middle.")
        
        # Verify the positions of the generated moves
        expected_moves = [
            [(4, 3, 3, 2)], [(4, 3, 3, 4)]
        ]
        
        self.assertEqual(valid_moves, expected_moves, "Generated move not in expected positions.")

    def test_generate_valid_moves_single_crowned_chip(self):
        # Initialize the board with a single crowned chip for player 1 in a playable position near the middle
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],  # target at (3, 2) and (3, 4)
            [3, 0, 3, 2, 3, 0, 3, 0],  # Crowned chip in a playable position near the middle
            [0, 3, 0, 3, 0, 3, 0, 3],  # target at (5, 2) and (5, 4)
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        self.assertEqual(len(valid_moves), 4, "There should be 4 valid moves for a crowned chip in a playable position near the middle.")
        # Verify the positions of the generated moves
        expected_moves = [[(4, 3, 3, 2)], [(4, 3, 3, 4)], [(4, 3, 5, 2)], [(4, 3, 5, 4)]]
        self.assertEqual(valid_moves, expected_moves, "Generated move not in expected positions.")
    
    def test_generate_valid_moves_capture(self):
        self.game = CheckersTraining()
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],   # the expected move is at (4, 1)
            [0, 3, -1, 3, 0, 3, 0, 3],  # Player -1 chip at (5, 2)
            [3, 0, 3, 1, 3, 0, 3, 0],   # Player 1 chip at (6, 3)
            [0, 3, 0, 3, 0, 3, 0, 3] 
        ])
        # Generate valid moves for player 1
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)

        # Check that there is a capture move
        self.assertEqual(len(valid_moves), 1, "There should be at least one valid move for player 1.")

        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()

        # Check if the captured piece is removed
        self.assertEqual(self.game.board[5, 2], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[6, 3], 0, "The piece was moved.")
        self.assertEqual(self.game.board[4, 1], 1, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 12, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")



    def test_generate_valid_moves_capture_by_crown(self):
        self.game = CheckersTraining()
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],   # the expected move is at (4, 1)
            [0, 3, -2, 3, 0, 3, 0, 3],  # Player -1 chip -2 at (5, 2)
            [3, 0, 3, 2, 3, 0, 3, 0],   # Player 1 chip  2 at (6, 3)
            [0, 3, 0, 3, 0, 3, 0, 3] 
        ])
        # Generate valid moves for player 1
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)

        # Check that there is a capture move
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")

        # Apply the first valid move
        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()

        # Check if the captured piece is removed
        self.assertEqual(self.game.board[5, 2], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[6, 3], 0, "The piece was moved.")
        self.assertEqual(self.game.board[4, 1], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 12, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")


    def test_get_scores(self):
        # Test board with initial setup
        board = np.array([
            [3, -1, 3, -1, 3, -1, 3, -1],
            [-1, 3, -1, 3, -1, 3, -1, 3],
            [3, -1, 3, -1, 3, -1, 3, -1],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [1, 3, 1, 3, 1, 3, 1, 3],
            [3, 1, 3, 1, 3, 1, 3, 1],
            [1, 3, 1, 3, 1, 3, 1, 3]
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 0, "Initial score for player 1 should be 0.")
        self.assertEqual(player2_score, 0, "Initial score for player -1 should be 0.")

        # Test board with a small game
        board = np.array([
            [3, -1, 3, -1, 3, -1, 3, -1],
            [-1, 3, -1, 3, -1, 3, -1, 3],
            [3, -1, 3, -1, 3, -2, 3, -1],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [1, 3, 1, 3, 1, 3, 2, 3],
            [3, 1, 3, 1, 3, 1, 3, 1],
            [1, 3, 1, 3, 1, 3, 1, 3]
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 3, "Initial score for player 1 should be 3, since it has a crown.")
        self.assertEqual(player2_score, 3, "Initial score for player -1 should be 3, since it has a crown.")

        # Test board with one capture by player 1
        board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 1, 3]  # Player 1 piece capturing
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 12, "Score for player 1 should be 12.")
        self.assertEqual(player2_score, 11, "Score for player -1 should be 11.")

        # Test board with one capture by player -1
        board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, -1, 3, 0, 3, 0, 3, 0],  # Player -1 piece capturing
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 11, "Score for player 1 should be 11.")
        self.assertEqual(player2_score, 12, "Score for player -1 should be 12.")

        # Test board with crowned pieces
        board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 2, 3, 0, 3, 0],  # Player 1 crowned piece
            [0, 3, 0, 3, 0, 3, 0, -2]  # Player -1 crowned piece
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 11 + 3, "Score for player 1 should be 11 + the crowned piece points for a total of 14.")
        self.assertEqual(player2_score, 11 + 3, "Score for player -1 should be 11 + the crowned piece points for a total of 14.")

        # Test board with all pieces captured (both players should have max score)
        board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        player1_score, player2_score = self.game.get_scores(board)
        self.assertEqual(player1_score, 12, "Score for player 1 should be 12.")
        self.assertEqual(player2_score, 12, "Score for player -1 should be 12.")


    def test_generate_valid_moves_capture_by_crown(self):
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, -1, 3, -1, 3, 0, 3, 0],  # Player -1 chips at (4, 1) and (4, 3)
            [0, 3, 2, 3, 0, 3, 0, 3],  # Player 1 crowned piece at (5, 2)
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        
        # Ensure there are valid moves
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")
        

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[4, 1], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[5, 2], 0, "The piece was moved.")
        self.assertEqual(self.game.board[3, 0], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly, both players have 11 points since each of them have a piece only each.
        self.assertEqual(self.game.player1_score, 11 + 3, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")

    def test_generate_valid_moves_capture_crown_by_not_crown(self):
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, -1, 3, -1, 3, 0, 3, 0],  # Player -1 chips at (4, 1) and (4, 3)
            [0, 3, 2, 3, 0, 3, 0, 3],    # Player 1 crowned piece at (5, 2)
            [3, 0, 3, 0, 3, 0, 3, 0],    # target positions will be on this row at (6, 1) and (6, 3)
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        valid_moves = self.game.generate_valid_moves(self.game.board, -1) # this test the turn is for the -1 player
        
        # Ensure there are valid moves
        self.assertEqual(len(valid_moves), 2, "Expects to have 2 Moves for the player -1.")
        
        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[4, 1], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[5, 2], 0, "The piece was moved.")
        self.assertEqual(self.game.board[6, 3], -1, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly, both players have 11 points since each of them have a piece only each.
        self.assertEqual(self.game.player1_score, 10, "Player 1's score should be 10, since -1 has 2 pieces.")
        self.assertEqual(self.game.player2_score, 12, "Player -1's score should be 12, since 1 has no pieces.")


    def test_diagonal_multiple_jumps(self):
        """
        There exist scenarios where multiple jumps over multiple tiles in checkers are valid, for instance,
        a crown at the (7, 0 ) and then enemies in a diagonal such in (6, 1) and (4, 3) and (2, 5)
        a jump to (1, 6) should be valid.
        """
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],  # (1, 6) should be valid
            [3, 0, 3, 0, 3, -1, 3, 0], # (2, 5) enemy tile
            [0, 3, 0, 3, 0, 3, 0, 3],  # (3, 4) should be open
            [3, 0, 3, -1, 3, 0, 3, 0], # (4, 3) player -1
            [0, 3, 0, 3, 0, 3, 0, 3],  # (5, 2) should be open
            [3, -1, 3, 0, 3, 0, 3, 0], # (6, 1) player -1
            [2, 3, 0, 3, 0, 3, 0, 3]   # (7, 0) player 1 crown
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        
        # Ensure there are valid moves
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")
        
        # Check if the valid moves include the correct multiple capture moves
        capture_move_found = [(7, 0, 5, 2), (5, 2, 3, 4), (3, 4, 1, 6)] in valid_moves
        self.assertTrue(capture_move_found, "The multiple jump capture move should be included in valid moves.")

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()
        self.game.print_board()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[6, 1], 0, "The first captured piece should be removed.")
        self.assertEqual(self.game.board[3, 4], 0, "The second captured piece should be removed.")
        self.assertEqual(self.game.board[2, 5], 0, "The third captured piece should be removed.")
        self.assertEqual(self.game.board[7, 0], 0, "The piece was moved.")
        self.assertEqual(self.game.board[1, 6], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 15, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")


    def test_diagonal_multiple_jumps_multiple_possible_ends(self):
        """
        There exist scenarios where multiple jumps over multiple tiles in checkers are valid, for instance,
        a crown at the (7, 0 ) and then enemies in a diagonal such in (6, 1) and (4, 3) and (2, 5)
        a jump to (1, 6) should be valid.
        """
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],  # (1, 2) should be valid
            [3, 0, 3, -1, 3, -1, 3, 0], # (2, 5) and (2, 3) enemy tile
            [0, 3, 0, 3, 0, 3, 0, 3],  # (3, 4) should be open
            [3, 0, 3, -1, 3, 0, 3, 0], # (4, 3) player -1
            [0, 3, 0, 3, 0, 3, 0, 3],  # (5, 2) should be open
            [3, -1, 3, 0, 3, 0, 3, 0], # (6, 1) player -1
            [2, 3, 0, 3, 0, 3, 0, 3]   # (7, 0) player 1 crown
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.board, 1)
        
        # Ensure there are valid moves
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")
        
        # Check if the valid moves include the correct multiple capture moves
        capture_move_found = [(7, 0, 5, 2), (5, 2, 3, 4), (3, 4, 1, 6)] in valid_moves
        self.assertTrue(capture_move_found, "The multiple jump capture move should be included in valid moves.")

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.update_score_and_board(chosen_move, 1)
        self.game.update_game_scores()
        self.game.print_board()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[6, 1], 0, "The first captured piece should be removed.")
        self.assertEqual(self.game.board[3, 4], 0, "The second captured piece should be removed.")
        self.assertEqual(self.game.board[2, 3], 0, "The third captured piece should be removed.")
        self.assertEqual(self.game.board[7, 0], 0, "The piece was moved.")
        self.assertEqual(self.game.board[1, 2], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 14, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")

    def test_key_generator(self):
        """
        This test key generator actually converts the board in to a string, and returns also a mirror key
        """
        board_1 = np.array([
            [3, 2, 3, 1, 3, 0, 3, 0],
            [-1, 3, -2, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 1, 3, -2],
            [0, 3, 0, 3, 2, 3, -1, 3]
        ])
        board_mirrored = np.array([
            [3, 1, 3, -2, 3, 0, 3, 0],
            [2, 3, -1, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 2, 3, 1],
            [0, 3, 0, 3, -1, 3, -2, 3]
        ])
        result_1 = f"{self.game.filter_and_flatten_board(board_1, 1)}"
        result_2 = f"{self.game.filter_and_flatten_board(board_mirrored, -1)}"
        result_1 = clean_string(result_1)
        result_2 = clean_string(result_2)
        
        self.assertEqual("-1-20001", self.game.mirror_play("1-10002"), "error reversing")
        self.assertEqual("1-10002", self.game.mirror_play("-1-20001"), "error reversing")
        self.assertEqual(result_2, self.game.mirror_play(result_1), "error reversing")
        self.assertEqual(result_1, self.game.mirror_play(result_2), "error reversing")

    def test_remove_zero_values(self):
        """
        This test that remove_zero_values works by removing values from a dictionary that are equals to 0
        """
        test_cases = [
            ({'a': 1, 'b': 0, 'c': 3, 'd': 0}, {'a': 1, 'c': 3}, 2),
            ({'a': 0, 'b': 0, 'c': 0}, {}, 3),
            ({'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, 0),
            ({}, {}, 0)
        ]

        for i, (input_dict, expected_dict, expected_count) in enumerate(test_cases):
            original_dict = input_dict.copy()
            self.game.remove_zero_values(input_dict)
            assert input_dict == expected_dict, f"Test case {i+1} failed: expected {expected_dict} but got {input_dict}"
            removed_count = len(original_dict) - len(input_dict)
            assert removed_count == expected_count, f"Test case {i+1} failed: expected {expected_count} removals but got {removed_count}"


    def test_tooling__base5_to_base64(self):
        input_dict = {
            "123": 1,
            "321": 2,
            "132": 3,
            "44444444444444444444444444444444": 4
        }
        expected_output = {
            "R": 1,   # 123 in base 4 is R in base 72 with the new alphabet
            "v": 2,   # 321 in base 4 is v in base 72 with the new alphabet
            "U": 3,   # 132 in base 4 is U in base 72 with the new alphabet
            "6f0RRvISrg&": 4  # 44444444444444444444444444444444 in base 4 is 6f0RRvISrg& in base 72 with the new alphabet
        }
        result = transform_dict_keys_base4_to_base72(input_dict)
        self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()
