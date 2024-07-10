import unittest
import numpy as np
from CheckersGame import CheckersGame

class TestCheckersGame(unittest.TestCase):
    def setUp(self):
        self.game = CheckersGame()
    
    def test_generate_valid_moves_single_chip(self):
        # Initialize the board with a single chip for player 1 in a playable position near the middle
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 1, 3, 0, 3, 0],  # Chip in a playable position near the middle
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)
        self.assertEqual(len(valid_moves), 2, "There should be 2 valid moves for a single chip in a playable position near the middle.")
        
        # Verify the positions of the generated moves
        expected_moves = [
            (3, 2), (3, 4)
        ]
        
        for move in valid_moves:
            move_board = move.reshape((8, 8))
            chip_position = np.argwhere(move_board == 1)
            self.assertIn((chip_position[0][0], chip_position[0][1]), expected_moves, "Generated move not in expected positions.")

    def test_generate_valid_moves_single_crowned_chip(self):
        # Initialize the board with a single crowned chip for player 1 in a playable position near the middle
        self.game.board = np.array([
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 2, 3, 0, 3, 0],  # Crowned chip in a playable position near the middle
            [0, 3, 0, 3, 0, 3, 0, 3],
            [3, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 3, 0, 3, 0, 3]
        ])
        
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)
        self.assertEqual(len(valid_moves), 4, "There should be 4 valid moves for a crowned chip in a playable position near the middle.")
        
        # Verify the positions of the generated moves
        expected_moves = [
            (3, 2), (3, 4), (5, 2), (5, 4)
        ]
        
        for move in valid_moves:
            move_board = move.reshape((8, 8))
            chip_position = np.argwhere(move_board == 2)
            self.assertIn((chip_position[0][0], chip_position[0][1]), expected_moves, "Generated move not in expected positions.")
    
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
        
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)
        self.assertEqual(len(valid_moves), 2, "There should be 2 valid move for a single chip one move away from being crowned.")
        
        # Verify that the chip is crowned after the move
        move_board = valid_moves[0].reshape((8, 8))
        crowned_positions = [(0, 5), (0, 7)]
        crowned_pieces = [move_board[crowned_positions[0]], move_board[crowned_positions[1]]]
        self.assertIn(2, crowned_pieces, "The chip should be crowned after reaching the opposite end.")

    def test_generate_valid_moves_capture(self):
        self.game = CheckersGame()
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
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)

        # Check that there is a capture move
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")

        # Apply the first valid move
        chosen_move = valid_moves[0]
        self.game.get_move_positions(self.game.board, chosen_move.reshape((8, 8)))
        self.game.board = chosen_move.reshape((8, 8))
        self.game.update_game_scores()

        # Check if the captured piece is removed
        self.assertEqual(self.game.board[5, 2], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[6, 3], 0, "The piece was moved.")
        self.assertEqual(self.game.board[4, 1], 1, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 12, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")



    def test_generate_valid_moves_capture_by_crown(self):
        self.game = CheckersGame()
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
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)

        # Check that there is a capture move
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")

        # Apply the first valid move
        chosen_move = valid_moves[0]
        self.game.get_move_positions(self.game.board, chosen_move.reshape((8, 8)))
        self.game.board = chosen_move.reshape((8, 8))
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
        self.assertEqual(player1_score, 0, "Initial score for player 1 should be 0.")
        self.assertEqual(player2_score, 0, "Initial score for player -1 should be 0.")

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
        self.assertEqual(player1_score, 11, "Score for player 1 should be 11.")
        self.assertEqual(player2_score, 11, "Score for player -1 should be 11.")

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
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)
        
        # Ensure there are valid moves
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")
        
        # Check if the valid moves include the correct capture moves
        capture_move_found = any(
            (move.reshape((8, 8))[3, 0] == 2 and move.reshape((8, 8))[4, 1] == 0 and move.reshape((8, 8))[5, 2] == 0) or
            (move.reshape((8, 8))[3, 4] == 2 and move.reshape((8, 8))[4, 3] == 0 and move.reshape((8, 8))[5, 2] == 0)
            for move in valid_moves
        )
        self.assertTrue(capture_move_found, "Capture moves by the crowned piece should be included in valid moves.")

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.get_move_positions(self.game.board, chosen_move.reshape((8, 8)))
        self.game.board = chosen_move.reshape((8, 8))
        self.game.update_game_scores()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[4, 1], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[5, 2], 0, "The piece was moved.")
        self.assertEqual(self.game.board[3, 0], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly, both players have 11 points since each of them have a piece only each.
        self.assertEqual(self.game.player1_score, 11, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 11, "Player -1's score should be updated to reflect the remaining pieces.")

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
        
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), 1)
        
        # Ensure there are valid moves
        self.assertGreater(len(valid_moves), 0, "There should be at least one valid move for player 1.")
        
        # Check if the valid moves include the correct multiple capture moves
        capture_move_found = any(
            (move.reshape((8, 8))[1, 6] == 2 and move.reshape((8, 8))[2, 5] == 0 and move.reshape((8, 8))[4, 3] == 0 and move.reshape((8, 8))[6, 1] == 0)
            for move in valid_moves
        )
        self.assertTrue(capture_move_found, "The multiple jump capture move should be included in valid moves.")

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.get_move_positions(self.game.board, chosen_move.reshape((8, 8)))
        self.game.board = chosen_move.reshape((8, 8))
        self.game.update_game_scores()
        self.game.print_board()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[2, 5], 0, "The first captured piece should be removed.")
        self.assertEqual(self.game.board[4, 3], 0, "The second captured piece should be removed.")
        self.assertEqual(self.game.board[6, 1], 0, "The third captured piece should be removed.")
        self.assertEqual(self.game.board[7, 0], 0, "The piece was moved.")
        self.assertEqual(self.game.board[1, 6], 2, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly
        self.assertEqual(self.game.player1_score, 11, "Player 1's score should be updated to reflect the capture.")
        self.assertEqual(self.game.player2_score, 9, "Player -1's score should be updated to reflect the remaining pieces.")

        


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
        valid_moves = self.game.generate_valid_moves(self.game.get_board_state(), -1) # this test the turn is for the -1 player
        
        # Ensure there are valid moves
        self.assertEqual(len(valid_moves), 2, "Expects to have 2 Moves for the player -1.")
        
        # Check if the valid moves include the correct capture moves
        capture_move_found = any(
            (move.reshape((8, 8))[6, 1] == -1 and move.reshape((8, 8))[5, 2] == 0 and move.reshape((8, 8))[4, 3] == 0) or
            (move.reshape((8, 8))[6, 3] == -1 and move.reshape((8, 8))[5, 2] == 0 and move.reshape((8, 8))[4, 1] == 0)
            for move in valid_moves
        )
        self.assertTrue(capture_move_found, "Capture moves by the crowned piece should be included in valid moves.")

        # Apply the first valid capture move
        chosen_move = valid_moves[0]
        self.game.get_move_positions(self.game.board, chosen_move.reshape((8, 8)))
        self.game.board = chosen_move.reshape((8, 8))
        self.game.update_game_scores()

        # Check if the captured pieces are removed
        self.assertEqual(self.game.board[4, 1], 0, "The captured piece should be removed.")
        self.assertEqual(self.game.board[5, 2], 0, "The piece was moved.")
        self.assertEqual(self.game.board[6, 3], -1, "The player's piece should move to the capture position.")

        # Check if the score is updated correctly, both players have 11 points since each of them have a piece only each.
        self.assertEqual(self.game.player1_score, 10, "Player 1's score should be 10, since -1 has 2 pieces.")
        self.assertEqual(self.game.player2_score, 12, "Player -1's score should be 12, since 1 has no pieces.")


if __name__ == "__main__":
    unittest.main()
