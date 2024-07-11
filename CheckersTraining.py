
import time
import random
from CheckersGame import CheckersGame, debug_print, DEBUG_ON
from NeuralNetworkGPU import NeuralNetworkMultiGPU


class CheckersTraining(CheckersGame):

    def run_simulation(self):
        """
        Run a slow simulation of the game where players make random valid moves.
        """

        # self.board = np.array([
        #     [3, 0, 3, 0, 3, 0, 3, 0],
        #     [0, 3, 0, 3, 0, 3, 0, 3],  # (1, 6) should be valid
        #     [3, 0, 3, 0, 3, -1, 3, 0], # (2, 5) enemy tile
        #     [0, 3, 0, 3, 0, 3, 0, 3],  # (3, 4) should be open
        #     [3, 0, 3, -1, 3, 0, 3, 0], # (4, 3) player -1
        #     [0, 3, 0, 3, 0, 3, 0, 3],  # (5, 2) should be open
        #     [3, -1, 3, 0, 3, 0, 3, 0], # (6, 1) player -1
        #     [2, 3, 0, 3, 0, 3, 0, 3]   # (7, 0) player 1 crown
        # ])
        print(f"Started in debug mode: {DEBUG_ON} ")
        total_games = 0;
        while True: # run until interrupted by the user this way it runs many games one after the other
            print(f"Total Games played: {total_games}")
            self.board = self.initialize_board()
            self.place_players_chips()
            total_games+=1
            player = 1  # Start with player 1
            debug_print("Current Board:")
            self.print_board()  # Print the board for debugging
            while True:
                # second loop for a game
                valid_moves = self.generate_valid_moves(self.board, player)
                if not valid_moves:
                    debug_print(f"Player {player} has no valid moves. Game over.")
                    debug_print(f"Player 1 moves: {self.player1_moves}")
                    debug_print(f"Player -1 moves: {self.player2_moves}")
                    debug_print(f"Player 1 score: {self.player1_score}")
                    debug_print(f"Player -1 score: {self.player2_score}")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

                chosen_move = random.choice(valid_moves)
                self.update_score_and_board(chosen_move, player)
                # self.board = chosen_move[-1][-2:]  # Update board to the final position after the sequence
                
                # Check for loop
                if self.detect_loop():
                    debug_print("Loop detected. Game ends in a tie.")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

                self.print_board()
                debug_print(f"Player 1 score: {self.player1_score}")
                debug_print(f"Player -1 score: {self.player2_score}")
                debug_print(f"Total moves: {self.total_moves}")
                time.sleep(1)  # Wait for 1 second

                if player == 1:
                    self.player1_moves += 1
                else:
                    self.player2_moves += 1

                player = -player  # Switch player

                self.total_moves += 1
                if self.total_moves >= self.move_limit:
                    debug_print("Move limit reached. Game ends in a tie.")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

# Quick run:
if __name__ == "__main__":
    game = CheckersTraining()
    game.run_simulation()
