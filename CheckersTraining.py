
import math
import time
from CheckersGame import CheckersGame, debug_print, DEBUG_ON
from CheckersNN import CheckersNN
import numpy as np


class CheckersTraining(CheckersGame):

    def __init__(self):
        super().__init__()
        self.nn = CheckersNN()  # Initialize neural network

    def simulate_play_on_board(self, board, move, player):
        new_board = self.update_score_and_board(move, player, board)
        return new_board.flatten()

    def calculate_reward(self, player):
        if self.player1_score > self.player2_score:
            return 100 if player == 1 else -100
        elif self.player1_score < self.player2_score:
            return -100 if player == 1 else 100
        return 0  # Tie or no significant change

    def have_nn_select_moves(self, valid_moves, player):
        best_percentage = -math.inf 
        best_move = None
        flat_board_with_player = None
        for current_move in valid_moves:
            new_board = self.board.copy()
            # we send the current player as input and the state of the board
            flat_board = self.simulate_play_on_board(new_board, current_move, player)
            flat_board_with_player = np.array([player] + flat_board.tolist()).reshape(1, -1)
            score_move_percentage = self.nn.predict(flat_board_with_player)[0][0]
            if best_percentage < score_move_percentage:
                best_move = current_move
                best_percentage = score_move_percentage
        return best_move, flat_board_with_player


    def run_simulation(self):
        """
        Run a slow simulation of the game where players make random valid moves.
        """
        print(f"Started in debug mode: {DEBUG_ON} ")
        total_games = 0
        while True: # run until interrupted by the user this way it runs many games one after the other
            print(f"Total Games played: {total_games}")
            self.board = self.initialize_board()
            plays_from_players = {
                1: [],
                -1: []
            }
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

                # let the nn select a move  
                chosen_move, flat_board_with_player = self.have_nn_select_moves(valid_moves, player)
                if not chosen_move:
                    raise ("There is a problem, no move was chosen.")
                
                plays_from_players[player].append(flat_board_with_player) # [0])
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
            # ended game so we can train based off result:
            # Train the neural network
            # plays_from_players[player].append(flat_board)
            if self.player1_score != self.player2_score:
                for play in plays_from_players[1]: # player 1
                    reward = self.calculate_reward(1)
                    self.nn.train(np.array([play]), np.array([reward]))
                for play in plays_from_players[-1]: # player -1
                    reward = self.calculate_reward(-1)
                    self.nn.train(np.array([play]), np.array([reward]))

# Quick run:
if __name__ == "__main__":
    game = CheckersTraining()
    game.run_simulation()
