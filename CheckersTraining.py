import math
import time
import numpy as np
import os
from CheckersGame import CheckersGame, debug_print, DEBUG_ON
from CheckersNN import CheckersNN
import tensorflow as tf


tf.compat.v1.enable_eager_execution()

class CheckersTraining(CheckersGame):

    def __init__(self):
        super().__init__()
        self.nn = CheckersNN()  # Initialize neural network
        self.save_interval = 50
        self.save_directory = "model_saves"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def simulate_play_on_board(self, board, move, player):
        new_board = self.update_score_and_board(move, player, board)
        return new_board.flatten()

    def calculate_reward(self, player):
        """
        The reward is proportional to the success of the game,
        so a game won with multiple chips will be consider even better.
        """
        res = 0
        if self.player1_score > self.player2_score:
            res = self.player1_score if player == 1 else -self.player2_score
        elif self.player1_score < self.player2_score:
            res = -self.player1_score if player == 1 else self.player2_score
        else:
            return -50
        return res * 100

    def have_nn_select_moves(self, valid_moves, player):
        best_percentage = -math.inf 
        best_move = None
        flat_board_with_player = None
        for current_move in valid_moves:
            new_board = self.board.copy()
            flat_board = self.simulate_play_on_board(new_board, current_move, player)
            flat_board_with_player = np.array([player] + flat_board.tolist()).reshape(1, -1)
            score_move_percentage = self.nn.predict(flat_board_with_player)[0][0]
            if best_percentage < score_move_percentage:
                best_move = current_move
                best_percentage = score_move_percentage
        return best_move, flat_board_with_player

    def save_model_periodically(self, game_count):
        if game_count % self.save_interval == 0:
            # Rotate the saved models
            for i in range(4, 0, -1):
                src = os.path.join(self.save_directory, f"checkers_model{i}.h5")
                dst = os.path.join(self.save_directory, f"checkers_model{i+1}.h5")
                if os.path.exists(src):
                    os.rename(src, dst)
            self.nn.save_model(os.path.join(self.save_directory, "checkers_model1.h5"))
            self.nn.save_model(os.path.join(self.save_directory, "checkers_model.h5"))
            print(f"Model saved after {game_count} games.")

    def run_simulation(self):
        print(f"Started in debug mode: {DEBUG_ON} ")
        total_games = 0
        while True:
            print(f"Total Games played: {total_games}")
            self.board = self.initialize_board()
            plays_from_players = {
                1: [],
                -1: []
            }
            self.place_players_chips()
            total_games += 1
            player = 1  # Start with player 1
            debug_print("Current Board:")
            self.print_board()  # Print the board for debugging
            while True:
                valid_moves = self.generate_valid_moves(self.board, player)
                if not valid_moves:
                    debug_print(f"Player {player} has no valid moves. Game over.")
                    debug_print(f"Player 1 moves: {self.player1_moves}")
                    debug_print(f"Player -1 moves: {self.player2_moves}")
                    debug_print(f"Player 1 score: {self.player1_score}")
                    debug_print(f"Player -1 score: {self.player2_score}")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

                chosen_move, flat_board_with_player = self.have_nn_select_moves(valid_moves, player)
                if not chosen_move:
                    raise ("There is a problem, no move was chosen.")
                
                plays_from_players[player].append(flat_board_with_player)
                self.update_score_and_board(chosen_move, player)
                
                if self.detect_loop():
                    debug_print("Loop detected. Game ends in a tie.")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

                self.print_board()
                debug_print(f"Player 1 score: {self.player1_score}")
                debug_print(f"Player -1 score: {self.player2_score}")
                debug_print(f"Total moves: {self.total_moves}")
                debug_print(f"Total Games played: {total_games}")
                if DEBUG_ON:
                    time.sleep(1)

                if player == 1:
                    self.player1_moves += 1
                else:
                    self.player2_moves += 1

                player = -player  

                self.total_moves += 1
                if self.total_moves >= self.move_limit:
                    debug_print("Move limit reached. Game ends in a tie.")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

            if self.player1_score != self.player2_score:
                for play in plays_from_players[1]:
                    reward = self.calculate_reward(1)
                    self.nn.train(np.array(play), np.array([reward]))
                for play in plays_from_players[-1]:
                    reward = self.calculate_reward(-1)
                    self.nn.train(np.array(play), np.array([reward]))

            self.save_model_periodically(total_games)

if __name__ == "__main__":
    game = CheckersTraining()
    game.run_simulation()
