import math
import random
import time
import numpy as np
import os
from CheckersGame import RANDOM_AFTER_PLAYS, CheckersGame, debug_print, DEBUG_ON, RANDOM_AFTER_PLAYS
from CheckersNN import CheckersNN
import tensorflow as tf
import json
from concurrent.futures import ThreadPoolExecutor



tf.compat.v1.enable_eager_execution()

class CheckersTraining(CheckersGame):

    def __init__(self):
        super().__init__()
        self.save_interval = 10
        self.total_games = 0
        self.tie_detected = False
        self.save_directory = "model_saves"
        self.predicted_player1 = 0
        self.predicted_player2 = 0
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        self.nn = CheckersNN()  # Initialize neural network
        self.nn.load(os.path.join(self.save_directory, "checkers_model.h5"))
        self.monte_carlo_scoring = dict()
        self.executor = ThreadPoolExecutor(max_workers=1)  # Create an executor for asynchronous tasks


    def simulate_play_on_board(self, board, move, player):
        new_board = self.update_score_and_board(move, player, board)
        return new_board.flatten()

    def calculate_reward(self, player):
        """
        The reward is proportional to the success of the game,
        so a game won with multiple chips will be consider even better.
        A losing player will be punished proportionally to the points its opponent got
        And rewarded proprotionally to the points the player got.
        ties are slighly punished
        """
        if self.tie_detected:
            return -50  # slight penalization on tie
        res = 0
        if self.player1_score > self.player2_score:
            res = self.player1_score if player == 1 else -self.player1_score
        elif self.player1_score < self.player2_score:
            res = -self.player2_score if player == 1 else self.player2_score
        else:
            return -5
        return res * 10

    def filter_and_flatten_board(self, board, player):
        # Convert the board to a list of lists if it is a NumPy array
        board_list = board.flatten() # .tolist() if isinstance(board, np.ndarray) else board
        # Only include playable tiles
        filtered_board = [tile for tile in board_list if tile != 3]
        # Include the player as the first element
        return np.array([player] + filtered_board).reshape(1, -1)


    def have_nn_select_moves(self, valid_moves, player):
        best_percentage = -math.inf 
        best_move = None
        flat_board_with_player = None
        for current_move in valid_moves:
            new_board = self.board.copy()
            flat_board = self.simulate_play_on_board(new_board, current_move, player)
            flat_board_with_player = self.filter_and_flatten_board(flat_board, player)
            score_move_percentage = self.nn.predict(flat_board_with_player)[0][0]
            debug_print(f"Predicted: {score_move_percentage} for {flat_board_with_player}")
            if best_percentage < score_move_percentage:
                best_move = current_move
                best_percentage = score_move_percentage
            elif best_percentage == score_move_percentage:
                best_move = random.choice([best_move, current_move])
                best_percentage = score_move_percentage
        if player == -1:
            self.predicted_player1 = best_percentage
        else:
            self.predicted_player2 = best_percentage
        debug_print(f"Predicted: {self.predicted_player1} - player 1")
        debug_print(f"Predicted: {self.predicted_player2} - player -1")
        return best_move, flat_board_with_player
    
    def select_random_play(self, valid_moves, player):
        chosen_move = random.choice(valid_moves)
        new_board = self.board.copy()
        flat_board = self.simulate_play_on_board(new_board, chosen_move, player)
        flat_board_with_player = self.filter_and_flatten_board(flat_board, player)
        return chosen_move, flat_board_with_player

    def save_model_periodically(self, game_count):
        if game_count % self.save_interval == 0:
            self.executor.submit(self.save_status)
            self.executor.submit(self._save_model)


    def _save_model(self):
        # Rotate the saved models
        if os.path.exists(os.path.join(self.save_directory,'checkers_model5.h5')):
            os.remove(os.path.join(self.save_directory,'checkers_model5.h5'))
        for i in range(4, 0, -1):
            src = os.path.join(self.save_directory, f"checkers_model{i}.h5")
            dst = os.path.join(self.save_directory, f"checkers_model{i+1}.h5")
            if os.path.exists(src):
                os.rename(src, dst)
        
        src = os.path.join(self.save_directory, f"checkers_model.h5")
        dst = os.path.join(self.save_directory, f"checkers_model1.h5")
        if os.path.exists(src):
            os.rename(src, dst)
        self.nn.save_model(src)
        print(f"Model saved after {self.total_games} games.")



    def save_status(self):
        print(f"************************************")
        print(f"Total moves: {self.total_moves}")
        print(f"Player 1 score: {self.player1_score}")
        print(f"Player -1 score: {self.player2_score}")
        print(f"Total Games played: {self.total_games}")

        dst = os.path.join(self.save_directory, f"game_status.json")
        status = {
            'total_games': self.total_games,
            'valid_moves_memo': list(self.valid_moves_memo.items()),  # Convert to list for JSON
            'transition_memo': list(self.transition_memo.items()),  # Convert to list for JSON
            'monte_carlo_scoring': dict(self.monte_carlo_scoring)
        }
        with open(dst, 'w') as f:
            json.dump(status, f)


    def load_status(self):
        src = os.path.join(self.save_directory, f"game_status.json")
        if os.path.exists(src):
            with open(src, 'r') as f:
                status = json.load(f)
                self.total_games = status.get('total_games', 0)
                self.valid_moves_memo = dict(status.get('valid_moves_memo', []))
                self.transition_memo = dict(status.get('transition_memo', []))
                self.monte_carlo_scoring =  dict(status.get('monte_carlo_scoring', []))


    def update_reward_monte_carlo_score(self, inputs, reward):
        # inputs normally shaped as flat_board_with_player
        state = f"{inputs}"
        reward = reward / 100
        if state in self.monte_carlo_scoring:
            self.monte_carlo_scoring[state]+=reward
        else:
            self.monte_carlo_scoring[state]=reward


    def run_simulation(self):
        self.load_status()
        print(f"Started in debug mode: {DEBUG_ON} ")
        while True:
            self.board = self.initialize_board()
            plays_from_players = {
                1: [],
                -1: []
            }
            self.total_games += 1
            player = random.choice([1, -1]) # Start with random player
            debug_print("Current Board:")
            self.print_board()  # Print the board for debugging
            self.tie_detected = False
            while True:
                valid_moves = self.generate_valid_moves(self.board, player)
                if not valid_moves:
                    break

                if RANDOM_AFTER_PLAYS > self.total_moves:
                    chosen_move, flat_board_with_player = self.select_random_play(valid_moves, player)
                else:
                    chosen_move, flat_board_with_player = self.have_nn_select_moves(valid_moves, player)
                if not chosen_move:
                    raise ("There is a problem, no move was chosen.")
                
                plays_from_players[player].append(flat_board_with_player)
                self.update_score_and_board(chosen_move, player)
                
                if self.detect_loop():
                    self.tie_detected = True
                    debug_print("Loop detected. Game ends in a tie.")
                    debug_print(f"Total moves: {self.total_moves}")
                    break

                self.print_board()
                debug_print(f"Player 1 score: {self.player1_score}")
                debug_print(f"Player -1 score: {self.player2_score}")
                debug_print(f"Total moves: {self.total_moves}")
                debug_print(f"Total Games played: {self.total_games}")
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


            self.update_game_scores()
            debug_print(f"Player {player} has no valid moves. Game over.")
            debug_print(f"Player 1 moves: {self.player1_moves}")
            debug_print(f"Player -1 moves: {self.player2_moves}")
            debug_print(f"Player 1 score: {self.player1_score}")
            debug_print(f"Player -1 score: {self.player2_score}")

            reward = self.calculate_reward(1)
            print(f"Delivering reward for P{1}: {reward}")
            for play in plays_from_players[1]:
                self.update_reward_monte_carlo_score(play, reward)
                self.nn.train(np.array(play), np.array([reward]))
            
            reward = self.calculate_reward(-1)
            print(f"Delivering reward for P{-1}: {reward}")
            for play in plays_from_players[-1]:
                self.update_reward_monte_carlo_score(play, reward)
                self.nn.train(np.array(play), np.array([reward]))

            self.save_model_periodically(self.total_games)

if __name__ == "__main__":
    game = CheckersTraining()
    game.run_simulation()
