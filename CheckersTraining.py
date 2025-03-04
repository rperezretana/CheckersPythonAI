import math
import random
import time
import numpy as np
import os
from CheckersRulesGame import CheckersRulesGame
from Enumerators import Engines
from MathTooling import average, clean_string, transform_key_to_base72
from SimpleConfig import (
    EXECUTE_SAVE_ASYNC,
    PLAYER_1_ENGINE,
    PLAYER_2_ENGINE,
    RANDOM_FIRST_PLAYS,
    STOP_CHECK_INTERVAL,
    TRAINING,
    debug_print,
    DEBUG_ON,
    RANDOM_FIRST_PLAYS,
    SAVES_INTERVAL
)
from CheckersNN import CheckersNN
import tensorflow as tf
import json
from concurrent.futures import ThreadPoolExecutor
import string



tf.compat.v1.enable_eager_execution()

class CheckersTraining(CheckersRulesGame):

    def __init__(self):
        super().__init__()
        self.save_interval = SAVES_INTERVAL # when NN is not running this has to be a big number
        self.total_games = 0
        self.tie_detected = False
        self.save_directory = "model_saves"
        self.predicted_player1 = 0
        self.predicted_player2 = 0
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        self.nn = CheckersNN()  # Initialize neural network
        # load NN model if exist and if it is going to be used:
        if Engines.NN in [PLAYER_1_ENGINE, PLAYER_2_ENGINE] and\
            os.path.exists(os.path.join(self.save_directory,'checkers_model.h5')):
            self.nn.load(os.path.join(self.save_directory, "checkers_model.h5"))
        self.monte_carlo_scoring = dict()
        self.executor = ThreadPoolExecutor(max_workers=1)  # Create an executor for asynchronous tasks
        self.player_1_win_count = 0
        self.player_2_win_count = 0
        self.tie_games = 0
        self.new_branches_created = 0
        self.old_branches_updated = 0
        self.inverted_branches_used = 0
        self.loop_run = True


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
        if  self.player1_score == self.player2_score:
            return -self.total_moves  # slight penalization on tie
        res = 0
        if self.player1_score > self.player2_score:
            res = self.player1_score if player == 1 else -self.player1_score
        elif self.player1_score < self.player2_score:
            res = -self.player2_score if player == 1 else self.player2_score
        return (res * 10) - self.total_moves # moves become a penalizing factor, the longer the game takes the more it is punished

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
    

    def mirror_play(self, key):
        player = int(key[:2]) * -1 if key[0] == '-' else int(key[0]) * -1
        key = key[2:] if key[0] == '-' else key[1:]
        key = key.replace('-2', '8').replace('-1', '9')[::-1]
        key = key.replace('2', '-2').replace('1', '-1').replace('8', '2').replace('9', '1')
        return f'{player}{key}'

    
    def clean_dict_keys(self, input_dict):
        # Create a new dictionary with cleaned keys
        cleaned_dict = {}
        keys_to_replace = list(input_dict.keys())
        total_keys = len(keys_to_replace)
        for i, value in enumerate(keys_to_replace):
            cleaned_key = clean_string(value)
            cleaned_dict[cleaned_key] = input_dict[value]
            # Print progress every 1%
            if i%1000 == 0:
                print(f"Progress: {((i + 1) / total_keys) * 100:.2f}%")
        return cleaned_dict

    def save_model_periodically(self, game_count):
        if game_count % self.save_interval == 0 or self.loop_run == False:
            self.reduce_cache_size()
            if EXECUTE_SAVE_ASYNC:
                print("Saving file executing async")
                future = self.executor.submit(self.save_status)
                future.add_done_callback(self.save_callback)
            else:
                # execute sync:
                self.save_status()

    def save_callback(self, future):
        try:
            future.result()
        except Exception as e:
            print(f"Error occurred during save: {e}")

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
        print(f"Player 1 score: {self.player_1_win_count}")
        print(f"Player -1 score: {self.player_2_win_count}")
        print(f"Total Games played: {self.total_games}")
        p1, p2, p3 = self.calculate_percentages(self.player_1_win_count, self.player_2_win_count, self.tie_games)
        print(f"{p1}% ({PLAYER_1_ENGINE}) - {p2}% ({PLAYER_2_ENGINE}) - {p3}% ties")
        print("Saving results in a file...")
        self.remove_zero_values(self.monte_carlo_scoring)

        src = os.path.join(self.save_directory, f"game_status.json")
        dst = os.path.join(self.save_directory, f"game_status2.json")
        if os.path.exists(dst):
            os.remove(os.path.join(self.save_directory,'game_status2.json'))
        if os.path.exists(src):
            os.rename(src, dst)

        dst = os.path.join(self.save_directory, f"game_status.json")
        status = {
            'total_games': self.total_games,
            'valid_moves_memo': dict(),  # Convert to list for JSON
            'transition_memo': dict(),  # Convert to list for JSON
            'monte_carlo_scoring': dict(self.monte_carlo_scoring)
        }
        with open(dst, 'w') as f:
            json.dump(status, f)
        print(f"Status saved after {self.total_games} games.")
        if TRAINING:
            self._save_model()


    def reduce_cache_size(self):
        self.valid_moves_memo.remove_least_used()
        self.transition_memo.remove_least_used()


    def load_status(self):
        src = os.path.join(self.save_directory, f"game_status.json")
        if os.path.exists(src):
            with open(src, 'r') as f:
                status = json.load(f)
                self.total_games = status.get('total_games', 0)
                self.monte_carlo_scoring =  dict(status.get('monte_carlo_scoring', []))


    def have_mc_select_moves(self, valid_moves, player):
        best_percentage = -math.inf 
        best_move = None
        flat_board_with_player = None
        for current_move in valid_moves:
            new_board = self.board.copy()
            flat_board = self.simulate_play_on_board(new_board, current_move, player)
            flat_board_with_player = self.filter_and_flatten_board(flat_board, player)
            state = clean_string(f"{flat_board_with_player}")
            inv_state = transform_key_to_base72(self.mirror_play(state))
            state = transform_key_to_base72(state)
            
            if state in self.monte_carlo_scoring:
                score_move_percentage = self.monte_carlo_scoring[state]
            elif inv_state in self.monte_carlo_scoring:
                score_move_percentage = self.monte_carlo_scoring[inv_state]
            else:
                score_move_percentage = 0
            debug_print(f"Predicted: {score_move_percentage} for {state}")
            if best_percentage < score_move_percentage:
                best_move = current_move
                best_percentage = score_move_percentage
            elif best_percentage == score_move_percentage:
                # choses random, mostly useful if they are unknown zeros
                best_move = random.choice([best_move, current_move])
                best_percentage = score_move_percentage
        if player == -1:
            self.predicted_player1 = best_percentage
        else:
            self.predicted_player2 = best_percentage
        debug_print(f"Predicted: {self.predicted_player1} - player 1")
        debug_print(f"Predicted: {self.predicted_player2} - player -1")
        return best_move, flat_board_with_player


    def calculate_percentages(self, mc_wins, random_wins, ties):
        total_events = mc_wins + random_wins + ties
        if total_events == 0:
            return 0, 0, 0
        
        percentage_mc_wins = (mc_wins / total_events) * 100
        percentage_random_wins = (random_wins / total_events) * 100
        percentage_ties = (ties / total_events) * 100
        
        return percentage_mc_wins, percentage_random_wins, percentage_ties


    def update_reward_monte_carlo_score(self, inputs, reward):
        # inputs normally shaped as flat_board_with_player
        state = clean_string(f"{inputs}")
        inv_state = self.mirror_play(state)
        # reward = reward / 100
        state = transform_key_to_base72(state)
        inv_state = transform_key_to_base72(inv_state)
        if state in self.monte_carlo_scoring:
            self.monte_carlo_scoring[state] = average(reward, self.monte_carlo_scoring[state])
            self.old_branches_updated += 1
        elif inv_state in self.monte_carlo_scoring:
            self.monte_carlo_scoring[inv_state] = average(reward, self.monte_carlo_scoring[inv_state])
            self.inverted_branches_used += 1
        else:
            self.monte_carlo_scoring[state]=reward
            self.new_branches_created += 1


    def play_with_selected_engine(self, valid_moves, player):
        if RANDOM_FIRST_PLAYS > self.total_moves:
            return self.select_random_play(valid_moves, player)
        if player == 1:
            if PLAYER_1_ENGINE == Engines.MC:
                return self.have_mc_select_moves(valid_moves, player)
            elif PLAYER_1_ENGINE == Engines.NN:
                return self.have_nn_select_moves(valid_moves, player)
            else:
                return self.select_random_play(valid_moves, player)
        if player == -1:
            if PLAYER_2_ENGINE == Engines.MC:
                return self.have_mc_select_moves(valid_moves, player)
            elif PLAYER_2_ENGINE == Engines.NN:
                return self.have_nn_select_moves(valid_moves, player)
            else:
                return self.select_random_play(valid_moves, player)
        return None, None


    def run_simulation(self):
        self.load_status()
        print(f"Started in debug mode: {DEBUG_ON} ")
        self.player_1_score = 0
        self.player_2_score = 0
        self.tie_games = 0
        try:
            while self.loop_run:
                self.new_branches_created = 0
                self.old_branches_updated = 0
                self.inverted_branches_used = 0
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

                    chosen_move, flat_board_with_player =  self.play_with_selected_engine(valid_moves, player)
                    if not chosen_move:
                        raise ("There is a problem, no move was chosen.")
                    
                    plays_from_players[player].append(flat_board_with_player)
                    self.update_score_and_board(chosen_move, player)

                    self.update_reward_monte_carlo_score(flat_board_with_player, self.calculate_reward(player))
                    
                    if self.detect_loop():
                        self.tie_detected = True
                        debug_print("Loop detected. Game ends in a tie.")
                        debug_print(f"Total moves: {self.total_moves}")
                        break

                    if DEBUG_ON:
                        self.print_board()
                        debug_print(f"Player 1 score: {self.player1_score} ({PLAYER_1_ENGINE})")
                        debug_print(f"Player -1 score: {self.player2_score} ({PLAYER_2_ENGINE})")
                        debug_print(f"Total moves: {self.total_moves}")
                        debug_print(f"Total Games played: {self.total_games}")
                        self.update_game_scores()
                        time.sleep(1)

                    if player == 1:
                        self.player1_moves += 1
                    else:
                        self.player2_moves += 1

                    player = -player  # reverse player playing

                    self.total_moves += 1
                    if self.total_moves >= self.move_limit:
                        # self.tie_detected = True
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
                if self.player1_score > self.player2_score:
                    self.player_1_win_count += 1
                elif self.player1_score < self.player2_score:
                    self.player_2_win_count += 1
                else:
                    self.tie_games += 1

                debug_print(f"Delivering reward for P{1}: {reward}")
                for play in plays_from_players[1]:
                    self.update_reward_monte_carlo_score(play, reward)
                    if TRAINING:
                        self.nn.train(np.array(play), np.array([reward]))
                
                reward = self.calculate_reward(-1)
                debug_print(f"Delivering reward for P{-1}: {reward}")
                for play in plays_from_players[-1]:
                    self.update_reward_monte_carlo_score(play, reward)
                    if TRAINING:
                        self.nn.train(np.array(play), np.array([reward]))
                self.save_model_periodically(self.total_games) # periodically
                if self.total_games % STOP_CHECK_INTERVAL == 0:
                    print(f"MC new branches: {self.new_branches_created} - Old Branches {self.old_branches_updated} - Mirror used {self.inverted_branches_used}")
                    print(f"Global score: P1: {self.player_1_win_count}  ({PLAYER_1_ENGINE}) P-1: {self.player_2_win_count}" \
                        f"({PLAYER_2_ENGINE}) - Tie {self.tie_games} - MC:{len(self.monte_carlo_scoring)}")
                    p1, p2, p3 = self.calculate_percentages(self.player_1_win_count, self.player_2_win_count, self.tie_games)
                    print(f"{p1}% ({PLAYER_1_ENGINE}) - {p2}% ({PLAYER_2_ENGINE}) - {p3}% ties")
                    if self.check_and_delete_stop_file():
                        break
            print("Preparing to stop the application...")
            self.save_model_periodically(self.total_games) # end of game loop, game is stopping
        finally:
            self.shutdown_executor()  # Ensure the executor is shut down before exiting



    def remove_zero_values(self, input_dict):
        print(f"Removing 0 values...")
        keys_to_remove = [key for key, value in input_dict.items() if value == 0]
        for key in keys_to_remove:
            del input_dict[key]
        print(f'Removed {len(keys_to_remove)} keys.')

    
    def save_game_results(self):
        # Generate a random 5 characters string
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        print(f"Storing Training Data ...")
        self.remove_zero_values(self.monte_carlo_scoring)
        status = {
            'total_games': self.total_games,
            'monte_carlo_scoring': dict(self.monte_carlo_scoring)
        }
        new_file = os.path.join(self.save_directory,f'{random_name}.json')
        with open(new_file, 'w') as f:
            json.dump(status, f)


    def check_and_delete_stop_file(self):
        """
        This helps to stop the program and garantee no damage to the files.
        Create a file stop.txt to stop the program and it will stop next time the back up finish.
        """
        stop_file_path = os.path.join(self.save_directory, 'stop.txt')
        if os.path.exists(stop_file_path):
            print("REQUEST TO STOP DETECTED.")
            os.remove(stop_file_path)
            self.loop_run = False
            return True
        return False

    def shutdown_executor(self):
        print("Executor still running, waiting for it to complete")
        self.executor.shutdown(wait=True)
        print("Executor shutdown complete")

if __name__ == "__main__":
    game = CheckersTraining()
    game.run_simulation()
