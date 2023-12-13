from flask import request, jsonify, render_template
from flask import Flask
from flask_cors import CORS

import pandas as pd
import numpy as np

from datetime import datetime
import csv
import os
import argparse
import json
import csv
import random

DIR_EXPERIMENTS = "experiments"
DIR_HUMAN = "human"
DIR_RAWDATA = "raw_responses"

DIR_DATA = "data"

class FlaskAppWrapper(object):

    def __init__(self, app, experiment_id, input_file_name, num_questions_per_user, seed, **configs):
        
        #Initialize
        self.app = app
        self.configs(**configs)

        self.assigning_user_number = 0

        np.random.seed(seed)

        self.experiment_id = experiment_id
        self.saving_loc = os.path.join(DIR_EXPERIMENTS, self.experiment_id, DIR_HUMAN)
        self.raw_saving_loc = os.path.join(DIR_EXPERIMENTS, self.experiment_id, DIR_HUMAN, DIR_RAWDATA)
        os.makedirs(self.saving_loc, exist_ok=True)
        os.makedirs(self.raw_saving_loc, exist_ok=True)

        # load in the data
        total_input_data = []
        with open(os.path.join(DIR_DATA, input_file_name), 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                total_input_data.append(row)
        np.random.shuffle(total_input_data)
        
        # split up the data into chunks that the user can handle
        m = num_questions_per_user
        sub_lists = [total_input_data[i:i + m] for i in range(0, len(total_input_data), m)]
        self.input_data = sub_lists

        # enable CORS:
        CORS(self.app)

        # add endpoints for flask app
        self.add_endpoint('/api/heartbeat', 'heartbeat', self.heartbeat, methods=["GET"])
        self.add_endpoint('/api/send_response', 'send_response', self.send_response, methods=["POST"])
        self.add_endpoint('/', 'index', self.index)

        
    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint = None, endpoint_name = None, handler = None, methods = ['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods = methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def index(self):
        
        #self.assigning_user_number += 1 #moved this to send_response

        data_for_user = self.input_data[self.assigning_user_number % len(self.input_data)]

        return render_template(
            'index.html', 
            user_number = self.assigning_user_number, 
            stimuli_batch = data_for_user,
        )
        
    def heartbeat(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def send_response(self):
        
        data = request.json.get('data')["trials"]
        user_number = request.json.get('user_number')

        #save raw data
        file_name = os.path.join(self.raw_saving_loc, "rawResponse" + str(user_number) + ".json")
        with open(file_name, 'w') as file:
            json.dump(data, file)

        file_name = os.path.join(self.saving_loc, "human_unaggregated_results.csv")
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)

            # If the file did not exist, write the header first
            if not file_exists:
                writer.writerow(['task_id', 'runtime', 'rating'])

            for row in data:
                if row["trial_type"] == 'survey-likert':
                    task_id = row["task_id"]
                    if (row["response"]["Q0"] ==""):
                        continue
                    rating = row["response"]["Q0"] + 1
                    time = row["time_elapsed"] / 10000

                    new_row = [task_id, time, rating]

                    # Write the data row
                    writer.writerow(new_row)

        self.assigning_user_number += 1

        self.aggregate_responses()

        return jsonify({
            'response': "OK",
        })

    def aggregate_responses(self):
        
        unaggregated_file_name = os.path.join(self.saving_loc, "human_unaggregated_results.csv")
        df = pd.read_csv(unaggregated_file_name)

        # Function to calculate probabilities
        def calculate_probs(ratings):
            values, counts = np.unique(ratings, return_counts=True)
            probs = counts / counts.sum()
            return probs, values

        # Group by 'task_id' and apply custom operations
        result = df.groupby('task_id').agg({
            'rating': lambda x: calculate_probs(x),
            'runtime': 'mean'
        })

        # Split the rating tuple into two separate columns
        result['probs'], result['support'] = zip(*result['rating'])
        result.drop('rating', axis=1, inplace=True)

        # Reset index to make 'task_id' a column again
        result = result.reset_index()

        def list_to_string(lst):
            """Converts a list to a string with elements separated by commas."""
            return '[' + ', '.join(map(str, lst)) + ']'
        result['probs'] = result['probs'].apply(list_to_string)
        result['support'] = result['support'].apply(list_to_string)

        aggregated_file_name = os.path.join(self.saving_loc, "human_aggregated_results.csv")
        result.to_csv(aggregated_file_name, index=False)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("input_file_name")
    parser.add_argument("--port", type=int, default=3210, help="Which port to run the app on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Which host to run the app on.")
    parser.add_argument("--num_questions_per_user", type=int, default=24, help="How many questions each user is presented with")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for shuffling of questions")
    args = parser.parse_args()

    app = FlaskAppWrapper(Flask(
        __name__,
        template_folder="/work/submit/juliush/llm_playground/NaLaPIP/human",
        static_folder="/work/submit/juliush/llm_playground/NaLaPIP/human/static",
    ), args.experiment_id, args.input_file_name, args.num_questions_per_user, args.seed)
    app.run(debug=True,port=args.port, host=args.host)

if __name__ == "__main__":
    main()