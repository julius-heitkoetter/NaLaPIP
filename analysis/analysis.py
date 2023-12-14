import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from scipy.odr import ODR, Model, RealData
from scipy.stats import t, wasserstein_distance
from scipy.interpolate import make_interp_spline
import pandas as pd
import os
import ast

DIR_EXPERIMENTS = "experiments"
DIR_ANALYSIS = "analysis"
DIR_HUMAN = "human"
DIR_WEBPPL = "webppl"

def fit(x,y, x_err, y_err, CL = 0.95):
    # Define a linear function to fit
    def linear_func(p, x):
        return p[0] * x + p[1]

    # Create a model for fitting
    linear_model = Model(linear_func)

    # Create RealData object using data and errors
    data = RealData(x, y, sx=x_err, sy=y_err)

    # Set up Orthogonal Distance Regression (ODR) with the model and data
    odr = ODR(data, linear_model, beta0=[0., 1.])

    # Run the regression
    out = odr.run()

    # Extract the parameters
    slope, intercept = out.beta

    # Prepare data for plotting the fit and confidence interval
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept

    # Calculate residuals and the standard error of the estimate
    residuals = y - (slope * x + intercept)
    df = len(x) - 2  # Degrees of freedom
    stderr = np.sqrt(np.sum(residuals**2) / df)

    # Calculate the confidence interval
    t_val = t.ppf(CL + (1-CL)/2, df)  # CL, CI, two-tailed test
    ci = t_val * stderr * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    return x_fit, y_fit, ci, slope, intercept

def get_trace_of_histogram(data, bins):

    hist, bin_edges = np.histogram(data, bins=bins)
    padded_hist = np.concatenate(([0], hist, [0]))
    trace = []

    for i in range(len(padded_hist)-1):
        trace.append((padded_hist[i] + padded_hist[i+1])/2)

    spline = make_interp_spline(bin_edges, trace)

    bin_edges_spline = np.linspace(min(bin_edges), max(bin_edges), 500)
    trace_spline = spline(bin_edges_spline)

    return bin_edges_spline, trace_spline


def plot_human_versus_model_likert(experiment_id):

    # load data into dataframes
    human_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_HUMAN, "human_aggregated_results.csv"))
    model_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, "simulator_results.csv"))
    
    human_df.set_index('task_id', inplace=True)
    model_df.set_index('task_id', inplace=True)

    # get means and standard deviations
    def weighted_mean(row):
        probs = ast.literal_eval(row['probs'] )
        support = ast.literal_eval(row['support'] )
        return np.sum(np.array(probs) * np.array(support))

    def weighted_std(row):
        probs = ast.literal_eval(row['probs'] )
        support = ast.literal_eval(row['support'] )
        mean = weighted_mean(row)
        variance = np.sum(probs * (np.array(support) - mean)**2)
        
        #TODO: remove this line
        if variance == 0:
            return 1.5
        
        return np.sqrt(variance)

    human_df['human_mean'] = human_df.apply(weighted_mean, axis=1)
    human_df['human_std'] = human_df.apply(weighted_std, axis=1)
    model_df['model_mean'] = model_df.apply(weighted_mean, axis=1)
    model_df['model_std'] = model_df.apply(weighted_std, axis=1)

    merged_df = pd.merge(human_df, model_df, left_index=True, right_index=True, how='outer')

    #TODO: fix this to make it more robust
    N_human = 6
    N_model = 10

    model_ratings = np.array(merged_df["model_mean"].tolist()) 
    model_ratings_err = np.array(merged_df["model_std"].tolist()) / np.sqrt(N_model)
    human_ratings = np.array(merged_df["human_mean"].tolist())
    human_ratings_err = np.array(merged_df["human_std"].tolist()) / np.sqrt(N_human)


    #TODO: clean this up
    """ensemble_index_number = np.floor(np.arange(1,49,1) / 8)
    #ensemble_index_number = np.arange(1,49,1) % 8
    color_map = {
        0: "black",
        1: "blue",
        2: "red",
        3: "green",
        4: "purple",
        5: "orange",
        6: "yellow",
        7: "pink",
    }"""

    x_fit, y_fit, ci, slope, intercept = fit(model_ratings, human_ratings, x_err=model_ratings_err, y_err=human_ratings_err)

    plt.figure(figsize=(12, 12))
    plt.errorbar(model_ratings, human_ratings, xerr=model_ratings_err, yerr=human_ratings_err, fmt='o',label='Data')
    #for xi, yi, xe, ye, et in zip(model_ratings, human_ratings, model_ratings_err, human_ratings_err, ensemble_index_number):
    #    plt.errorbar(xi, yi, xerr=xe, yerr=ye, fmt='o', color=color_map[et])
    plt.plot(x_fit, y_fit, '-', label=f'y = {slope:.2f}x+{intercept:.2f}')
    plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel("NaLaPIP Model Rating")
    plt.ylabel("Human Rating")
    plt.title("Human versus NaLaPIP ratings for across tasks")
    plt.xlim(0.5, 7.5)
    plt.ylim(0.5, 7.5)
    plt.legend()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "model_human_scatter_plot.png"))

def plot_wasserstein_distance(experiment_id):

    # load data into dataframes
    human_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_HUMAN, "human_aggregated_results.csv"))
    model_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, "simulator_results.csv"))

    human_df.set_index('task_id', inplace=True)
    model_df.set_index('task_id', inplace=True)

    likert_scale = np.arange(1,8, 1)

    def get_likert_weights(row):
        probs = ast.literal_eval(row['probs'] )
        support = ast.literal_eval(row['support'] )
        likert_weights = np.zeros(7)
        for i, rating in enumerate(support):
            likert_weights[rating-1] = probs[i]
        return likert_weights

    def get_wasserstein_distance_human_model(row):

        return wasserstein_distance(
            likert_scale, 
            likert_scale, 
            row["human_likert_weights"], 
            row["model_likert_weights"],
        )
        
    def get_wasserstein_distance_human_random(row):
        rand = [np.random.random() for _ in range(7)]
        rand = rand / np.sum(rand)

        return wasserstein_distance(
            likert_scale, 
            likert_scale, 
            row["human_likert_weights"], 
            rand,
        )

    human_df["human_likert_weights"] = human_df.apply(get_likert_weights, axis=1)
    model_df["model_likert_weights"] = model_df.apply(get_likert_weights, axis=1)

    merged_df = pd.merge(human_df, model_df, left_index=True, right_index=True, how='outer')
    
    wasserstein_distances_human_model = merged_df.apply(get_wasserstein_distance_human_model, axis=1)
    wasserstein_distances_human_random = merged_df.apply(get_wasserstein_distance_human_random, axis=1)

    bin_max = max(max(wasserstein_distances_human_model), max(wasserstein_distances_human_random))
    bins = np.arange(0, bin_max, 0.25)

    model_bin_edges, model_trace = get_trace_of_histogram(wasserstein_distances_human_model, bins)
    random_bin_edges, random_trace = get_trace_of_histogram(wasserstein_distances_human_random, bins)

    plt.figure(figsize=(12, 8))
    plt.hist(wasserstein_distances_human_model, bins = bins, alpha = 0.2, label = "NaLaPIP", color="blue")
    plt.plot(model_bin_edges, model_trace, color="blue", alpha = 0.7, linewidth=3)
    plt.hist(wasserstein_distances_human_random, bins = bins, alpha = 0.2, label = "Random", color="orange")
    plt.plot(random_bin_edges, random_trace, color="orange", alpha = 0.7, linewidth=3)
    plt.ylabel("Number of Tasks")
    plt.xlabel("Wasserstein Metric")
    plt.title("Distance from human ratings")
    plt.legend()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "wasserstein_distance.png"))

np.random.seed(124)

experiment_id = "run-2023-12-12-1031"

os.makedirs(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS), exist_ok=True)
plot_human_versus_model_likert(experiment_id)
plot_wasserstein_distance(experiment_id)