import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from scipy.odr import ODR, Model, RealData
from scipy.stats import t, wasserstein_distance
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
import pandas as pd
import os
import ast

DIR_EXPERIMENTS = "experiments"
DIR_ANALYSIS = "analysis"
DIR_HUMAN = "human"
DIR_WEBPPL = "webppl"
DIR_LLM = "llm_baseline"
DIR_BASELINE = "llm_baseline"

def sigmoid_func(x, a, b, c, d):
    return c / (1 + np.exp(-a * (x - b))) + d

def linear_func(p, x):
    return p[0] * x + p[1]

def fit(x,y, x_err, y_err, CL = 0.95):

    def objective(params):
        a, b, c, d = params
        c = 6
        d = 1
        x_transformed = sigmoid_func(y, a, b, c, d)
        return np.sum((x_transformed - y)**2)

    initial_guess = [1, 0, 6, 1]
    result = minimize(objective, initial_guess)
    a,b,c,d = result.x
    sigmoid_params = (a,b,c,d)

    # Transform y data using the fitted sigmoid function
    # y_transformed = sigmoid_func(x, sigmoid_a, sigmoid_b)
    x_transformed = sigmoid_func(x, *sigmoid_params)

    # Create a model for fitting
    linear_model = Model(linear_func)

    # Create RealData object using data and errors
    data = RealData(x_transformed, y, sx=x_err, sy=y_err)

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

    return x_fit, y_fit, ci, slope, intercept, sigmoid_params

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

def sigmoid_transform_histogram(values, probabilities, sigmoid_params, N = 500):

    # create CDF
    cumulative_probabilities = np.cumsum(probabilities)

    # Sample N times
    samples = []
    for _ in range(N):
        random_number = np.random.rand()  # Generate a random number between 0 and 1
        # Find the index where the random number would fit in the cumulative distribution
        index = np.where(cumulative_probabilities >= random_number)[0][0]
        samples.append(values[index])
    samples = np.array(samples)

    #apply transform
    transformed_samples = sigmoid_func(samples, *sigmoid_params)

    #rehistogram
    bin_width = (values[1] - values[0])/2
    bin_edges = np.arange(values[0] - bin_width, values[-1] + bin_width*2, 1)
    transformed_probabilities, _ = np.histogram(transformed_samples, bins=bin_edges)

    # re normalize
    transformed_probabilities = transformed_probabilities / np.sum(transformed_probabilities)

    return transformed_probabilities

def group_values_by_metric(metrics, values, n_groups = 3):
    # Pair each runtime with its index
    indexed_metrics = list(enumerate(metrics))

    # Sort the pairs by runtime
    indexed_metrics.sort(key=lambda x: x[1])

    # Calculate the size of each group
    group_size = len(metrics) // n_groups

    # Initialize groups
    metric_groups = [[] for _ in range(n_groups)]
    value_groups = [[] for _ in range(n_groups)]

    for i, (index, metric) in enumerate(indexed_metrics):
        # Determine the group for this index
        group_index = i // group_size
        if group_index > n_groups-1:  # Adjust if there are more than n_groups groups due to uneven division
            group_index = n_groups -1 

        # Add runtime and corresponding value to their respective groups
        metric_groups[group_index].append(metric)
        value_groups[group_index].append(values[index])

    return metric_groups, value_groups


def plot_human_versus_model_likert(experiment_id, model_dir = DIR_WEBPPL):

    # load data into dataframes
    human_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_HUMAN, "human_aggregated_results.csv"))
    if model_dir == DIR_WEBPPL:
        model_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, "simulator_results.csv"))
        model = "NaLaPIP"
    elif model_dir == DIR_LLM:
        model_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_LLM, "llm_aggregated_results.csv"))
        model = "LLM"
    else:
        raise "Model directory not found: " + str(model_dir)
    
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
    #ensemble_index_number = np.floor(np.arange(1,49,1) / 8)
    """ensemble_index_number = np.arange(1,49,1) % 8
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

    x_fit, y_fit, ci, slope, intercept, sigmoid_params = fit(model_ratings, human_ratings, x_err=model_ratings_err, y_err=human_ratings_err)

    modified_model_ratings = sigmoid_func(model_ratings, *sigmoid_params)
    print(sigmoid_params)

    plt.figure(figsize=(12, 12))
    #plt.errorbar(model_ratings, human_ratings, xerr=model_ratings_err, yerr=human_ratings_err, fmt='o',label='Data')
    plt.errorbar(modified_model_ratings, human_ratings, xerr=model_ratings_err, yerr=human_ratings_err, fmt='o',label='Data')
    #for xi, yi, xe, ye, et in zip(model_ratings, human_ratings, model_ratings_err, human_ratings_err, ensemble_index_number):
    #    xi_transformed = sigmoid_func(xi, sigmoid_a, sigmoid_b)
    #    plt.errorbar(xi_transformed, yi, xerr=xe, yerr=ye, fmt='o', color=color_map[et])
    plt.plot(x_fit, y_fit, '-', label=f'y = {slope:.2f}x+{intercept:.2f}')
    plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel(f"{model} Model Rating")
    plt.ylabel("Human Rating")
    plt.title(f"Human versus {model} ratings for across tasks")
    plt.xlim(0.5, 7.5)
    plt.ylim(0.5, 7.5)
    plt.legend()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, f"{model}_human_scatter_plot.png"))

    residuals = np.abs(modified_model_ratings - human_ratings)
    residuals_err = np.sqrt(model_ratings_err**2 + human_ratings_err**2)

    plt.figure(figsize=(12,12))
    plt.plot(np.linspace(0, 3, 500),np.linspace(0,3,500))
    plt.errorbar(human_ratings_err * np.sqrt(N_human), residuals, yerr = residuals_err, fmt='o')
    plt.xlabel("Human Uncertainty in Prediction")
    plt.ylabel(f"Residual between humans and {model}")
    plt.xlim(0, 3)
    plt.ylim(0,3)
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, f"{model}_human_residuals_scatter_plot.png"))

    return slope, intercept, sigmoid_params


def plot_wasserstein_distance(experiment_id, model_sigmoid_params=None, llm_sigmoid_params = None):

    # load data into dataframes
    human_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_HUMAN, "human_aggregated_results.csv"))
    model_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_WEBPPL, "simulator_results.csv"))
    llm_df = pd.read_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_BASELINE, "llm_aggregated_results.csv"))

    human_df.set_index('task_id', inplace=True)
    model_df.set_index('task_id', inplace=True)
    llm_df.set_index('task_id', inplace=True)

    likert_scale = np.arange(1,8, 1)

    def get_likert_weights(row):
        probs = ast.literal_eval(row['probs'] )
        support = ast.literal_eval(row['support'] )
        likert_weights = np.zeros(7)
        for i, rating in enumerate(support):
            likert_weights[rating-1] = probs[i]
        return likert_weights

    def get_wasserstein_distance_human_model(row):

        model_likert_weights = row["model_likert_weights"]
        if model_sigmoid_params is not None:
            model_likert_weights = sigmoid_transform_histogram(likert_scale, model_likert_weights, model_sigmoid_params)

        return wasserstein_distance(
            likert_scale, 
            likert_scale, 
            row["human_likert_weights"], 
            model_likert_weights,
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

    def get_wasserstein_distance_human_llm(row):

        model_likert_weights = row["llm_likert_weights"]
        if llm_sigmoid_params is not None:
            model_likert_weights = sigmoid_transform_histogram(likert_scale, model_likert_weights, llm_sigmoid_params)

        return wasserstein_distance(
            likert_scale, 
            likert_scale, 
            row["human_likert_weights"], 
            model_likert_weights,
        )

    human_df["human_likert_weights"] = human_df.apply(get_likert_weights, axis=1)
    model_df["model_likert_weights"] = model_df.apply(get_likert_weights, axis=1)
    llm_df["llm_likert_weights"] = llm_df.apply(get_likert_weights, axis=1)

    human_df.rename(columns={"runtime":"human_runtime"}, inplace=True)

    merged_df = pd.merge(human_df, model_df, left_index=True, right_index=True, how='outer')
    merged_df = pd.merge(merged_df, llm_df, left_index=True, right_index=True, how='outer')
    
    wasserstein_distances_human_model = merged_df.apply(get_wasserstein_distance_human_model, axis=1)
    wasserstein_distances_human_random = merged_df.apply(get_wasserstein_distance_human_random, axis=1)
    wasserstein_distances_human_llm = merged_df.apply(get_wasserstein_distance_human_llm, axis=1)
    bin_max = max(max(wasserstein_distances_human_model), max(wasserstein_distances_human_random), max(wasserstein_distances_human_llm))
    bins = np.arange(0, bin_max, 0.25)

    model_bin_edges, model_trace = get_trace_of_histogram(wasserstein_distances_human_model, bins)
    random_bin_edges, random_trace = get_trace_of_histogram(wasserstein_distances_human_random, bins)
    llm_bin_edges, llm_trace = get_trace_of_histogram(wasserstein_distances_human_llm, bins)

    plt.figure(figsize=(12, 8))
    plt.hist(wasserstein_distances_human_model, bins = bins, alpha = 0.2, label = "NaLaPIP", color="blue")
    plt.plot(model_bin_edges, model_trace, color="blue", alpha = 0.7, linewidth=3)
    plt.hist(wasserstein_distances_human_llm, bins = bins, alpha = 0.2, label = "GPT 4V", color="green")
    plt.plot(llm_bin_edges, llm_trace, color="green", alpha = 0.7, linewidth=3)
    plt.hist(wasserstein_distances_human_random, bins = bins, alpha = 0.2, label = "Random", color="orange")
    plt.plot(random_bin_edges, random_trace, color="orange", alpha = 0.7, linewidth=3)
    plt.ylabel("Number of Tasks")
    plt.xlabel("Wasserstein Metric")
    plt.title("Distance from human ratings")
    plt.legend()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "wasserstein_distance.png"))

    human_runtime = merged_df["human_runtime"].tolist()

    time_groups, model_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_model.tolist(), n_groups = 4)
    _, random_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_random.tolist(), n_groups = 4)
    _, llm_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_llm.tolist(), n_groups = 4)
    histograms_to_plot = [
        model_wasserstein_distance_groups[0],
        llm_wasserstein_distance_groups[0],
        random_wasserstein_distance_groups[0],
        [],
        model_wasserstein_distance_groups[1],
        llm_wasserstein_distance_groups[1],
        random_wasserstein_distance_groups[1],
        [],
        model_wasserstein_distance_groups[2],
        llm_wasserstein_distance_groups[2],
        random_wasserstein_distance_groups[2],
        ]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(histograms_to_plot, patch_artist=True)
    ax.set_xticklabels(['','Easy', '', '', '', 'Medium', '','', '', 'Hard', ''])
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "wasserstein_distance_by_runtime.png"))


np.random.seed(124)

experiment_id = "run-2023-12-12-1031"

os.makedirs(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS), exist_ok=True)
webppl_slope, webppl_intercept, webppl_sigmoid_params = plot_human_versus_model_likert(experiment_id, DIR_WEBPPL)
llm_slope, llm_intercept, llm_sigmoid_params = plot_human_versus_model_likert(experiment_id, DIR_LLM)

plot_wasserstein_distance(
    experiment_id, 
    model_sigmoid_params =webppl_sigmoid_params, 
    llm_sigmoid_params = llm_sigmoid_params,
)