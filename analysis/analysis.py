import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams.update({'font.size': 22})

import seaborn as sns

from scipy.odr import ODR, Model, RealData
from scipy.stats import t, wasserstein_distance
import scipy.stats as stats
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
import os
import ast

plt.style.use('ggplot')

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

    x_fit, y_fit, ci, slope, intercept, sigmoid_params = fit(model_ratings, human_ratings, x_err=model_ratings_err, y_err=human_ratings_err)

    modified_model_ratings = sigmoid_func(model_ratings, *sigmoid_params)

    # Adjust font size for better readability
    plt.rcParams.update({'font.size': 14})

    # Create a subplot layout with wider aspect ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Main plot (ax1)
    ax1.errorbar(modified_model_ratings, human_ratings, xerr=model_ratings_err, yerr=human_ratings_err, fmt='o', label='Data')
    ax1.plot(x_fit, y_fit, '-', label=f'y = {slope:.2f}x+{intercept:.2f}')
    ax1.fill_between(x_fit, y_fit - ci, y_fit + ci, color='gray', alpha=0.2, label='95% Confidence Interval')
    ax1.set_xlabel(f"{model} Model Rating")
    ax1.set_ylabel("Human Rating")
    ax1.set_title(f"Human versus {model} Ratings Across Tasks")
    ax1.set_xlim(0.5, 7.5)
    ax1.set_ylim(0.5, 7.5)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.legend()

    # Calculate deviations for subplot
    deviations = (human_ratings - (slope * modified_model_ratings + intercept)) #/ np.sqrt(model_ratings_err**2 + human_ratings_err**2)
    deviations_err = np.sqrt(model_ratings_err**2 + human_ratings_err**2)

    # Subplot (ax2) - Scatter plot for deviations
    ax2.errorbar(modified_model_ratings, deviations, xerr = model_ratings_err, yerr = deviations_err, fmt='o')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_yticks([-4,-2,0,2,4])
    ax2.set_xlabel(f"{model} Model Rating")
    ax2.set_ylabel("Std Devs from Fit")

    # Remove the title from the lower plot
    ax2.set_title("")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, f"{model}_human_scatter_plot_with_deviations.png"))

    statistic, p_value = stats.pearsonr(modified_model_ratings, human_ratings)
    print("r^2 coefficient: ", statistic)
    print("p value of r^2 coefficient: ", p_value)

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

    def get_wasserstein_distance_human_llm_unscaled(row):

        model_likert_weights = row["llm_likert_weights"]

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
    wasserstein_distances_human_llm = merged_df.apply(get_wasserstein_distance_human_llm, axis=1)
    wasserstein_distances_human_llm_unscaled = merged_df.apply(get_wasserstein_distance_human_llm_unscaled, axis=1)
    bin_max = max(max(wasserstein_distances_human_model), max(wasserstein_distances_human_llm_unscaled), max(wasserstein_distances_human_llm))
    bins = np.arange(0, bin_max, 0.25)

    model_bin_edges, model_trace = get_trace_of_histogram(wasserstein_distances_human_model, bins)
    random_bin_edges, llm_trace = get_trace_of_histogram(wasserstein_distances_human_llm, bins)
    llm_bin_edges, llm_unscaled_trace = get_trace_of_histogram(wasserstein_distances_human_llm_unscaled, bins)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("Set3", n_colors=6)[3:]
    plt.hist(wasserstein_distances_human_model, bins = bins, alpha = 0.6, color=colors[0])
    plt.plot(model_bin_edges, model_trace, color="red", alpha = 0.7 , label = "NaLaPIP", linewidth=3)
    plt.hist(wasserstein_distances_human_llm, bins = bins, alpha = 0.6, color=colors[1])
    plt.plot(llm_bin_edges, llm_trace, color="blue", alpha = 0.7, linewidth=3, label = "GPT 4V Few Shot")
    plt.hist(wasserstein_distances_human_llm_unscaled, bins = bins, alpha = 0.6, color=colors[2])
    plt.plot(random_bin_edges, llm_unscaled_trace, color=(204/255,204/255,0), alpha = 0.7, linewidth=3, label = "GPT 4V",)
    plt.ylabel("Number of Tasks")
    plt.xlabel("Wasserstein Metric")
    plt.title("Distance from human ratings")
    plt.ylim((0,12))
    plt.legend()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "wasserstein_distance.png"))

    human_runtime = merged_df["human_runtime"].tolist()

    time_groups, model_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_model.tolist(), n_groups = 4)
    _, llm_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_llm.tolist(), n_groups = 4)
    _, llm_unscaled_wasserstein_distance_groups = group_values_by_metric(human_runtime, wasserstein_distances_human_llm_unscaled.tolist(), n_groups = 4)
    histograms_to_plot = [
        model_wasserstein_distance_groups[0],
        llm_wasserstein_distance_groups[0],
        llm_unscaled_wasserstein_distance_groups[0],
        [],
        model_wasserstein_distance_groups[1],
        llm_wasserstein_distance_groups[1],
        llm_unscaled_wasserstein_distance_groups[1],
        [],
        model_wasserstein_distance_groups[2],
        llm_wasserstein_distance_groups[2],
        llm_unscaled_wasserstein_distance_groups[2],
        ]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(histograms_to_plot, patch_artist=True)
    colors = sns.color_palette("Set3", n_colors=6)[3:]
    labels = ['NaLaPIP', 'GPT 4V Few Shot', 'GPT 4V Zero Shot']
    label_hanldes = []
    group_pos = 0
    for i, group in enumerate(histograms_to_plot):
        if group:
            color_index = group_pos % len(colors)
            bp['boxes'][i].set_facecolor(colors[color_index])
            if len(label_hanldes) <= color_index:
                label_hanldes.append(mpatches.Patch(color=colors[color_index], label=labels[color_index]))
            group_pos += 1
        else:
            group_pos = 0
    ax.legend(handles = label_hanldes)
    ax.set_xticklabels(['','Easy', '', '', '', 'Medium', '','', '', 'Hard', ''])
    ax.set_xlabel("Task difficulty")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_title("Proximity to Human Ratings Across Various Categories")
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "wasserstein_distance_by_runtime.png"))
    plt.clf()

    difference_between_llm_model_wasserstein = wasserstein_distances_human_llm_unscaled - wasserstein_distances_human_model 
    # Creating a histogram of the data
    n, bins, patches = plt.hist(difference_between_llm_model_wasserstein, bins=30, density=True, alpha=0.6, color='g')

    # Best fit line (normal distribution)
    mu, std = norm.fit(difference_between_llm_model_wasserstein)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    # Calculate residuals
    hist_y, _ = np.histogram(difference_between_llm_model_wasserstein, bins=bins, density=True)
    mid_bin = (bins[:-1] + bins[1:]) / 2
    residuals = hist_y - norm.pdf(mid_bin, mu, std)

    # Create two subplots sharing the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Top plot (Histogram with normal distribution)
    ax1.hist(difference_between_llm_model_wasserstein, bins=9, density=True, alpha=0.6,label = "Data")
    ax1.plot(x, p, 'k', linewidth=2, label = "$\mathbb{N}(\mu = %.2f,  \sigma = %.2f)$" % (mu, std))
    ax1.set_title('Histogram and Normal Distribution Fit')
    ax1.set_ylabel('Density')

    # Bottom plot (Residuals)
    ax2.plot(mid_bin, residuals, 'o', label = "Residuals")
    ax2.set_ylabel('Residuals')
    ax2.set_xlabel('Difference between Wasserstein distance of LLM Baseline and NaLaPIP')
    ax2.set_yticks([-.4, -.2, 0, .2, .4])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, f"wasserstein_difference.png"))

    std_err = std / np.sqrt(len(difference_between_llm_model_wasserstein))
    t_statistic, p_value = stats.ttest_rel(difference_between_llm_model_wasserstein, [0]*len(difference_between_llm_model_wasserstein))
    print("P is of difference is: ", p_value)

def average_sublists(input_lists, sublist_size):
    """
    Averages sublists of a given size within a larger list of lists.

    :param input_lists: List of lists to be averaged.
    :param sublist_size: Size of each sublist to be averaged.
    :return: List of averaged sublists.
    """
    averaged_lists = []
    for i in range(0, len(input_lists), sublist_size):
        sublist = input_lists[i:i + sublist_size]
        averaged_sublist = [sum(col) / len(col) for col in zip(*sublist)]
        averaged_lists.append(averaged_sublist)

    return averaged_lists

def plot_violin_and_bar(experiment_id, model_sigmoid_params=None, llm_sigmoid_params = None):

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

    human_likert_weights = human_df.apply(get_likert_weights, axis=1)
    model_likert_weights = model_df.apply(get_likert_weights, axis=1)
    llm_likert_weights = llm_df.apply(get_likert_weights, axis=1)

    human_likert_by_image = average_sublists(human_likert_weights, 6)
    model_likert_by_image = average_sublists(model_likert_weights, 6)
    llm_likert_by_image = average_sublists(llm_likert_weights, 6)

    # Setting the style to 'ggplot'
    plt.style.use('ggplot')

    #Get colors
    palette = sns.color_palette("Set3", n_colors=8)

    # Creating a figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex = True)

    # Plotting the violin plots
    sns.violinplot(data=human_likert_by_image, ax=axs[0], palette=palette)
    #axs[0].set_title('Human Likert Scores by Image')
    #axs[0].set_xlabel('Image Index')
    axs[0].set_ylabel('Human Likert Score')

    sns.violinplot(data=model_likert_by_image, ax=axs[1], palette=palette)
    #axs[1].set_title('Model Likert Scores by Image')
    #axs[1].set_xlabel('Image Index')
    axs[1].set_ylabel('NaLaPIP Likert Score')

    sns.violinplot(data=llm_likert_by_image, ax=axs[2], palette=palette)
    #axs[2].set_title('LLM Likert Scores by Image')
    axs[2].set_xlabel('Image Index')
    axs[2].set_ylabel('GPT 4V Likert Score')


    # Adjusting the layout
    plt.tight_layout()

    plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "stacked_violin_plot.png"))

    os.makedirs(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "bar_plots"), exist_ok=True)
    for i in range(3):
        for model_name, data_to_plot in {
            "Human":human_likert_weights.tolist()[i+9], 
            "Model":model_likert_weights.tolist()[i+9], 
            "LLM":llm_likert_weights.tolist()[i+9],
        }.items():
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(8, 4))  # Professional size

            # Plotting the histogram
            #sns.histplot(data_to_plot, bins=15, kde=False, color=palette[3], edgecolor='black')
            plt.bar(np.arange(1,8,1), data_to_plot, color = palette[3])

            # Setting titles and labels
            ax.set_xlabel('Likert Score')
            ax.set_ylabel('Frequency')

            plt.savefig(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS, "bar_plots", f"bar_plot_for_{model_name}_task_{i}.png"))


np.random.seed(124)

experiment_id = "run-2023-12-17-1713"

os.makedirs(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_ANALYSIS), exist_ok=True)
webppl_slope, webppl_intercept, webppl_sigmoid_params = plot_human_versus_model_likert(experiment_id, DIR_WEBPPL)
llm_slope, llm_intercept, llm_sigmoid_params = plot_human_versus_model_likert(experiment_id, DIR_LLM)

print(webppl_sigmoid_params)
print(llm_sigmoid_params)

plot_wasserstein_distance(
    experiment_id, 
    model_sigmoid_params = webppl_sigmoid_params, 
    #llm_sigmoid_params = llm_sigmoid_params,
)
plot_violin_and_bar(
    experiment_id, 
    model_sigmoid_params =webppl_sigmoid_params, 
    #llm_sigmoid_params = llm_sigmoid_params,
)