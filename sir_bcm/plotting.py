import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np

def visualize_mean_and_std(activity_matrix):
    mean_opinions = activity_matrix.mean(axis=1)
    std_opinions = activity_matrix.std(axis=1)

    plt.figure(figsize=(6, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_opinions, label="Mean Opinion")
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Opinion')
    plt.title('Evolution of Mean Opinion')
    plt.xlim(0, len(mean_opinions))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(std_opinions, label="Standard Deviation", color='orange')
    plt.xlabel('Timesteps')
    plt.ylabel('Standard Deviation')
    plt.title('Evolution of Opinion Standard Deviation')
    plt.xlim(0, len(std_opinions))
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_sample_opinions(activity_matrix, influences, sample_size, sample_type='random'):
    
    change = activity_matrix[-1] - activity_matrix[0]
    
    if sample_type == 'variance_top':
        sample_indices = np.argsort(-np.var(activity_matrix, axis=0))[:sample_size]
        title = 'Opinion Evolution of Users with Top Variance'
    elif sample_type == 'variance_bottom':
        sample_indices = np.argsort(np.var(activity_matrix, axis=0))[:sample_size]
        title = 'Opinion Evolution of Users with Bottom Variance'
    elif sample_type == 'initial_opinion_top':
        sample_indices = np.argsort(-activity_matrix[0])[:sample_size]
        title = 'Opinion Evolution of Users with Top Initial Opinions'
    elif sample_type == 'initial_opinion_bottom':
        sample_indices = np.argsort(activity_matrix[0])[:sample_size]
        title = 'Opinion Evolution of Users with Bottom Initial Opinions'
    elif sample_type == 'influence_top':
        sample_indices = np.argsort(-influences)[:sample_size]
        title = 'Opinion Evolution of Users with Top Influence'
    elif sample_type == 'influence_bottom':
        sample_indices = np.argsort(influences)[:sample_size]
        title = 'Opinion Evolution of Users with Bottom Influence'
    elif sample_type == 'change_top':
        sample_indices = np.argsort(-change)[:sample_size]
        title = 'Opinion Evolution of Users with Top Opinion Change'
    elif sample_type == 'change_bottom':
        sample_indices = np.argsort(change)[:sample_size]
        title = 'Opinion Evolution of Users with Least Opinion Change'
    else:  # default to random
        sample_indices = np.random.choice(len(activity_matrix[0]), size=sample_size, replace=False)
        title = 'Opinion Evolution of Random Sample of Users'
    
    sample_opinions = activity_matrix[:, sample_indices]
    sample_influences = influences[sample_indices]
    sample_ids = np.arange(len(activity_matrix[0]))[sample_indices]
    
    colors = [f"rgba({255*c}, {255*(1-c)}, {255*0.5}, 0.5)" for c in sample_influences]

    traces = []
    for idx in range(sample_size):
        trace = go.Scatter(
            y=sample_opinions[:, idx],
            mode='lines',
            line=dict(color=colors[idx]),
            hoverinfo="text",
            text=f"User {sample_ids[idx]}: Influence: {sample_influences[idx]:.3f}",
            showlegend=False
        )
        traces.append(trace)

    return traces, title

def visualize_sample_opinions_all(activity_matrix, influences, sample_size):
    types = ['random', 'influence_top', 'change_top', 'influence_bottom']
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[t.replace("_", " ").title() for t in types])

    for i, t in enumerate(types):
        traces, title = visualize_sample_opinions(activity_matrix, influences, sample_size, sample_type=t)
        for trace in traces:
            fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)

    fig.update_layout(
        title='Opinion Evolutions: Comparison of Sampling Techniques',
        xaxis_title='Timesteps',
        yaxis_title='Opinion Value',
        width=9*144,  # Convert inches to points, 2x width
        height=6*144,  # 2x height
    )
    
    fig.show()

import matplotlib.pyplot as plt

def plot_clusters(df, epsilon, mu):
    """Plot a stacked bar chart of clusters for the dataframe from the parameter space exploration
    """

    df_selected = df[(df.epsilon == epsilon) & (df.mu == mu)]
    
    # Group 4 and above clusters into a '4+' category
    df_selected['n_clusters'] = df_selected['n_clusters'].apply(lambda x: '4+' if x >= 4 else x)
    
    # Count occurrences of each cluster number
    cluster_counts = df_selected.groupby(['n_users', 'n_clusters']).size().unstack().fillna(0)
    
    # Create custom color map
    unique_clusters = cluster_counts.columns.sort_values()
    base_red = [1, 0, 0]
    darkening_factor = 0.2

    def darken_red(factor):
        return (base_red[0] * (1 - factor), base_red[1], base_red[2])

    colors = []
    for cluster in unique_clusters:
        if cluster == 1:
            colors.append('tab:green')
        elif cluster == 2:
            colors.append('tab:red')
        else:
            # We assume here that clusters are sorted in increasing order
            # And that after 2, the next is either 3 or '4+' 
            darkening_amount = (int(cluster) - 2) * darkening_factor
            colors.append(darken_red(darkening_amount))
    
    ax = cluster_counts.plot(kind='bar', stacked=True, figsize=(10,6), color=colors)
    plt.title(f'Stacked Bar of Clusters for epsilon={epsilon}, mu={mu}')
    plt.xlabel('Number of Users')
    plt.ylabel('Count')
    plt.legend(title='Number of Clusters')
    plt.show()
