import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from src.learning.generate.generate_synthetic_graph_params import BIPARTITE_DEFAULT_PARAMS
from src.learning.generate.generate_synthetic_graph import GraphParams, generateSyntheticBipartiteGraph

def plot_extrema_matrices(result_df,test_param_1, test_param_2):
    def plot(x, y):
        params = GraphParams()
        setattr(params, test_param_1, x)
        setattr(params, test_param_2, y)
        _, _, interaction_matrix, _ = generateSyntheticBipartiteGraph(params, save_graph=False)
        plt.title(f"{test_param_1}: {x} {test_param_2}: {y}")
        plt.imshow(interaction_matrix.todense(), "Greys")
        plt.show()

    x_min = int(result_df[test_param_1].quantile(0.10))
    x_max = int(result_df[test_param_1].quantile(0.90))
    y_min = int(result_df[test_param_2].quantile(0.10))
    y_max = int(result_df[test_param_2].quantile(0.90))
    plot(x_min, y_min)
    plot(x_min, y_max)
    plot(x_max, y_min)
    plot(x_max, y_max)

def plot_surface(result_df,test_param_1, test_param_2, methods):
    for method in methods:
        try:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # Make data.
            X, Y, Z = get_XYZ_for_surface_plot(result_df, test_param_1, test_param_2, method)
            X_plot, Y_plot = np.meshgrid(X, Y)
            Z_plot = np.array(Z).reshape(len(X_plot[0]), len(X_plot)).T

            # Plot the surface.
            surf = ax.plot_surface(X_plot, Y_plot, Z_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

            # # Customize the z axis.
            # ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.set_xlabel(test_param_1)
            ax.set_ylabel(test_param_2)

            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"Method: {method}. {test_param_1} vs {test_param_2}")
            plt.show()
        except:

            pass


def get_XYZ_for_surface_plot(result_df, param1, param2, method=None):
    X = []
    Y = []
    Z = []
    max_len = 0
    def keep_default_other_than(df, param1, param2):
        flags = []
        for param, value in BIPARTITE_DEFAULT_PARAMS.items():
            if param == param1 or param == param2: continue
            flag = df[param] == value
            flags.append(flag)
        for flag in flags:
            flags[0] = flags[0] & flag
        return flags[0]

    result_df = result_df[keep_default_other_than(result_df, param1, param2)]
    grouped = result_df.groupby([param1, param2]).agg({method: "mean"}).reset_index()
    grouped = grouped.sort_values([param1, param2]).groupby([param1])

    for key, examples in grouped:
        examples = examples.drop_duplicates()
        if max_len == 0:
            max_len = len(examples)
            X.append(key)
            Y = examples[param2]
            Z.extend(list(examples[method]))
        elif len(examples)==max_len:
            X.append(key)
            Z.extend(list(examples[method]))
    return X, Y, Z