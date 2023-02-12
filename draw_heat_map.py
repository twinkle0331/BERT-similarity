import matplotlib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import matplotlib.ticker as ticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    plt.rcParams['font.size'] = 12
    if not ax:
        ax = plt.gca()
    # create divider
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05, shrink=0.8)
    # Plot the heatmap
    im = ax.imshow(data, cmap='Blues', **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.1, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', 2)
    # cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    # formatter = ticker.FuncFormatter(lambda x, pos: '{:.1e}'.format(x))
    # cbar.ax.yaxis.set_major_formatter(formatter)
    # cbar.set_ticks([0,3e-3,6e-3])
    cbar.formatter.set_powerlimits((0, 0))

    # Show all ticks and label them with the respective list entries.
    if title=="attn_output" or title=="attn_prob":
        ax.set_xticks(np.arange(0, data.shape[1], 144))
        ax.set_yticks(np.arange(0, data.shape[0], 144))
    else:
        ax.set_xticks(np.arange(0,data.shape[1]+1,12))
        ax.set_yticks(np.arange(0,data.shape[0]+1,12))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(0,data.shape[1]+1,3)-.5, minor=True)
    ax.set_yticks(np.arange(0,data.shape[0]+1,3)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)

    return im, cbar

fig, (ax) = plt.subplots(1, 1, figsize=(6, 4.5)) # 12，9
input_dir = r'/home/ubuntu//trinkle/probing/output/caillen/similarity/mutual/mnli/mode_6/seed_14_6.npy'
seq = np.load(os.path.join(input_dir))
im, _ = heatmap(seq, None, None, ax=ax, vmin=np.min(seq), vmax=np.max(seq))

title = f"pre._att._mnli_INN"
# fig.suptitle(title)

plt.tight_layout()
output_dir = r'/home/ubuntu/trinkle/probing/output/caillen/similarity_figure'
file=f"{output_dir}/{title}_blues.pdf"
plt.savefig(file, figsize=(3, 2), dpi=900)
# print(f"successfully save {file}")

plt.show()

plt.close()