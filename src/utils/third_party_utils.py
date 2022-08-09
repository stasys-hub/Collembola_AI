"""
Project title:       CollembolAI
Authors:             Stephan Weißbach, Stanislav Sys, Clément Schneider
Original repository: https://github.com/stasys-hub/Collembola_AI.git
Module title:        third_party_utils
Purpose:             Gathering here functions that were forked from openly accessible solutions,
                     in order to emphasize on the credits of original authors and sources
LICENCE:             Our project is licensed under BSD (see LICENCE file). For each third party function here
                     we reproduce the original authors notes and disclamers, and added any sources we deemed 
                     useful to clarification. Some of the functions may be licensed under different open-source 
                     licence. In such case we ensured compatibility by abiding to the most restrictive requirements.
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap="Blues", write=None, show=False
):
    """
    Purpose:   given a sklearn confusion matrix (cm), make a nice plot
    Authors:   Originally from Scikit-learn, adapted by George Fisher, minor modification
               by C. Schneider (adding the image writing lines).
    Sources:   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
               https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    Licence:   The code of this function is adapted from George Fisher's apache 2.0 licenced work. Please preserve
               the content of this docstring to this point.

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    write:        File path: if not None, write the plot to the file path. Otherwise don't write.

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
                          write        = None
    """

    # calculate metrics
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    # set colormap
    cmap = plt.get_cmap(cmap)
    # initalize plot
    fig, axes = plt.subplots(figsize=(14, 14))
    # plot
    im = axes.imshow(cm, interpolation="nearest", cmap=cmap)
    axes.set_title(title, fontsize=20)
    plt.colorbar(im, ax=axes)
    # change x & y tick labels
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        axes.set_xticks(tick_marks, target_names)
        axes.set_yticks(tick_marks, target_names)
        plt.setp(axes.get_xticklabels(), rotation=90)


    # apply colormap & text labels
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    # axes labels
    axes.set_ylabel("True label", fontsize=16)
    axes.set_xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass
        ),
        fontsize=16,
    )
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(16)
    if write:
        plt.savefig(write)
        plt.savefig(write.replace(".png", ".svg"), format="svg")
    if show:
        plt.show()
