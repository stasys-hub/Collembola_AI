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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          write=None):
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
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if write:
        plt.savefig(write)
        plt.savefig(write.replace(".png", ".svg"), format="svg")
    plt.show()
