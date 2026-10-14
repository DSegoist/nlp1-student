import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools 
from collections import Counter, OrderedDict
import re 
from tabulate import tabulate
import random 
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from pathlib import Path

from . import loglin
from . import smlp


def download_ag_news_texts():
    """
    Returns the texts in the AG news collection (but not the labels).
    """
    #Loading our training corpus
    dataset = load_dataset("fancyzhx/ag_news") 
    corpus = dataset["train"]["text"]
    return corpus


def infer_num_classes(pmfs: dict) -> int:
    """
    Inspects the conditional pmfs, dicts mapping conditions to K-dimensional probability vectors, to infer K.
    """
    K = None
    for x, probs in pmfs.items():
        if K is None:
            K = len(probs)
        else:
            if K != len(probs):
                raise ValueError(f"Every pmf must be specified by a {K}-dimensional vector, got {len(probs)} for {x}")
    return K
    
def pmfs_to_str(pmfs: dict):
    """
    Return a table-formatted view of the given conditional pmfs in a single string ready to be displayed.
    """
    table = []
    K = None
    for x, probs in pmfs.items():
        table.append([x] + [p for p in probs])
        if K is None:
            K = len(probs)
        else:
            if K != len(probs):
                raise ValueError(f"Every pmf must be specified by a {K}-dimensional vector, got {len(probs)} for {x}")
        
    return tabulate(table, headers=['x'] + [f'p({k+1}|x)' for k in range(K)])

def encodings_to_str(encodings: dict, coords=None):
    """
    Return a table-formatted view of a dict of encodings in a single string ready to be displayed.
    coords: headers for the columns ('names' of each axis)
    """
    table = []
    num_dims = 0
    for text, h in encodings.items():
        table.append([text] + [hd for hd in h])
        num_dims = len(h)
    if coords is None:
        headers = ['x'] + [f"h_{d}(x)" for d in range(num_dims)]
    else:
        headers = ['x'] + [f"h_{d}(x): {name}" for d, name in enumerate(coords)]
    return tabulate(table, headers=headers)


def plot_pmf(outcomes, probs, hlabel="Outcome (y)", vlabel="Probability p(y)", title="Probability mass funciton", figsize=(3,2)):
    """
    Plot a pmf as a bar plot.
    outcomes: possible outcomes (xticks labels)
    probs: heights of the bars
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    _ = ax.bar(outcomes, probs)
    _ = ax.set_xlabel(hlabel, fontsize=10)
    _ = ax.set_ylabel(vlabel, fontsize=10)
    _ = ax.set_ylim(0, 1)
    _ = ax.set_title(title, fontsize=10)
    return fig, ax

def plot_pmfs_barh(pmfs, category_names, legendtitle="category", plottitle=None, fmt="{:.2f}"):
    """
    Plot a collection of conditional pmfs using barh plots.
    
    pfms : dict
        A mapping from conditions to a list of K probability masses.
    category_names : list of category names
    fmt: format for probability values
    """
    labels = list(pmfs.keys())
    data = np.array(list(pmfs.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['Blues'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    if plottitle is not None:
        ax.set_title(plottitle, fontsize=10)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color, fmt=fmt)
    ax.legend(
        title=legendtitle,
        ncols=len(category_names), 
        bbox_to_anchor=(1, -0.15),
        loc="upper right",
        fontsize="small",
        title_fontsize='small',
    )
    ax.figure.subplots_adjust(bottom=0.25)


    return fig, ax

class ThreeWaySentimentExample:
    """
    Our running example in the earlier sections of the tutorial.
    """

    category_names = ['negative', 'neutral', 'positive']

    pmfs = OrderedDict(
        [
            ("what a great day", [0.2, 0.1, 0.7]),
            ("a lovely day", [0.05, 0.1, 0.85]),
            ("a day that never ends", [1., 0., 0.])
        ]
    )

    @staticmethod
    def plot(stacked_bars=False):
        """Plot the example pfms"""
        pmfs = ThreeWaySentimentExample.pmfs
        category_names = ThreeWaySentimentExample.category_names
        
        if stacked_bars:
            return plot_pmfs_barh(pmfs, category_names)
        else:        
            fig, axs = plt.subplots(1, len(pmfs), figsize=(3*len(pmfs), 2.5))
            
            for i, (x, probs) in enumerate(pmfs.items()):
                _ = axs[i].bar(category_names, probs)
                _ = axs[i].set_xlabel("Sentiment (y)", fontsize=10)
                _ = axs[i].set_ylabel("p(y|x)", fontsize=10)
                _ = axs[i].set_ylim(0, 1)
                _ = axs[i].set_title(f"Text (x): {x}", fontsize=10)
            
            fig.tight_layout(w_pad=1, h_pad=1)
        return fig, axs


def cross_validation_split(pairs, k, seed=0):
    """
    Shuffles a list of labelled texts and splits the result into k folds.
    
    pairs: lists of tuples (sentence, label)
    k: number of folds
    seed: for reproducible shuffling
    """
    
    # Shuffle the pairs
    random.seed(seed)
    pairs = list(pairs)
    ids = list(range(len(pairs)))
    random.shuffle(ids)

    # Split without coercing (text, int) tuples into a string array
    ids_parts = np.array_split(np.array(ids, dtype=object), k)
    
    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]] 
        training_ids = list(itertools.chain.from_iterable(part for k, part in enumerate(ids_parts) if k != i))
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set, 
            "validation": validation_set, 
            'training_ids': training_ids,   # these are not often needed, but we return it just in case
            'validation_ids': ids_parts[i]  # these are not often needed, but we return it just in case
        }

    return cross_validation_sets

def cross_validation_experiment(data_splits, encoder: 'Encoder', model_option:str, save_to="tmp", loglin_args=dict(), smlp_args=dict()):
    """
    Trains and evaluates k models using a k-fold data split for cross validation. 
    This function returns a dictionary with 
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition
    
    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()

    
    
    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True) 
        
        if model_option == 'logistic':                        
            llmodel = loglin.fit(training_set, encoder, save_to=f'{fold_folder}/model', **loglin_args) 
            models[fold] = f'{fold_folder}/model.pickle' # loglin.fit adds the suffix .pickle
            
            predictions[fold] = loglin.predict([text for text, gold_label in validation_set], llmodel, encoder)    
            gold_labels_validation = [gold_label for text, gold_label in validation_set]
            accuracies[fold] = accuracy_score(gold_labels_validation, predictions[fold])
        elif model_option == 'mlp':            
            metrics, best_epoch, predictions[fold] = smlp.train_neural_network_model(
                training_set, validation_set, encoder, save_to=f'{fold_folder}/model', **smlp_args
            )
            models[fold] = f'{fold_folder}/model.pt' # smlp.train_neural_network_model adds the suffix .pt
            accuracies[fold] = metrics[best_epoch]['val_metrics']['accuracy']            
        else:
            raise ValueError(f"Unkonwn model_option={model_option}")

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}