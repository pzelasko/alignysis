from functools import partial
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def to_matrix(
        df: pd.DataFrame,
        log: bool = True,
        norm: bool = False,
        occ_thresh: int = 0,
        ref_field: str = 'ref',
        hyp_field: str = 'hyp',
        label_sorting: Optional[List[str]] = None
):
    """
    Aggregate all confusions in the dataframe inside one confusion matrix.
    Can pass per-language or even per-experiment dataframe with confusions.
    Expected dataframe schema:

        ref	hyp	count	total_ref	lang	SYSTEM	AM	LM	TOKEN_TYPE	EXP
    0	w	w	3050	4131	Cantonese	e2e_phones	mono	None	per	e2e_phones_mono_None_per
    1	w	*	382	4131	Cantonese	e2e_phones	mono	None	per	e2e_phones_mono_None_per
    """
    labels = set(df[ref_field].unique()) | set(df[hyp_field].unique())
    if label_sorting is not None:
        int2sym = [l for l in label_sorting if l in labels] + list(labels - set(label_sorting))
    else:
        int2sym = sorted(labels)
    sym2int = {s: i for i, s in enumerate(int2sym)}
    mtx = np.zeros((len(int2sym), len(int2sym)))
    for idx, row in df.iterrows():
        ref_idx = sym2int[row[ref_field]]
        hyp_idx = sym2int[row[hyp_field]]
        mtx[ref_idx, hyp_idx] += row['count']
    if occ_thresh > 0:
        mtx = np.where(mtx >= occ_thresh, mtx, 0)
    if norm:
        den = np.sum(mtx, axis=1)
        mtx = (mtx.T / den).T
        mtx[np.isnan(mtx)] = 0
    if log:
        mtx = np.log10(mtx)
    fig = ConfusionMatrixDisplay(mtx, display_labels=np.array(list(sym2int)))
    return fig


def symmetrize_confusions(mtx: np.ndarray) -> np.ndarray:
    mtx = mtx.copy()
    for i in range(0, mtx.shape[0]):
        mtx[i, i] = 0
        for j in range(i + 1, mtx.shape[1]):
            tot = (mtx[i, j] + mtx[j, i]) / 2
            mtx[i, j] = mtx[j, i] = tot
    return mtx


def plot_similarity_dendrogram(similarities, labels):
    # Source: https://plotly.com/python/dendrogram/
    # To display this in jupyter lab:
    # https://plotly.com/python/getting-started/#jupyterlab-support

    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage

    data = similarities

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(data, orientation='bottom', labels=labels,
                               distfun=partial(pdist, metric='jensenshannon'),
                               linkagefun=partial(linkage,
                                                  optimal_ordering=True,
                                                  method='weighted',
                                                  metric='jensenshannon'
                                                  )
                               )
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(data, orientation='right',
                               distfun=partial(pdist, metric='jensenshannon'),
                               linkagefun=partial(linkage,
                                                  optimal_ordering=True,
                                                  method='weighted',
                                                  metric='jensenshannon'
                                                  )
                               )
    # dendro_side = ff.create_dendrogram(data, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data_ in dendro_side['data']:
        fig.add_trace(data_)

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    data_dist = pdist(data, metric='jensenshannon')
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]

    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Blues'
        )
    ]

    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data_ in heatmap:
        fig.add_trace(data_)

    fig.update_layout({'width': 1300, 'height': 1300,
                       'showlegend': False, 'hovermode': 'closest',
                       })
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'ticks': ""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, .85],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': True,
                             'ticks': ""
                             })
    # Edit yaxis2
    fig.update_layout(yaxis2={'domain': [.825, .975],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

    # Plot!
    fig.show()
