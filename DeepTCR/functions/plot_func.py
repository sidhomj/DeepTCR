import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logomaker


def get_max_val(matrices, masks):
    max_max_diff = []
    max_mean_diff = []
    for ii in range(len(matrices)):
        obs_val = matrices[ii] * masks[ii]
        obs_val = obs_val[np.nonzero(obs_val)]
        diff = matrices[ii] - obs_val[:, np.newaxis]
        max_max_diff.append(np.max(np.max(np.abs(diff), 1)))
        max_mean_diff.append(np.max(np.mean(np.abs(diff), 1)))
    # return np.max(max_max_diff),np.max(max_mean_diff)
    return max_max_diff,max_mean_diff

def sensitivity_logo(sequences,matrices,masks,ax=None,low_color='red',medium_color='white',high_color='blue',
                     font_name='Times New Roman',cmap=None,max_max_diff=None,max_mean_diff=None,
                     min_size=0.0,edgecolor='black',edgewidth=0.25,background_color='white'):

    sequences = np.flip(sequences,axis=0)
    matrices = np.flip(matrices,axis=0)
    masks = np.flip(masks,axis=0)
    max_mean_diff = np.flip(max_mean_diff,axis=0)
    max_max_diff = np.flip(max_max_diff,axis=0)

    if max_max_diff is None:
        max_max_diff,max_mean_diff = get_max_val(matrices,masks)

    cvals = [-1,0,1]
    colors = [low_color,medium_color,high_color]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    num_seq = len(matrices)
    max_len = np.max([len(x) for x in sequences])
    if ax is None:
        fig, ax = plt.subplots(figsize=(max_len, num_seq * 0.5))

    ax.set_xlim([0, max_len])
    ax.set_ylim([0, num_seq])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(background_color)
    xticks = np.arange(0.5, max_len + .5)
    dir_list = []
    mag_list = []
    for ii, sequence in enumerate(sequences, 0):
        obs_val = matrices[ii] * masks[ii]
        obs_val = obs_val[np.nonzero(obs_val)]
        diff = matrices[ii] - obs_val[:, np.newaxis]
        dir = np.mean(diff, 1)/max_mean_diff[ii]
        mag = np.max(np.abs(diff), 1)/max_max_diff[ii]
        dir_list.append(dir)
        mag_list.append(mag)
        for jj, (m,d, c) in enumerate(zip(mag,dir, sequence), 0):
            color = cmap(norm(d))[0:3]
            if m < min_size:
                m = min_size
            ceiling = ii + m
            logomaker.Glyph(p=xticks[jj], c=c, floor=ii, ceiling=ceiling, ax=ax, color=color, vpad=0.1,
                            font_name=font_name,edgecolor=edgecolor,edgewidth=edgewidth)

    return np.flip(dir_list), np.flip(mag_list)