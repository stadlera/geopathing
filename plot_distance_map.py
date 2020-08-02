# -*- coding: utf-8 -*-

import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from helper import read_distance_csv, read_geocoding_csv


def plot_contour_data(ax, data, levels, image_size, unit):
    n = 256
    color_pad = 20
    cmap = cm.get_cmap('RdYlGn_r', n)
    colors = cmap(range(n))
    colors = np.array([*colors[: n // 2 - color_pad], *colors[n // 2 + color_pad:]])

    cmap = ListedColormap(colors[np.round(np.linspace(0, n - 2 * color_pad - 1, len(levels))).astype(
        int)])  # cm.get_cmap('RdYlGn_r', len(levels))
    cs = ax.contour(data, levels,
                    extent=(0, image_size[0],
                            0, image_size[1]),
                    cmap=cmap,
                    linewidths=1)

    ax.clabel(cs, inline=1, fontsize=8, fmt='%1.0f')
    for collection, level in zip(cs.collections, levels):
        collection.set_label('{} {}'.format(level, unit))

    cmap = ListedColormap(colors)
    im = ax.imshow(data, cmap=cmap, alpha=0.3,
                   extent=(0, image_size[0],
                           image_size[1], 0),
                   vmin=min(levels), vmax=max(levels))

    cbar = plt.colorbar(im, cax=align_colorbar(ax))
    return cbar


def plot_image_data(ax, data, labels, image_size):
    cmap = cm.get_cmap('Set1', 7)
    cmap.colors[:, -1] = 0.3
    # change color for pfaffengrund
    cmap.colors[6, :-1] = [0.9, 0.9, 0.15]

    norm_bins = np.arange(0, 7) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = FuncFormatter(lambda x, pos: labels[norm(x)])

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im = ax.imshow(data,
                   extent=(0, image_size[0],
                           image_size[1], 0),
                   cmap=cmap,
                   norm=norm)

    cbar = plt.colorbar(im, format=fmt, ticks=tickz, cax=align_colorbar(ax))
    return cbar


def align_colorbar(ax, aspect=20, pad_fraction=0.5):
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_size.Fraction(pad_fraction, width)

    return divider.append_axes('right', size=width, pad=pad)


def plot_locations(ax, transform):
    locations = {'altstadt': [49.412333, 8.702226],
                 'neuenheim': [49.424115, 8.679423],
                 'kirchheim': [49.386453, 8.662292],
                 'rohrbach': [49.382914, 8.680439],
                 'wieblingen': [49.433261, 8.640435],
                 'ziegelhausen': [49.417705, 8.759337],
                 'pfaffengrund': [49.404745, 8.649053],
                 'bf': [49.397323, 8.670868]}

    for loc in locations:
        ax.plot(*transform(np.flip(locations[loc])), 'r.')


def plot_base_image(image_size, image_area):
    dpi = 300
    filename = 'data/heidelberg.pdf'
    image_pdf = convert_from_path(filename)[0]
    fig = plt.figure(figsize=(image_size[0] / dpi, image_size[1] / dpi), dpi=dpi)
    ax = plt.gca()
    ax.imshow(image_pdf.crop(image_area.flatten()), cmap='gray')
    plt.axis('off')
    return fig, ax


def run():
    sections = ['neuenheim', 'altstadt', 'kirchheim', 'rohrbach', 'wieblingen', 'ziegelhausen', 'pfaffengrund', 'bf']

    datastorage = dict()
    for section in sections:
        datastorage[section] = read_distance_csv('data/' + section + '.csv')

    all_durations = np.empty((len(datastorage[sections[0]]), len(sections)), dtype=float)
    all_distances = np.empty((len(datastorage[sections[0]]), len(sections)), dtype=float)
    for i, section in enumerate(datastorage):
        all_durations[:, i] = [x.duration for x in datastorage[section]]
        all_distances[:, i] = [x.distance for x in datastorage[section]]

    # correct wrong address for altstadt
    all_distances[:, 1] += 160
    all_durations[:, 1] += 15

    # coordinates
    geocoding = read_geocoding_csv('data/geocoding.csv')

    coordinates = [geocoding[d.target] for d in datastorage[sections[0]]]
    coordinates = np.array([[d.lon, d.lat] for d in coordinates], dtype=float)

    # bounding box coordinates of the pdf image
    bounding_box = np.array([[8.573164, 49.499765],
                             [8.794068, 49.311769]])

    n = 1024
    xx, yy = np.meshgrid(np.linspace(bounding_box[0, 0], bounding_box[1, 0], n),
                         np.linspace(bounding_box[0, 1], bounding_box[1, 1], n))

    # build kd-tree to find grid coordinates near data points and create a mask
    threshold = 0.005
    from scipy.spatial import cKDTree
    tree = cKDTree(coordinates)
    dists, indexes = tree.query(np.vstack((xx.flatten(), yy.flatten())).transpose())
    mask = dists.reshape(xx.shape) < threshold

    image_area = np.array([[42, 370],
                           [4630, 6380]])

    image_size = image_area[1, :] - image_area[0, :]
    transform = lambda x: (image_area[1, :] - image_area[0, :]) / \
                          (bounding_box[1, :] - bounding_box[0, :]) * (x - bounding_box[0, :])

    px_coords = np.array([transform(x) for x in coordinates], dtype=int)
    in_image = ((px_coords[:, 0] >= 0) & (px_coords[:, 1] >= 0)) | \
               ((px_coords[:, 0] < image_size[0]) & (px_coords[:, 1] < image_size[1]))

    # filter out-of-image points
    px_coords = px_coords[in_image, :]
    coordinates = coordinates[in_image]
    all_durations = all_durations[in_image, :]
    all_distances = all_distances[in_image, :]

    # interpolation and smoothing
    sigma = 1.5
    minimum_size = 3
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import minimum_filter

    interp_duration = np.zeros((n, n, all_durations.shape[1]))
    interp_distance = np.zeros((n, n, all_distances.shape[1]))
    for i in range(all_durations.shape[1]):
        interp = griddata(coordinates, all_durations[:, i], (xx, yy), method='nearest', fill_value=np.nan)
        interp = minimum_filter(interp, minimum_size)
        interp_duration[:, :, i] = gaussian_filter(interp, sigma)

        interp = griddata(coordinates, all_distances[:, i], (xx, yy), method='nearest', fill_value=np.nan)
        interp = minimum_filter(interp, minimum_size)
        interp_distance[:, :, i] = gaussian_filter(interp, sigma)

    # plotting
    labels = ['Wache Nord', 'Altstadt', 'Kirchheim', 'Rohrbach', 'Wieblingen', 'Ziegelhausen', 'Pfaffengrund']

    # plot the data points
    plot_base_image(image_size, image_area)
    plt.scatter(px_coords[:, 0], px_coords[:, 1])
    plt.savefig('figures/coords.png')

    # plot the distribution of fastest and shortest way
    sorted_durations = np.argsort(interp_duration[:, :, :-1], axis=2)
    sorted_durations = sorted_durations.astype(float)
    sorted_durations[np.invert(mask), :] = np.nan

    sorted_distances = np.argsort(interp_distance[:, :, :-1], axis=2)
    sorted_distances = sorted_distances.astype(float)
    sorted_distances[np.invert(mask), :] = np.nan
    for datastorage, filename in zip([sorted_distances, sorted_durations], ['fastest', 'shortest']):
        for i in range(7):
            fig, ax = plot_base_image(image_size, image_area)
            cbar = plot_image_data(ax, datastorage[:, :, i], labels, image_size)
            cbar.set_label('Verteilung der optimalen Ausrückegebiete')

            plot_locations(ax, transform)

            plt.savefig('figures/' + filename + '_{}.png'.format(i))
            plt.close(fig)

    # plot the duration and distance distribution for each section
    levels = [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    interp_duration /= 60
    interp_duration[np.invert(mask), :] = np.nan

    interp_distance /= 1000
    interp_distance[np.invert(mask), :] = np.nan

    filenames = ['duration', 'distance']
    labels = ['Zeit', 'Distanz']
    units = ['min', 'km']
    datasets = [interp_duration, interp_distance]

    for datastorage, level, filename, label, unit in zip(datasets, levels, filenames, labels, units):
        for i, section in enumerate(sections):
            fig, ax = plot_base_image(image_size, image_area)
            cbar = plot_contour_data(ax, datastorage[:, :, i], level, image_size, unit)
            cbar.set_label('{} [{}]'.format(label, unit))

            plot_locations(ax, transform)

            ax.legend()
            plt.savefig('figures/' + section + '_' + filename + '.png')
            plt.close(fig)


if __name__ == '__main__':
    run()
