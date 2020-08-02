# -*- coding: utf-8 -*-

import googlemaps
import csv
import io
import numpy as np
from helper import read_distance_csv, read_api_key


def query_geocode(writer, error_writer, address):
    gmaps = googlemaps.Client(key=read_api_key())
    g = gmaps.geocode(address=address)

    if len(g) != 1:
        error_writer.writerow([address])

    for result in g:
        writer.writerow([
            address,
            result['geometry']['location']['lat'],
            result['geometry']['location']['lng']
        ])


def run():
    data = read_distance_csv('data/neuenheim.csv')

    filename = 'data/geocoding'

    with io.open(filename + '.csv', mode='a', encoding='utf-8', newline='') as csvfile:
        with io.open(filename + '_error.csv', mode='a', encoding='utf-8', newline='') as errorfile:
            error_writer = csv.writer(errorfile, delimiter=';')
            writer = csv.writer(csvfile, delimiter=';')
            data = np.unique([x.target for x in data])
            for i, address in enumerate(data):
                query_geocode(writer, error_writer, address)

                if i % 100 == 0:
                    print(i)


if __name__ == '__main__':
    run()
