# -*- coding: utf-8 -*-

import googlemaps
import csv
import io
from address_crawler import get_address_set
from helper import read_api_key


def send_request(filename, departure_address, destination_address):
    gmaps = googlemaps.Client(key=read_api_key())
    distance = gmaps.distance_matrix(departure_address, destination_address)

    with io.open('data/' + filename + '.csv', mode='a', encoding='utf-8', newline='') as csvfile:
        with io.open('data/' + filename + '_error.csv', mode='a', encoding='utf-8', newline='') as errorfile:
            error_writer = csv.writer(errorfile, delimiter=';')
            writer = csv.writer(csvfile, delimiter=';')

            error_count = 0
            destination_addresses = distance['destination_addresses']
            elements = distance['rows'][0]['elements']
            for destination, result in zip(destination_addresses, elements):
                if result['status'] == 'OK':
                    writer.writerow([
                        distance['origin_addresses'][0],
                        destination,
                        result['distance']['value'],
                        result['duration']['value']])
                else:
                    error_writer.writerow([
                        distance['origin_addresses'][0],
                        destination,
                        destination_address
                    ])
                    error_count += 1
            print("Encountered %d errors." % (error_count))


def run():
    street_dict, _ = get_address_set(1)

    depature_addresses = {'altstadt': 'Untere Neckarstraße 50, 69117 Heidelberg',
                          'kirchheim': 'Pleikartsförster Straße 99, 69124 Heidelberg',
                          'pfaffengrund': 'Pfaffengrundstraße 1, 69123 Heidelberg',
                          'wieblingen': 'Mittelgewannweg 2, 69123 Heidelberg',
                          'rohrbach': 'Felix-Wankel-Straße 8, 69126 Heidelberg',
                          'ziegelhausen': 'Kleingemünder Straße 18, 69118 Heidelberg',
                          'bf': 'Baumschulenweg 10, 69124 Heidelberg'}

    batch_size = 100
    for section in depature_addresses:
        destination_address = ''
        batch_number = 0
        current_size = 0
        departure_address = depature_addresses[section]
        for street in street_dict:
            for house_number in street_dict[street]:
                if current_size >= batch_size:
                    send_request(section, departure_address, destination_address)
                    destination_address = ''
                    current_size = 0
                    batch_number += 1
                    print("Finished %d elements." % (batch_number * batch_size))

                if current_size != 0:
                    destination_address += "|"
                destination_address += street + " " + house_number + ", Heidelberg"

                current_size += 1

        send_request(section, departure_address, destination_address)


if __name__ == '__main__':
    run()
