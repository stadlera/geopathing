import csv
import numpy as np
import io


def get_address_set(crop_factor=1):
    street_dict = dict()
    address_list = list()
    lat_list = list()
    lon_list = list()
    with io.open('data/heidelberg_address_list.csv', mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)

        for row in reader:
            street = row[1]
            house_number = row[2]
            lat = row[3]
            lon = row[4]

            if street == '' or house_number == '':
                continue

            if street in street_dict:
                street_dict[street] = np.append(street_dict[street], house_number)
            else:
                street_dict[street] = np.array([house_number])

            address_list.append(street + " " + house_number + ", Heidelberg")
            lat_list.append(lat)
            lon_list.append(lon)

    geocoding = {'Address': address_list,
                 'Latitude': lat_list,
                 'Longitude': lon_list}

    sorted_streets = sorted(street_dict.keys())
    for street in sorted_streets:
        a = np.sort(street_dict[street])
        street_dict[street] = a[::crop_factor]

    return street_dict, geocoding
