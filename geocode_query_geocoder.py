import geocoder
import csv
import io
import time
from helper import read_distance_csv


def query_geocode(writer, address):
    g = geocoder.osm(address)

    if g.status == 'OK':
        writer.writerow([
            address,
            g.latlng[0],
            g.latlng[1]])

        return True
    return False


def run():
    data = read_distance_csv('data/neuenheim.csv')

    start = time.time()
    filename = 'data/geocoding.csv'

    with io.open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        row_count = sum(1 for row in reader)

    with io.open(filename, mode='a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for address in data['Address'][1154:]:
            success = query_geocode(writer, address)

            if not success:
                address = address.split()
                address.pop(-3)

                address = ' '.join(address)
                query_geocode(writer, address)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    run()
