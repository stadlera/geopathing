from dataclasses import dataclass
import io
import csv

KEY_FILE = 'googleapi.key'


def read_api_key():
    with io.open(KEY_FILE, mode='r', encoding='utf-8') as file:
        return file.readline()


@dataclass
class DistanceData:
    origin: str
    target: str
    distance: float
    duration: float


def read_distance_csv(filename):
    with io.open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = list()
        for row in reader:
            data.append(DistanceData(row[0], row[1], row[2], row[3]))

        return data


@dataclass
class Coordinate:
    lat: float
    lon: float


def read_geocoding_csv(filename):
    with io.open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = dict()
        for row in reader:
            data[row[0]] = Coordinate(row[1], row[2])

        return data
