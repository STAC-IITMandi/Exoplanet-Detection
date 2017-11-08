import csv
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def count_zero_means(data_mean):
    count = 0
    for mean in data_mean:
        if mean == 0.0:
            count += 1
    return count

def data_centering(data):
    data_mean = [np.mean(data_i) for data_i in data]
    count_zero_means(data_mean)
    per, i = 0, 0
    while per < 90:
        data = np.array([np.array([(x - mean) for x in data_i]) for data_i, mean in zip(data, data_mean)])
        data_mean = [np.mean(data_i) for data_i in data]
        c = count_zero_means(data_mean)
        per = c*100/len(data_mean)
        print(i, per)
        i += 1
    return data

def box_fitting_algorithm(data):
    data = data_centering(data)
    return data

def load_dataset_from_file(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    header = next(reader)
    data = []
    for line in reader:
        line = [float(flux) for flux in line]
        data.append(line)
    return header, np.array([np.array(data_i) for data_i in data])


if __name__ == "__main__":
    header, data = load_dataset_from_file('ExoTrain.csv')
    # data = box_fitting_algorithm(data)
    trace = go.Scatter(
        x = list(range(len(data[0]))),
        y = data[0],
        mode = 'markers'
    )
    d = [trace]
    py.plot(d, filename='basic-line.html')
