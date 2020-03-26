import csv
import os


def LOG2CSV(data, csv_file, flag = 'a'):
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()