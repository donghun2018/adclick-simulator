"""
Scripts to help generating result plots

Donghun Lee 2018
"""

import csv

from policy_loader import get_puids


class plot_helper:

    @staticmethod
    def make_aggregates_csv(cost, revenue, profit, ofname='helper_aggregate.csv'):
        """
        Prepare csv for easy chart generation in Excel

        :param cost:
        :param revenue:
        :param profit:
        :param ofname:
        :return:
        """
        puids = get_puids()
        headers = ['puid (policy name)', 'cost', 'revenue', 'profit']

        with open(ofname, 'w', newline='') as ofh:
            csv_file = csv.writer(ofh)
            csv_file.writerow(headers)
            for row in zip(puids, cost, revenue, profit):
                csv_file.writerow(row)


if __name__ == "__main__":

    # run01 result
    c = [13275.0, 53056.0, 99.0, 217650.0, 9384.0, 0.0, 0.0, 142.0, 10080.0, 9108.0, 270.0, 100.0, 390.0, 108.0, 42084.0, 155756.0, 158112.0]
    r = [7188.168188662067, 28823.26882159405, 0.0, 119543.51063301227, 7959.356255280112, 0.0, 0.0, 61.062642855362085, 7678.9842860390945, 7670.966370645092, 196.4074310163872, 172.55449882325274, 555.5380702786829, 28.470578926117426, 31008.270907965107, 95739.28395800761, 98362.40919845125]
    p = [-6086.831811337933, -24232.73117840595, -99.0, -98106.48936698773, -1424.6437447198878, 0.0, 0.0, -80.93735714463791, -2401.0157139609055, -1437.033629354908, -73.59256898361281, 72.55449882325274, 165.53807027868288, -79.52942107388257, -11075.729092034893, -60016.71604199239, -59749.59080154875]
    plot_helper.make_aggregates_csv(c, r, p)