import re

import numpy


def check_numbers_overlaps(labls_cords: dict) -> list:

    """
    Check each number's BB and correlate it with car's BB

    return: list - the list has following structure [
        [(number's cords), (car's cords), 'car_type'],
        [(number's cords), (car's cords), 'car_type'],
        ...
        ]
    """

    new_cars = []

    for number in labls_cords["numbers"]:

        for car in labls_cords["cars"]:

            # check if number's bounding box fully overlaps car's
            if (car[0] <= number[0] <= number[2] <= car[2]) and (
                car[1] <= number[1] <= number[3] <= car[3]
            ):
                new_cars.append([number, car, "car"])

        for car in labls_cords["trucks"]:

            # check if number's bounding box fully overlaps car's
            if (car[0] <= number[0] <= number[2] <= car[2]) and (
                car[1] <= number[1] <= number[3] <= car[3]
            ):
                new_cars.append([number, car, "truck"])

        for car in labls_cords["busses"]:

            # check if number's bounding box fully overlaps car's
            if (car[0] <= number[0] <= number[2] <= car[2]) and (
                car[1] <= number[1] <= number[3] <= car[3]
            ):
                new_cars.append([number, car, "bus"])

    return new_cars

