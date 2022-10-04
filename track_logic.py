import re


def detect_car(labls_cords: dict) -> list:
    """
    labls_cords: dict with labels and coordinates

    return: list detected cars 
    """

    new_cars = []
    detected_cars = []

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

    for car in new_cars:

        # send car's number to the number model

        number = "АА422А63"

        if not re.match("[А-Я]{2}[0-9]{3}[А-Я]{1}[0-9]{2,3}", number) is None:

            car[0] = number

            # send car to the colour model
            colour = "green"
            car[1] = colour

            detected_cars.append(tuple(car))

    return detected_cars


def track_cars(pf_detected_cars: list, detected_cars: list) -> list:
    """
    pf_detected_cars - cars on the past frame
    detected_cars - cars that've been detected on the current frame
    return:list - values to write to DB
    """

    # campare cars in the past frame with
    # cars on the current frame

    for pf_det_car in pf_detected_cars:

        for det_car in detected_cars:

            # if cars types are the same
            if pf_det_car[2] == det_car[2]:

                # if cars colours are the same
                if pf_det_car[1] == det_car[1]:

                    # if numbers are closely equails
                    if (
                        len(set(pf_det_car[0]).symmetric_difference(set(det_car[0])))
                        <= 2
                    ):
                        # I think this is the same car
                        pf_detected_cars.remove(det_car)

    values = list(set(pf_detected_cars) - set(detected_cars))

    return values
