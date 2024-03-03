def __wrap_to360(angle):
    pos_in = angle > 0
    angle = angle % 360
    if angle == 0 and pos_in:
        angle = 360
    return angle


def __wrap_to_180(angle):
    q = (angle < -180) or (180 < angle)
    if q:
        angle = __wrap_to360(angle + 180) - 180
    return angle


def heading_to_standard(hdg):
    theta = __wrap_to360(90 - __wrap_to_180(hdg))
    return theta
