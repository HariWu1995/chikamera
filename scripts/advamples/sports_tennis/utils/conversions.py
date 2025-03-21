def convert_pixels_to_meters(distance_in_pixels, reference_in_meters, reference_in_pixels):
    return (distance_in_pixels * reference_in_meters) / reference_in_pixels


def convert_meters_to_pixels(distance_in_meters, reference_in_meters, reference_in_pixels):
    return (distance_in_meters * reference_in_pixels) / reference_in_meters

