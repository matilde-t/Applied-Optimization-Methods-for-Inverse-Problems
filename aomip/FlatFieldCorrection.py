def flatFieldCorrection(measurements, dark_frame, flat_field):
    """Flat field correction

    Correct the image using the formula 
    (measurements - dark_frame) / (flat_field - dark_frame)
    applied element-wise.
    """
    return (measurements - dark_frame) / (flat_field - dark_frame)