
import numpy as np
import scipy.misc as misc


def angle_between_2d_points(pt1, pt2):
    rise = pt2[1] - pt1[1]
    run = pt2[0] - pt1[0]
    return np.arctan2(rise, run)


def angle_distance(ang1, ang2):
    phi = (ang2 - ang1) % 360
    if phi > 180:
        dist = 360 - phi
    else:
        dist = phi
    return dist


def normalize(v):
    v /= np.linalg.norm(v)
    return v


def vector_angle(v1, v2):
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    return np.arccos(np.dot(v1, v2) / (v1_mag * v2_mag))

'''
Quaternion functions
'''

def q_angle(q1, q2):
    q_inv = q_conjugate(q1)
    res = q_mult(q2, q_inv)
    return np.arccos(res[0]) * 2.0


def q_mult(q1, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)
    w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    y = q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3]
    z = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return np.hstack((w, x, y, z))


def q_conjugate(q):
    q = normalize(q)
    q[1:] *= -1
    return q


def qv_mult(q1, v1):
    q2 = np.hstack((0.0, v1))
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def axis_angle_to_q(v, theta):
    v = normalize(v)
    q = np.zeros(4)
    theta /= 2
    q[0] = np.cos(theta)
    q[1] = v[0] * np.sin(theta)
    q[2] = v[1] * np.sin(theta)
    q[3] = v[2] * np.sin(theta)
    q = normalize(q)
    return q


def q_to_axis_angle(q):
    theta = np.acos(q[0]) * 2.0
    return normalize(q[1:]), theta


def q_from_vectors2(v1, v2):
    angle = vector_angle(v1, v2)
    cp = np.cross(v1, v2)
    return axis_angle_to_q(cp, angle)


def q_from_vectors(v1, v2):
    v1norm = np.linalg.norm(v1)
    if v1norm != 0:
        v1n = v1 / v1norm
    else:
        v1n = v1.copy()
    v2norm = np.linalg.norm(v2)
    if v2norm != 0:
        v2n = v2 / v2norm
    else:
        v2n = v2.copy()
    q = np.zeros(4)
    vdot = np.dot(v1n, v2n)
    # Check for parallel vectors
    if vdot >= 1.0:
        q[0] = 1.0
        return normalize(q)
    if vdot <= -1.0:
        q[1:] = -v1
        return normalize(q)
    cp = np.cross(v1n, v2n)
    q[1:] = cp
    q[0] = vdot
    return normalize(q)


def q_from_mat(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    from Gohlke's transformations.py
    :param matrix: 3x3 rotation matrix
    :param isprecise:
    :return:
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def q_from_hpr(heading, pitch, roll):
    """
    Converts heading pitch and roll to a quaternion
    :param heading: The heading in radians
    :param pitch: The pitch in radians
    :param roll: The roll in radians
    :return: The output quaternion
    """
    pitch_quat = axis_angle_to_q(np.array([0.0, 1.0, 0.0]), pitch)
    print('pitch quat: ', pitch_quat)
    roll_quat = axis_angle_to_q(np.array([1.0, 0.0, 0.0]), roll)
    print('roll quat: ', roll_quat)
    result = q_mult(pitch_quat, roll_quat)
    print('result: ', result)
    head_quat = axis_angle_to_q(np.array([0.0, 0.0, 1.0]), -heading)
    print('head quat: ', head_quat)
    hpr_quat = q_mult(head_quat, result)
    return hpr_quat


def q_from_rpy(roll, pitch, yaw):
    """
    Converts roll, pitch, yaw to a quaternion
    :param roll: The roll in radians
    :param pitch: The pitch in radians
    :param yaw: The yaw in radians
    :return: The output quaternions
    """
    t0 = np.cos(yaw * 0.5)
    t1 = np.sin(yaw * 0.5)
    t2 = np.cos(roll * 0.5)
    t3 = np.sin(roll * 0.5)
    t4 = np.cos(pitch * 0.5)
    t5 = np.sin(pitch * 0.5)

    q = np.zeros((len(roll), 4))
    q[:, 0] = t0 * t2 * t4 + t1 * t3 * t5
    q[:, 1] = t0 * t3 * t4 - t1 * t2 * t5
    q[:, 2] = t0 * t2 * t5 + t1 * t3 * t4
    q[:, 3] = t1 * t2 * t4 - t0 * t3 * t5
    return q


def mat_from_translation_quaternion_rotation_scale(translation, rotation,
                                                   scale):
    scale_x = scale[0]
    scale_y = scale[1]
    scale_z = scale[2]

    x2 = rotation[1] * rotation[1]
    xy = rotation[1] * rotation[2]
    xz = rotation[1] * rotation[3]
    xw = rotation[1] * rotation[0]
    y2 = rotation[2] * rotation[2]
    yz = rotation[2] * rotation[3]
    yw = rotation[2] * rotation[0]
    z2 = rotation[3] * rotation[3]
    zw = rotation[3] * rotation[0]
    w2 = rotation[0] * rotation[0]

    m00 = x2 - y2 - z2 + w2
    m01 = 2.0 * (xy - zw)
    m02 = 2.0 * (xz + yw)

    m10 = 2.0 * (xy + zw)
    m11 = -x2 + y2 - z2 + w2
    m12 = 2.0 * (yz - xw)

    m20 = 2.0 * (xz - yw)
    m21 = 2.0 * (yz + xw)
    m22 = -x2 - y2 + z2 + w2

    result = np.zeros(16)
    result[0] = m00 * scale_x
    result[1] = m10 * scale_x
    result[2] = m20 * scale_x
    result[3] = 0.0
    result[4] = m01 * scale_y
    result[5] = m11 * scale_y
    result[6] = m21 * scale_y
    result[7] = 0.0
    result[8] = m02 * scale_z
    result[9] = m12 * scale_z
    result[10] = m22 * scale_z
    result[11] = 0.0
    result[12] = translation[0]
    result[13] = translation[1]
    result[14] = translation[2]
    result[15] = 1.0
    result = result.reshape((4, 4))
    return result


def rpy_to_rotation_matrix(roll, pitch, yaw):
    row1 = [np.cos(roll) * np.cos(pitch),
            np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) *
            np.cos(yaw),
            np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) *
            np.sin(yaw)]

    row2 = [np.sin(roll) * np.cos(pitch),
            np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) *
            np.cos(yaw),
            np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) *
            np.sin(yaw)]

    row3 = [-np.sin(pitch), np.cos(pitch) * np.sin(yaw), np.cos(pitch) *
            np.cos(yaw)]

    return np.array([row1, row2, row3])

'''
Cartesian and Spherical Transformations
'''


def sph2cart(azimuths, elevations):
    carts = np.zeros((len(azimuths), 3))
    carts[:, 0] = np.sin(elevations) * np.cos(azimuths)
    carts[:, 1] = np.sin(elevations) * np.sin(azimuths)
    carts[:, 2] = np.cos(elevations)
    return carts


def sph2cart(azimuths, elevations):
    carts = np.zeros((len(azimuths), 3))
    carts[:, 0] = np.sin(elevations) * np.cos(azimuths)
    carts[:, 1] = np.sin(elevations) * np.sin(azimuths)
    carts[:, 2] = np.cos(elevations)
    return carts

'''
Rounding operations
'''


def ceil_nearest(x, nearest):
    """
    Ceils up to the nearest value supplied
    :param x: The value to round
    :param nearest: The nearest value to round to
    :return: The rounded value
    """
    return np.ceil(x / float(nearest)) * nearest


def floor_nearest(x, nearest):
    """
    Floors up to the nearest value supplied
    :param x: The value to round
    :param nearest: The nearest value to round to
    :return: The rounded value
    """
    return np.floor(x / float(nearest)) * nearest


def round_nearest(x, nearest):
    """
    Rounds up to the nearest value supplied
    :param x: The value to round
    :param nearest: The nearest value to round to
    :return: The rounded value
    """
    return np.round(x / float(nearest)) * nearest


'''
Smoothing operations
'''


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window,
                                                             half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * misc.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def find_nearest_to_value(input_array, value, return_index=True):
    idx = (np.abs(input_array - value)).argmin()
    if return_index:
        return input_array[idx], idx
    return input_array[idx]


def find_nearest_datetime(dts, dt, return_index=True):
    diffs = np.array([(d - dt).total_seconds() for d in dts])
    diffs = np.abs(diffs)
    idx = np.argmin(diffs)
    if return_index:
        return dts[idx], idx
    return dts[idx]


def clip(val, min_val, max_val):
    return np.minimum(np.maximum(val, min_val), max_val)
