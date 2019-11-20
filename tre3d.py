import numpy as np

class params:
    def __init__(self):
        self.LENGTH = 168
        self.HEIGHT = 168
        self.WIDTH = 168

def rotation_by_any_axis_theta2(u_start, u_end, theta):
    """
        For some applications, it is helpful to be able to make a rotation with a given axis.
        Given a unit vector u = (a,b,c),
        the matrix for a rotation by an angle of theta about an axis in the direction of u.
    :param : u_start is (a1, b1, c1), u_end is (a2, b2, c2)
    :return: the matrix
    """
    a1, b1, c1 = u_start
    a2, b2, c2 = u_end
    a, b, c = a2 - a1, b2 - b1, c2 - c1

    # (1)  shift the vector u to the origin
    M1 = np.array([[1, 0, 0, -a1],
                   [0, 1, 0, -b1],
                   [0, 0, 1, -c1],
                   [0, 0, 0, 1]])

    M1_inv = np.array([[1, 0, 0, a1],
                   [0, 1, 0, b1],
                   [0, 0, 1, c1],
                   [0, 0, 0, 1]])
    # parallel to the x-axis
    if b == 0 and c == 0:
        Mx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), np.sin(theta), 0],
                   [0, -np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])
        return M1_inv.dot(Mx).dot(M1)

    # parallel to the y-axis
    elif a == 0 and c == 0:
        My = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                       [0, 1, 0, 0],
                       [np.sin(theta), 0, np.cos(theta), 0],
                       [0, 0, 0, 1]])
        return M1_inv.dot(My).dot(M1)

    # parallel to the z-axis
    elif b == 0 and a == 0:
        Mz = np.array([[np.cos(theta), np.sin(theta), 0, 0],
                       [-np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return M1_inv.dot(Mz).dot(M1)

    else:
        cos_alpha = c / np.sqrt(b * b + c * c)
        sin_alpha = b / np.sqrt(b * b + c * c)
        M2 = np.array([[1, 0, 0, 0],
                   [0, cos_alpha, sin_alpha, 0],
                   [0, -sin_alpha, cos_alpha, 0],
                   [0, 0, 0, 1]])
        M2_inv = np.array([[1, 0, 0, 0],
                       [0, cos_alpha, -sin_alpha, 0],
                       [0, sin_alpha, cos_alpha, 0],
                       [0, 0, 0, 1]])

        cos_beta = np.sqrt(b * b + c * c) / np.sqrt(a * a + b * b + c * c)
        sin_beta = -a / np.sqrt(a * a + b * b + c * c)
        M3 = np.array([[cos_beta, 0, -sin_beta, 0],
                       [0, 1, 0, 0],
                       [sin_beta, 0, cos_beta, 0],
                       [0, 0, 0, 1]])
        M3_inv = np.array([[cos_beta, 0, sin_beta, 0],
                       [0, 1, 0, 0],
                       [-sin_beta, 0, cos_beta, 0],
                       [0, 0, 0, 1]])

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        M4 = np.array([[cos_theta, sin_theta, 0, 0],
                       [-sin_theta, cos_theta, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        return M1.dot(M2).dot(M3).dot(M4).dot(M3_inv).dot(M2_inv).dot(M1_inv)

def rotation_by_any_axis_theta(u_start, u_end, theta):
    """
        For some applications, it is helpful to be able to make a rotation with a given axis.
        Given a unit vector u = (a,b,c),
        the matrix for a rotation by an angle of theta about an axis in the direction of u.
    :param : u_start is (a1, b1, c1), u_end is (a2, b2, c2)
    :return: the matrix
    """
    a1, b1, c1 = u_start
    a2, b2, c2 = u_end
    a, b, c = a2 - a1, b2 - b1, c2 - c1

    # (1)  shift the vector u to the origin
    M1 = np.array([[1, 0, 0, -a1],
                   [0, 1, 0, -b1],
                   [0, 0, 1, -c1],
                   [0, 0, 0, 1]])

    # (2) rotation alpha degrees around the X-axis
    if b == 0 and c == 0:
        cos_alpha = 1
        sin_alpha = 0
    else:
        cos_alpha = c / np.sqrt(b * b + c * c)
        sin_alpha = b / np.sqrt(b * b + c * c)
    M2 = np.array([[1, 0, 0, 0],
                   [0, cos_alpha, sin_alpha, 0],
                   [0, -sin_alpha, cos_alpha, 0],
                   [0, 0, 0, 1]])

    # (3) rotation beta degrees around the Y-axis, u-axis and z-axis is overlapped
    cos_beta = np.sqrt(b * b + c * c) / np.sqrt(a * a + b * b + c * c)
    sin_beta = a / np.sqrt(a * a + b * b + c * c)
    M3 = np.array([[cos_beta, 0, -sin_beta, 0],
                   [0, 1, 0, 0],
                   [sin_beta, 0, cos_beta, 0],
                   [0, 0, 0, 1]])

    # (4) rotation theta degrees around the Z-axis
    # theta = np.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    M4 = np.array([[cos_theta, sin_theta, 0, 0],
                   [-sin_theta, cos_theta, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    # (5) reverse rotation beta degrees around the Y-axis
    M5 = np.array([[cos_beta, 0, sin_beta, 0],
                   [0, 1, 0, 0],
                   [-sin_beta, 0, cos_beta, 0],
                   [0, 0, 0, 1]])

    # (6) reverse rotation alpha degrees around the X-axis
    M6 = np.array([[1, 0, 0, 0],
                   [0, cos_alpha, -sin_alpha, 0],
                   [0, sin_alpha, cos_alpha, 0],
                   [0, 0, 0, 1]])

    # (7) reverse shift the vector u to the origin
    M7 = np.array([[1, 0, 0, a1],
                   [0, 1, 0, b1],
                   [0, 0, 1, c1],
                   [0, 0, 0, 1]])

    M = M7.dot(M6).dot(M5).dot(M4).dot(M3).dot(M2).dot(M1)
    return M


def get_transform_matrix(paras, order=0):
    # order default 0:ground to moving  1:moving to fixed
    tx, ty, tz, theta_x, theta_y, theta_z, scale = paras
    theta_x, theta_y, theta_z = np.radians(theta_x), np.radians(theta_y), np.radians(theta_z)
    """
    # T is combining translation, rotation and scale
    # (x', y', z', 1).T = T * (x, y, z, 1).T
    # T: the order of operations
        (1) rotation about the  x axis
            R(theta_x) = [
                            [1, 0, 0, 0],
                            [0, cos(theta_x), -sin(theta_x), 0],
                            [0, sin(theta_x), cos(theta_x), 0],
                            [0 , 0, 0, 1]
            ]
        (2) rotation about the  y axis
            R(thata_y) = [
                            [cos(theta_y), 0, sin(theta_y), 0],
                            [0, 1, 0, 0],
                            [-sin(theta_y), 0, cos(theta_y), 0],
                            [0, 0, 0, 1]
            ]
        (3) rotation about the  z axis
            R(theta_z) = [
                            [cos(theta_z), -sin(theta_z), 0, 0],
                            [sin(theta_z), cos(theta_z), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
            ]
        (4) translate by tx, ty, tz
            R(tx,ty,tz) = [
                            [1, 0, 0, tx],
                            [0, 1, 0, ty],
                            [0, 0, 1, tz],
                            [0, 0, 0, 1 ]
            ]    
        # T = R(tx,ty,tz) x R(theta_z) x R(theta_y) x R(theta_x)
        # scale is equal 1.

        # T = np.array([[scale * np.cos(theta_z) * np.cos(theta_y),
    #                np.cos(theta_z) * np.sin(theta_y) * np.sin(theta_x) - np.sin(theta_z) * np.cos(theta_x),
    #                np.cos(theta_z) * np.sin(theta_y) * np.cos(theta_x) + np.sin(theta_z) * np.sin(theta_x), tx],
    #               [np.sin(theta_z) * np.cos(theta_y),
    #                scale * np.sin(theta_z) * np.sin(theta_y) * np.sin(theta_x) + np.cos(theta_z) * np.cos(theta_x),
    #                np.sin(theta_z) * np.sin(theta_y) * np.cos(theta_x) - np.cos(theta_z) * np.sin(theta_x), ty],
    #               [-np.sin(theta_y), np.cos(theta_y) * np.sin(theta_x), scale * np.cos(theta_y) * np.cos(theta_x), tz],
    #               [0, 0, 0, 1]])

    """
    # sin_x, cos_x = np.sin(theta_x), np.cos(theta_x)
    # sin_y, cos_y = np.sin(theta_y), np.cos(theta_y)
    # sin_z, cos_z = np.sin(theta_z), np.cos(theta_z)

    # M = np.zeros((4, 4))
    # M[0, 0] = scale * cos_z * cos_y
    # M[0, 1] = cos_z * sin_y * sin_x - sin_z * cos_x
    # M[0, 2] = cos_z * sin_y * cos_x + sin_z * sin_x
    # M[0, 3] = tx
    #
    # M[1, 0] = sin_z * cos_y
    # M[1, 1] = scale * sin_z * sin_y * sin_x + cos_z * cos_x
    # M[1, 2] = sin_z * sin_y * cos_x - cos_z * sin_x
    # M[1, 3] = ty
    #
    # M[2, 0] = -sin_y
    # M[2, 1] = cos_y * sin_x
    # M[2, 2] = scale * cos_y * cos_x
    # M[2, 3] = tz
    #
    # M[3, 3] = 1

    # rotation around the center of 3D image (LENGTH/2, WIDTH/2, HEIGHT/2)
    pa = params()
    u_end = np.array([pa.LENGTH // 2, pa.WIDTH // 2, pa.HEIGHT // 2])
    # step 1: rotation around (u_start(0, WIDTH/2, HEIGHT/2), u_end(LENGTH/2, WIDTH/2, HEIGHT/2)) x-axis
    # u_start = np.array([0, pa.WIDTH/2, pa.HEIGHT/2])
    Mx = rotation_by_any_axis_theta(np.array([0, pa.WIDTH // 2, pa.HEIGHT // 2]), u_end, theta_x)
    My = rotation_by_any_axis_theta(np.array([pa.LENGTH // 2, 0, pa.HEIGHT // 2]), u_end, theta_y)
    Mz = rotation_by_any_axis_theta(np.array([pa.LENGTH // 2, pa.WIDTH // 2, 0]), u_end, theta_z)

    Ms = np.array([[1 * scale, 0, 0, tx],
                   [0, 1 * scale, 0, ty],
                   [0, 0, 1 * scale, tz],
                   [0, 0, 0, 1]])

    if order==0:
        M = Ms.dot(Mz).dot(My).dot(Mx)

    else:
        M = Mx.dot(My).dot(Mz).dot(Ms)
    # M = Mz.dot(My).dot(Mx).dot(Ms)
    return M

def cal_tre_3d(im_shape, matrix_ground2moving, pre_matrix_moving2fixed, SIFT_POINTS_LIMIT_PER_IMAGE = 100):
    """
    :param im_shape: (length, width, high) // eg.64x64x64
    :param matrix_ground2moving: (shift_x, shift_y, shift_z, rotation_x, rotation_y, rotation_z, scale=1)
    :param pre_matrix_moving2fixed: (shift_x, shift_y, shift_z, rotation_x, rotation_y, rotation_z, scale=1)
    :param SIFT_POINTS_LIMIT_PER_IMAGE: the number of points
    :return: one pair of registration( TRE result)
    """
    # step 1: random choice some points in ground truth
    detect_points = np.random.randint(1, min(im_shape), size=[SIFT_POINTS_LIMIT_PER_IMAGE, 3])
    keypoints_in = []
    for point in detect_points:
        # print(point)
        keypoints_in.append(np.array([point[0], point[1], point[2], 1]))
    keypoints_in = np.array(keypoints_in)

    # step 2: points will transform from ground truth to moving by matrix_ground2moving
    # paras :  ground_im --> moving_im
    M = get_transform_matrix(matrix_ground2moving, 0)
    m_keypoints = np.dot(keypoints_in, M.T)

    # step 3: points transform from moving to fixed by pre_matrix_moving2fixed
    pre_M = get_transform_matrix(pre_matrix_moving2fixed, 1)
    # pre_M = np.linalg.inv(M)
    pt_points = np.dot(m_keypoints, pre_M.T)

    print('------------------------------------')
    print(M, '\n', np.linalg.inv(M), '\n', pre_M)
    print('------------------------------------')
    print(keypoints_in[:2], '\n', pt_points[:2])

    # step 4: calculate the distance of points between origin points and transformed points
    # eu_dist = np.sqrt(np.sum(np.square(keypoints_in - pt_points), axis=1))
    # tre_dist = np.mean(eu_dist)
    eu_dist = np.sqrt(np.sum(np.square(keypoints_in - pt_points), axis=1))
    # print(eu_dist)
    tre_dist = np.sqrt(np.mean(eu_dist))

    return tre_dist

def testinv():
    theta = np.radians(30)
    # x
    Mx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), np.sin(theta), 0],
                   [0, -np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])
    Mx_inv = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    # y
    My = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])
    My_inv = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])

    #z
    Mz = np.array([[np.cos(theta), np.sin(theta), 0, 0],
                   [-np.sin(theta), np.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    Mz_inv = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    Mz_ne = np.array([[np.cos(-theta), np.sin(-theta), 0, 0],
                   [-np.sin(-theta), np.cos(-theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    print(Mx.dot(Mx_inv))
    print(My.dot(My_inv))
    print(Mz.dot(Mz_inv))

    loc = np.array([2, 3, 4, 1])
    Mt = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [-2, -3, -4, 1]])
    print(loc.dot(Mt))

    print(Mz, '\n', Mz_ne)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    y1, y2, y3, y4, y5, y6 = [], [], [], [], [], []
    x = []

    pa = params()
    shape = (pa.LENGTH, pa.HEIGHT, pa.WIDTH)
    for i in range(1, 31):
        y1.append(cal_tre_3d(shape, [i, 0, 0, 0, 0, 0, 1], [-i, 0, 0, 0, 0, 0, 1]))
        y2.append(cal_tre_3d(shape, [i, i, 0, 0, 0, 0, 1], [-i, -i, 0, 0, 0, 0, 1]))
        y3.append(cal_tre_3d(shape, [i, i, i, 0, 0, 0, 1], [-i, -i, -i, 0, 0, 0, 1]))
        y4.append(cal_tre_3d(shape, [i, i, i, i, 0, 0, 1], [-i, -i, -i, -i, 0, 0, 1]))
        y5.append(cal_tre_3d(shape, [i, i, i, i, i, 0, 1], [-i, -i, -i, -i, -i, 0, 1]))
        y6.append(cal_tre_3d(shape, [i, i, i, i, i, i, 1], [-i, -i, -i, -i, -i, -i, 1]))
        x.append(i)
        # print(i, res)

    print(y6)
    plt.axis([0, 31, 0, 1])
    plt.plot(x, y1, color="c", linestyle="-", marker="^", linewidth=1, label='y1')
    plt.plot(x, y2, color="b", linestyle="-", marker="s", linewidth=1, label='y2')
    plt.plot(x, y3, color="y", linestyle="-", marker="*", linewidth=1, label='y3')
    plt.plot(x, y4, color="g", linestyle="-", marker="o", linewidth=1, label='y4')
    plt.plot(x, y5, color="r", linestyle="-", marker="d", linewidth=1, label='y5')
    plt.plot(x, y6, color="k", linestyle="-", marker="+", linewidth=1, label='y6')
    plt.legend(loc='upper left')
    plt.show()
    # testinv()