import numpy as np
def Log(R):
    theta = 0.0 if R.trace() > 3.0 - 1e-6 else np.arccos(0.5 * (R.trace() - 1))
    K = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.abs(theta) < 0.001:
        return 0.5 * K
    else:
        return 0.5 * theta / np.sin(theta) * K

def Exp(ang):
    ang_norm = np.linalg.norm(ang)
    Eye3 = np.eye(3)
    if ang_norm > 0.0000001:
        r_axis = ang / ang_norm
        K = np.array([[0, -r_axis[2], r_axis[1]],
                      [r_axis[2], 0, -r_axis[0]],
                      [-r_axis[1], r_axis[0], 0]])
        return Eye3 + np.sin(ang_norm) * K + (1.0 - np.cos(ang_norm)) * np.dot(K, K)
    else:
        return Eye3
