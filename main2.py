import cv2
from pupil_apriltags import Detector
import os
from pathlib import Path
import numpy as np
print(cv2.__version__)

imgpath=Path(r"C:\Users\wehao\Downloads\Python\Markers")

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,     # belangrijk
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25
)

sizeByShape = {
    "Square" : {12 : 0.036, 146 : 0.036, 173 : 0.036, 255 : 0.036, 206 : 0.036},
    "Dodecahedron" : {193 : 0.019, 556 : 0.019, 386 : 0.019, 462 : 0.019, 360 : 0.019, 340 : 0.019, 284 : 0.019, 70 : 0.019, 493 : 0.019, 300 : 0.019, 183 : 0.019},
    "Icosahedron" : {348 : 0.013, 495 : 0.013, 163 : 0.013, 392 : 0.013, 93 : 0.013, 123 : 0.013, 92 : 0.013, 290 : 0.013, 169 : 0.013, 12 : 0.013, 416 : 0.013, 394 : 0.013, 513 : 0.013, 203 : 0.013, 434 : 0.013, 380 : 0.013,
                     45 : 0.013, 33 : 0.013,576 : 0.013, 324 : 0.013, 494 : 0.01, 536 : 0.01, 321 : 0.01, 349 : 0.01, 586 : 0.01, 293 : 0.01, 582 : 0.01, 222 : 0.01, 18 : 0.01, 337 : 0.01, 339 : 0.01, 510 : 0.01}
}

dodecahedron_markers = {

    # upper markers
    462: {"t": ( 0.01368,  0.01883, -0.01165), "r_deg": ( 36.00,  68.79,  63.43)},
    556: {"t": (-0.01368,  0.01883, -0.01165), "r_deg": (144.00,  68.79,  63.40)},
    300: {"t": (-0.02213, -0.00719, -0.01165), "r_deg": (108.00,  97.94,  63.43)},
    70:  {"t": ( 0.00000, -0.02327, -0.01165), "r_deg": (  0.00, 116.57,  63.43)},
    340: {"t": ( 0.02213, -0.00719, -0.01165), "r_deg": (108.00,  97.94,  63.43)},

    # under markers
    360: {"t": ( 0.02213,  0.00719,  0.01162), "r_deg": (108.00,  97.94, 116.53)},
    386: {"t": ( 0.00000,  0.02327,  0.01162), "r_deg": (  0.00, 116.57, 116.57)},
    183: {"t": (-0.02213,  0.00719,  0.01162), "r_deg": ( 72.00,  97.94, 116.57)},
    493: {"t": (-0.01368, -0.01883,  0.01162), "r_deg": (144.00,  68.79, 116.57)},
    284: {"t": ( 0.01368, -0.01883,  0.01162), "r_deg": (144.00,  68.79, 116.57)},

    # top marker
    193: {"t": ( 0.00000,  0.00000, -0.02603), "r_deg": (108.00, 108.00,   0.00)},
}

square_markers = {
    146: {"t": ( -0.025,  0,  0), "r_deg": (90, 90, 90)},
    12: {"t": (  0, -0.025,  0), "r_deg": ( 0, 90, 90)},
    206: {"t": (0.025,  0,  0), "r_deg": (90, 90, 90)},
    255: {"t": (  0,0.025,  0), "r_deg": ( 0, 90, 90)},
    173: {"t": (  0,  0, -0.025), "r_deg": ( 0,  0,  0)},
}

icosahedron_markers = {

    # bottom row
    324: {"t": (-0.00402, -0.01981,  0.02646), "r_deg": (179.99,  48.19, 142.63)},
    380: {"t": ( 0.01760, -0.00994,  0.02646), "r_deg": (108.00,  82.68, 142.63)},
    45:  {"t": ( 0.01490,  0.01367,  0.02646), "r_deg": ( 36.00, 138.19, 142.62)},
    33:  {"t": (-0.00840,  0.01839,  0.02646), "r_deg": ( 36.00, 138.18, 142.61)},
    576: {"t": (-0.02009,  0.00230,  0.02646), "r_deg": (108.00,  82.68, 142.61)},

    # second row
    394: {"t": (-0.00650,  0.03206,  0.00624), "r_deg": (179.99,  90.00, 100.81)},
    337: {"t": ( 0.01270,  0.02781,  0.01529), "r_deg": (144.00,  79.19, 116.57)},
    290: {"t": ( 0.02848,  0.01609,  0.00624), "r_deg": (107.99,  97.31, 100.82)},
    222: {"t": ( 0.03038,  0.00348,  0.01529), "r_deg": ( 72.00, 107.67, 116.57)},
    392: {"t": ( 0.02410,  0.02212,  0.00624), "r_deg": ( 36.00, 109.49, 100.81)},
    586: {"t": ( 0.00607,  0.02997,  0.01529), "r_deg": (  0.01, 127.37, 116.55)},
    163: {"t": (-0.01359,  0.02976,  0.00624), "r_deg": ( 36.00, 109.46, 100.80)},
    321: {"t": ( 0.02662,  0.01504,  0.01529), "r_deg": ( 72.00, 107.66, 116.55)},
    203: {"t": (-0.03250, -0.00373,  0.00624), "r_deg": (108.00,  97.31, 100.80)},
    510: {"t": (-0.02253, -0.02067,  0.01529), "r_deg": (144.00,  79.19, 116.56)},

    # third row
    339: {"t": (-0.00608, -0.02996, -0.01529), "r_deg": (179.99, 127.38,  63.44)},
    169: {"t": ( 0.01359, -0.02975, -0.00625), "r_deg": (144.00, 109.47,  79.19)},
    18:  {"t": ( 0.02662, -0.01503, -0.01529), "r_deg": (107.99, 107.67,  63.44)},
    123: {"t": ( 0.03250,  0.00373, -0.00625), "r_deg": ( 71.99,  97.32,  79.19)},
    582: {"t": ( 0.02252,  0.02068, -0.01528), "r_deg": ( 36.01,  79.19,  63.44)},
    495: {"t": ( 0.00650,  0.03206, -0.00625), "r_deg": (  0.01,  89.99,  79.17)},
    349: {"t": (-0.01270,  0.02781, -0.01529), "r_deg": ( 36.00,  79.18,  63.42)},
    434: {"t": (-0.02848,  0.01610, -0.00625), "r_deg": ( 71.74,  97.27,  79.19)},
    536: {"t": (-0.03038, -0.00348, -0.01528), "r_deg": (108.03, 107.68,  63.44)},
    416: {"t": (-0.02410, -0.02212, -0.00625), "r_deg": (143.99, 109.47,  79.18)},

    # fourth row
    92:  {"t": (-0.00840,  0.01839, -0.02647), "r_deg": (143.99, 138.19,  37.38)},
    93:  {"t": ( 0.02008,  0.00231, -0.02646), "r_deg": ( 72.00,  82.68,  37.38)},
    348: {"t": ( 0.00401,  0.01982, -0.02646), "r_deg": (  0.01,  48.19,  37.38)},
    513: {"t": (-0.01761,  0.00995, -0.02645), "r_deg": ( 72.01,  82.69,  37.38)},
    12:  {"t": (-0.01490, -0.01367, -0.02646), "r_deg": (144.02, 138.20,  37.38)},

    # top
    293: {"t": (-0.00001,  0.00001, -0.03418), "r_deg": (0.00, 0.00, 0.00)},
}

def Rz(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], float)

def apply_marker_yaw_to_T_marker_obj(T_marker_obj, yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    T_corr = np.eye(4)
    T_corr[:3,:3] = Rz(yaw)

    return T_marker_obj @ T_corr

def pose_delta(T_a, T_b):
    # returns translation diff (m) and rotation diff (deg)
    dT = np.linalg.inv(T_a) @ T_b
    dt = np.linalg.norm(dT[:3,3])
    R = dT[:3,:3]
    ang = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    return dt, np.degrees(ang)


def best_yaw_for_marker(T_cam_obj_ref, T_cam_marker_meas_B, T_marker_obj_B):
    best = None
    for yaw_deg in (0, 90, 180, 270):
        T_marker_obj_try = apply_marker_yaw_to_T_marker_obj(T_marker_obj_B, yaw_deg)
        T_cam_marker_pred = T_cam_obj_ref @ np.linalg.inv(T_marker_obj_try)
        dt, dang = pose_delta(T_cam_marker_pred, T_cam_marker_meas_B)
        score = dang
        if best is None or score < best["score"]:
            best = {"yaw": yaw_deg, "dt": dt, "dang": dang, "score": score}
    return best

def choosePose(corners, imagePoints, cameraMatrix, distortionCoefficients):
    ok, rVecs, tVecs,_ = cv2.solvePnPGeneric(corners, imagePoints, cameraMatrix, distortionCoefficients,  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or len(rVecs) == 0:
        return None
    best = None
    error = 1e18

    for rVecs, tVecs in zip(rVecs, tVecs):
        if tVecs[2,0] <= 0:
            print("z negative -> corner order / pose ambiguity issue")
        projection, _ = cv2.projectPoints(corners, rVecs, tVecs, cameraMatrix, distortionCoefficients)
        projection = projection.reshape(-1,2)
        calculatedError = float(np.mean(np.linalg.norm(projection-imagePoints, axis=1))  )
        if calculatedError <= error:
            error = calculatedError
            best = (rVecs, tVecs)
    return best

def buildMarkers(markers):
    output = {}
    for mid, d in markers.items():
        
        T_marker_CoM = np.array(d["t"], float)
        T_CoM_marker= -T_marker_CoM
        R_obj_marker = facePlane(d["t"], outward=True, up = (0, 0, 1), eps = 1e-9)
        T_obj_marker = Hmatrix(T_CoM_marker, R_obj_marker)
        T_marker_obj = np.linalg.inv(T_obj_marker)

        output[mid]  = T_marker_obj
    return output

def Hmatrix(tXYZ,R):
    T=np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.array(tXYZ, dtype = float)

    return T

def facePlane(tXYZ, outward=True, up =(0, 0, 1), eps = 1e-9):
    t = np.array(tXYZ, dtype = float)
    up = np.array(up, dtype = float)
    up /= np.linalg.norm(up) + 1e-12
    z = -t
    z /= np.linalg.norm(z) + 1e-12
    if not outward:
        z = -z
    y = up - (up @ z) * z
    normalizedY = np.linalg.norm(y)

    if normalizedY < eps:
        alt = np.array([1,0,0], float)
        if abs(alt @ z) > 0.9:
            alt = np.array([0,1,0], float)
        y = alt - (alt @ z) * z
        y /= np.linalg.norm(y) + 1e-12
    else:
        y /= normalizedY

    x = np.cross(y,z)
    x /= np.linalg.norm(x) + 1e-12
    y=np.cross(z,x)
    y /= np.linalg.norm(y) + 1e-12

    return np.column_stack([x,y,z])

def getMarkerSize(markerId, shape):
    markerId = int(markerId)
    print(markerId)
    return sizeByShape.get(shape, {}).get(markerId, None)

def cornerStone(markerSize):
    h=markerSize/2
    return np.array([[-h, -h, 0.0],
                    [h, -h, 0.0],
                    [h, h, 0.0],
                    [-h, h, 0.0]])

def referencePicker(mids, T_cam_marker_meas, marker_obj_dict):
    bestA = None
    bestScore = float("inf")
    bestB = None
    for A in mids:
        T_cam_obj_ref = T_cam_marker_meas[A] @ marker_obj_dict[A]
        totalScore = 0.0
        perB = {}
        for B in mids:
            if B == A:
                continue
            res = best_yaw_for_marker(T_cam_obj_ref, T_cam_marker_meas[B], marker_obj_dict[B])
            score = res["dang"] + 50.0*res["dt"]
            totalScore += score
            perB[B] = res
        if totalScore < bestScore:
            bestScore = totalScore
            bestA = A
            bestB = perB
    return bestA, bestScore, bestB

def fuse_T(T_list):
    t = np.mean([t[:3,3] for t in T_list], axis = 0)
    Rsum = np.zeros((3,3), dtype = float)
    for T in T_list:

        Rsum += T[:3,:3]
    U,_,Vt = np.linalg.svd(Rsum)
    R =  U@Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    T = np.eye(4,4)
    T[:3,:3] = R
    T[:3,3] = t
    return T



def detection(path, shape, distance, tag=None, degrees=None):
    
    T_cam_marker_meas_right = {}
    T_cam_marker_meas_left = {}
    img=cv2.imread(path)
    h,w = img.shape[:2]
    left = img[:,:w//2]
    right = img[:,w//2:]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    left_g  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    dL_Raw=detector.detect(left_g)
    dR_Raw=detector.detect(right_g)
    used_clahe_L = False
    used_clahe_R = False

    dL= dL_Raw
    dR= dR_Raw  
    if len(dL_Raw) == 0:
        dL = detector.detect(clahe.apply(left_g))
        used_clahe_L = True
    if len(dR_Raw ) == 0:
        dR = detector.detect(clahe.apply(right_g))
        used_clahe_R = True
    
    if shape == "Square":
        marker_obj_dict = buildMarkers(square_markers)
    elif shape == "Dodecahedron":
        marker_obj_dict = buildMarkers(dodecahedron_markers)
    elif shape == "Icosahedron":
        marker_obj_dict = buildMarkers(icosahedron_markers)

    for c in dL:  
        tagId =  int(c.tag_id)
        leftSize = getMarkerSize(tagId, shape)
        if leftSize is None:
            print("not in working order")
            continue
        leftCornerPoints = cornerStone(float(leftSize))
        leftCornerPoints = leftCornerPoints.reshape(4, 3).astype(np.float32)
        imagePointLeft = c.corners.reshape(4,2).astype(np.float32)
        for i, (x,y) in enumerate(c.corners):
            cv2.putText(left, str(i), (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(left, f"id={str(tagId)}" , (int(c.center[0])+25, int(c.center[1])),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        rVec_markers_left, tVec_markers_left = choosePose(leftCornerPoints, imagePointLeft, cameraMatrixLeft, distortionCoefficientsLeft)

        testrVec_markers_left = rVec_markers_left.reshape(3,1)
        testtVec_markers_left = tVec_markers_left.reshape(3,1)
        R_cam_marker, _ = cv2.Rodrigues(testrVec_markers_left)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3,:3] = R_cam_marker
        T_cam_marker[:3,3]  = testtVec_markers_left[:,0]
        print(tagId)
        T_marker_obj = marker_obj_dict[tagId]
        T_cam_obj = T_cam_marker @ T_marker_obj 

        T = np.eye(4)
        T[:3,:3] = R_cam_marker
        T[:3,3]  = tVec_markers_left[:,0]
        T_cam_marker_meas_left[tagId] = T
        


        # ok, rVecLeft, tVecLeft = cv2.solvePnP(leftCornerPoints, imagePointLeft, cameraMatrixLeft, distortionCoefficientsLeft)
        cv2.circle(left, center=(int(c.center[0]), int(c.center[1])), radius=5, color=(0, 250,0))
        cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVec_markers_left, tVec_markers_left, 0.01)
    mids_left = list(T_cam_marker_meas_left.keys())
    if len(mids_left) < 2:
        if len(mids_left) == 0:
            print("Consistency check skipped: <2 markers detected in left frame.")
        else:
            T_cam_obj = T_cam_marker_meas_left[mids_left[0]] @ marker_obj_dict[mids_left[0]]
            
            R_obj = T_cam_obj[:3, :3]
            T_obj = T_cam_obj[:3, 3]
            rVec_obj_left,_  =cv2.Rodrigues(R_obj)
            tVec_obj_left = T_obj
            cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVec_obj_left, tVec_obj_left, 0.01)
    else:
        A, score, perB = referencePicker(mids_left, T_cam_marker_meas_left, marker_obj_dict)

        inliers = [A]
        outliers = []
        for B, res in perB.items():
            if res["dang"] > 12.0 or res["dt"] > 0.02:
                outliers.append(B)
            else:
                inliers.append(B)
        
        print("inliers:", inliers, "outliers:", outliers)

        T_list = [T_cam_marker_meas_left[inliers[mid]] @ marker_obj_dict[inliers[mid]] for mid in range(len(inliers))]
        T_cam_obj = fuse_T(T_list)

        R_obj= T_cam_obj[:3, :3]
        T_obj = T_cam_obj[:3, 3]
        rVec_obj_left,_  =cv2.Rodrigues(R_obj)
        tVec_obj_left = T_obj
        cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVec_obj_left, tVec_obj_left, 0.01)
        print("Detected IDs:", tagId if tagId is not None else None)

        for l in range(c.corners.shape[0]):
            cv2.circle(left, center=(int(c.corners[l][0]), int(c.corners[l][1])), radius=5, color=(0, 0,250))

    for d in dR:  
        tagId =  int(d.tag_id)
        rightSize = getMarkerSize(tagId, shape)
        if rightSize is None:
            print("not in working order")
            continue
        rightCornerPoints = cornerStone(float(rightSize))
        rightCornerPoints = rightCornerPoints.reshape(4, 3).astype(np.float32)
        imagePointRight = d.corners.reshape(4,2).astype(np.float32)
        for i, (x,y) in enumerate(d.corners):
            cv2.putText(right, str(i), (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.putText(right, f"id={str(tagId)}" , (int(d.center[0])+25, int(d.center[1])),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        rVec_markers_right, tVec_markers_right = choosePose(rightCornerPoints, imagePointRight, cameraMatrixRight, distortionCoefficientsRight)
        testrVec_markers_right = rVec_markers_right.reshape(3,1)
        testtVec_markers_right = tVec_markers_right.reshape(3,1)
        R_cam_marker, _ = cv2.Rodrigues(testrVec_markers_right)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3,:3] = R_cam_marker
        T_cam_marker[:3,3]  = testtVec_markers_right[:,0]
        print(tagId)
        T_marker_obj = marker_obj_dict[tagId]
        T_cam_obj = T_cam_marker @ T_marker_obj 

        T = np.eye(4)
        T[:3,:3] = R_cam_marker
        T[:3,3]  = tVec_markers_right[:,0]
        T_cam_marker_meas_right[tagId] = T
        


        cv2.circle(right, center=(int(d.center[0]), int(d.center[1])), radius=5, color=(0, 250,0))
        cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVec_markers_right, tVec_markers_right, 0.01)
    mids_right = list(T_cam_marker_meas_right.keys())
    if len(mids_right) < 2:
        if len(mids_right) == 0:
            print("Consistency check skipped: <2 markers detected in left frame.")
        else:
            T_cam_obj = T_cam_marker_meas_right[mids_right[0]] @ marker_obj_dict[mids_right[0]]
            R_obj = T_cam_obj[:3, :3]
            T_obj = T_cam_obj[:3, 3]
            rVec_obj_right,_  =cv2.Rodrigues(R_obj)
            tVec_obj_right = T_obj
            cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVec_obj_right, tVec_obj_right, 0.01)
    else:
        A, score, perB = referencePicker(mids_right, T_cam_marker_meas_right, marker_obj_dict)

        inliers = [A]
        outliers = []
        for B, res in perB.items():
            if res["dang"] > 12.0 or res["dt"] > 0.02:
                outliers.append(B)
            else:
                inliers.append(B)
        
        print("inliers:", inliers, "outliers:", outliers)

        T_list = [T_cam_marker_meas_right[inliers[mid]] @ marker_obj_dict[inliers[mid]] for mid in range(len(inliers))]
        T_cam_obj = fuse_T(T_list)

        R_obj= T_cam_obj[:3, :3]
        T_obj = T_cam_obj[:3, 3]
        rVec_obj_right,_  =cv2.Rodrigues(R_obj)
        tVec_obj_right = T_obj
        cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVec_obj_right, tVec_obj_right, 0.01)
        print("Detected IDs:", tagId if tagId is not None else None)

        for f in range(d.corners.shape[0]):
            cv2.circle(right, center=(int(d.corners[f][0]), int(d.corners[f][1])), radius=5, color=(0, 0,250))

    cv2.imshow('detecting markers left', left)
    cv2.imshow('detecting markers right', right)
    
    
    print(used_clahe_L, used_clahe_R)

    print(f"Currently inspecting:"
          f"{tag + ' ' if tag else ' '}"
            f"{shape + " at "}"
            f"{degrees + "°" + ' ' if degrees else ''}" 
            f"{distance}m")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


shape = ["Square", "Dodecahedron", "Icosahedron"]
distance = ["0.25", "0.5", "0.75", "1"]
degrees = ["10", "20", "30", "40", "45"]
tag = ["Aruco", "Apriltag"]
folder = ["First day", "Second day", "Double"]

fx1 = 772.2
fy1 = 772.345
cx1 = 617.27
cy1 = 374.896

fx2 = 771.1150
fy2 = 771.2800
cx2 = 647.9950
cy2 = 349.3545

k11 = -0.0293
k12 = 0.0063
p11 = 0.0000
p12 = 0.0000
k13 = 0.0114
k14 = 0.0000
k15 = -0.0000
k16 = 0.0000

k21 = -0.0297
k22 = 0.0088
p21 = 0.0000
p22 = 0.0000
k23 = 0.0050
k24 = 0.0000
k25 = -0.0000
k26 = 0.0000

distortionCoefficientsLeft = np.array([k11, k12, p11, p12, k13, k14, k15, k16])

distortionCoefficientsRight = np.array([k21, k22, p21, p22, k23, k24, k25, k26])

cameraMatrixLeft= np.array([[fx1, 0, cx1],
               [0, fy1, cy1],
               [0, 0, 1]])

cameraMatrixRight= np.array([[fx2, 0, cx2],
               [0, fy2, cy2],
               [0, 0, 1]])


choice = input("Please input, which data file you would wish to access from: First Day (1), Second Day (2), Double (3)")
shapechoice = input("Please input, which shape you would like to inspect from: Square (1), Dodecahedron (2), Truncated Icosahedron (3), or all shapes (4)")

if int(shapechoice) == 4:
    shape_used = shape
else:
    shape_used = [shape[int(shapechoice)-1]]
if int(choice) == 1:    
        newImgPath = imgpath / folder[int(choice) - 1]
        for x in range(len(shape_used)):
             for l in range(len(distance)):
                currentImgPath = newImgPath / (shape_used[x] + " " + distance[l] + "m")
                for f in currentImgPath.iterdir():
                    if f.is_file() and f.suffix.lower() == ".png":
                        detection(f, shape_used[x], distance[l],)
            
elif int(choice) == 2:
        distance = distance [0:2]
        newImgPath = imgpath / folder[int(choice) - 1]
        for x in range(len(shape_used)):
                print(shape_used)
                for l in range(len(distance)):
                    for d in range(len(degrees)):
                        currentImgPath = newImgPath / (tag[1] + " " +shape_used[x] + " " + degrees[d] + "deg " + distance[l] + "m")
                        for f in currentImgPath.iterdir():
                            if f.is_file() and f.suffix.lower() == ".png":
                                detection(f, shape_used[x], distance[l], tag[1], degrees[d])