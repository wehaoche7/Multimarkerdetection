import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import csv
print(cv2.__version__)

squareSize = 0.036
dodecahedronSize =0.019
icosahedronSizePentagon = 0.01
icosahedronSizeHexagon =  0.013
size = None

# Marker size per shape (note Icosahedron has two different marker sizes)
sizeByShape = {
    "Square" : {134 : 0.036, 101 : 0.036, 99 : 0.036, 103 : 0.036, 183 : 0.036},
    "Dodecahedron" : {147 : 0.019, 186 : 0.019, 184 : 0.019, 133 : 0.019, 166 : 0.019, 86 : 0.019, 57 : 0.019, 174 : 0.019, 104 : 0.019, 3 : 0.019, 5 : 0.019},
    "Icosahedron" : {44 : 0.013, 47 : 0.013, 61 : 0.013, 209 : 0.013, 187 : 0.013, 132 : 0.013, 246 : 0.013, 227 : 0.013, 28 : 0.013, 144 : 0.013, 111 : 0.013, 232 : 0.013, 60 : 0.013, 182 : 0.013, 135 : 0.013, 12 : 0.013,
                     77 : 0.013, 237 : 0.013, 31 : 0.013, 37 : 0.013, 17 : 0.01, 211 : 0.01, 224 : 0.01, 202 : 0.01, 30 : 0.01, 15 : 0.01, 85 : 0.01, 55 : 0.01, 147 : 0.01, 225 : 0.01, 108 : 0.01, 116 : 0.01}
}
# Archaic position data, is still required by icosahedron as i didn't manage to make the new data files for dodecahedrons yet

icosahedron_markers = {

    # bottom row
    135: {"t": (-0.00402, -0.01981,  0.02646), "r_deg": (179.99,  48.19, 142.63)},
    28: {"t": ( 0.01760, -0.00994,  0.02646), "r_deg": (108.00,  82.68, 142.63)},
    144:  {"t": ( 0.01490,  0.01367,  0.02646), "r_deg": ( 36.00, 138.19, 142.62)},
    111:  {"t": (-0.00840,  0.01839,  0.02646), "r_deg": ( 36.00, 138.18, 142.61)},
    227: {"t": (-0.02009,  -0.00230,  0.02646), "r_deg": (108.00,  82.68, 142.61)},

    # second row
    77: {"t": (-0.00650,  -0.03206,  0.00624), "r_deg": (179.99,  90.00, 100.81)},
    225: {"t": ( 0.01270,  0.02781,  0.01529), "r_deg": (144.00,  79.19, 116.57)},
    12: {"t": ( 0.02848,  -0.01609,  0.00624), "r_deg": (107.99,  97.31, 100.82)},
    202: {"t": ( 0.03038,  0.00348,  0.01529), "r_deg": ( 72.00, 107.67, 116.57)},
    132: {"t": ( 0.02410,  0.02212,  0.00624), "r_deg": ( 36.00, 109.49, 100.81)},
    116: {"t": ( 0.00607,  0.02997,  0.01529), "r_deg": (  0.01, 127.37, 116.55)},
    31: {"t": (-0.01359,  0.02976,  0.00624), "r_deg": ( 36.00, 109.46, 100.80)},
    108: {"t": ( -0.02662,  0.01504,  0.01529), "r_deg": ( 72.00, 107.66, 116.55)},
    232: {"t": (-0.03250, -0.00373,  0.00624), "r_deg": (108.00,  97.31, 100.80)},
    147: {"t": (-0.02253, -0.02067,  0.01529), "r_deg": (144.00,  79.19, 116.56)},

    # third row
    85: {"t": (-0.00608, -0.02996, -0.01529), "r_deg": (179.99, 127.38,  63.44)},
    237: {"t": ( 0.01359, -0.02975, -0.00625), "r_deg": (144.00, 109.47,  79.19)},
    15:  {"t": ( 0.02662, -0.01503, -0.01529), "r_deg": (107.99, 107.67,  63.44)},
    246: {"t": ( 0.03250,  0.00373, -0.00625), "r_deg": ( 71.99,  97.32,  79.19)},
    55: {"t": ( 0.02252,  0.02068, -0.01528), "r_deg": ( 36.01,  79.19,  63.44)},
    37: {"t": ( 0.00650,  0.03206, -0.00625), "r_deg": (  0.01,  89.99,  79.17)},
    17: {"t": (-0.01270,  0.02781, -0.01529), "r_deg": ( 36.00,  79.18,  63.42)},
    61: {"t": (-0.02848,  0.01610, -0.00625), "r_deg": ( 71.74,  97.27,  79.19)},
    30: {"t": (-0.03038, -0.00348, -0.01528), "r_deg": (108.03, 107.68,  63.44)},
    182: {"t": (-0.02410, -0.02212, -0.00625), "r_deg": (143.99, 109.47,  79.18)},

    # fourth row
    44:  {"t": (-0.00840,  0.01839, -0.02647), "r_deg": (143.99, 138.19,  37.38)},
    60:  {"t": ( 0.02008,  0.00231, -0.02646), "r_deg": ( 72.00,  82.68,  37.38)},
    47: {"t": ( 0.00401,  0.01982, -0.02646), "r_deg": (  0.01,  48.19,  37.38)},
    187: {"t": (-0.01761,  0.00995, -0.02645), "r_deg": ( 72.01,  82.69,  37.38)},
    209:  {"t": (-0.01490, -0.01367, -0.02646), "r_deg": (144.02, 138.20,  37.38)},

    # top
    211: {"t": (-0.00001,  0.00001, -0.03418), "r_deg": (0.00, 0.00, 0.00)},
}

# def referencePicker(mids, T_cam_marker_meas, marker_obj_dict):
#     bestA = None
#     bestScore = float("inf")
#     bestB = None
#     for A in mids:
#         T_cam_obj_ref = T_cam_marker_meas[A] @ marker_obj_dict[A]
#         totalScore = 0.0
#         perB = {}
#         for B in mids:
#             if B == A:
#                 continue
#             res = best_yaw_for_marker(T_cam_obj_ref, T_cam_marker_meas[B], marker_obj_dict[B])
#             score = res["dang"] + 50.0*res["dt"]
#             totalScore += score
#             perB[B] = res
#         if totalScore < bestScore:
#             bestScore = totalScore
#             bestA = A
#             bestB = perB
#     return bestA, bestScore, bestB


def Hmatrix(tXYZ,R):
    T=np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.array(tXYZ, dtype = float)

    return T

def euler_deg_to_RzRyRx(r_deg):
    rx, ry, rz = np.deg2rad(r_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,0,0],
                   [0,cx,-sx],
                   [0,sx,cx]], float)
    Ry = np.array([[cy,0,sy],
                   [0,1,0],
                   [-sy,0,cy]], float)
    Rz = np.array([[cz,-sz,0],
                   [sz, cz,0],
                   [0,  0,1]], float)

    return Rz @ Ry @ Rx



def load_marker_obj_dict(csv_path, obj_name="CoM"):
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["name"].strip()
            if name.lower() == obj_name.lower():
                continue

            try:
                mid = int(name)
            except:
                continue

            tx = float(row["tx"]); ty = float(row["ty"]); tz = float(row["tz"])
            R = np.array([
                [float(row["r11"]), float(row["r12"]), float(row["r13"])],
                [float(row["r21"]), float(row["r22"]), float(row["r23"])],
                [float(row["r31"]), float(row["r32"]), float(row["r33"])],
            ], dtype=float)

            


            R[np.abs(R) < EPS] = 0.0
            t = np.array([tx, ty, tz], float)
            t[np.abs(t) < EPS] = 0.0

            T_marker_obj = np.eye(4)
            # if mid == 5:
            #     R=euler_deg_to_RzRyRx([0,121.72,116.57])
            # if mid == 174:
            #     R=euler_deg_to_RzRyRx([0,121.72,116.57])    
            # if mid == 147:
            #     R=euler_deg_to_RzRyRx([144,162,63.43]).T
            # if mid == 133:
            #     R=euler_deg_to_RzRyRx([36,162,116.57]).T
            T_marker_obj[:3,:3] = R.T
            T_marker_obj[:3,3]  = t
            print(mid, T_marker_obj)
            out[mid] = T_marker_obj

    return out


def getMarkerSize(markerId, shape):
    markerId = int(markerId)
    return sizeByShape.get(shape, {}).get(markerId, None)

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

def Rz(theta):
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))
    k = np.array([[cos,-sin,0],
    [sin, cos,0],
    [0,   0,  1]])
    return k


def detection(path, shape, distance, tag=None, degrees=None):
    T_cam_marker_meas_right = {}
    T_cam_marker_meas_left = {}
    testing = {}
    mids = []
    midstesting = []
    img=cv2.imread(path)
    h,w = img.shape[:2]
    left = img[:,:w//2]
    right = img[:,w//2:]
    left_gray  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    corners1,ids1,rejected1=detector.detectMarkers(left_gray)
    corners2,ids2,rejected2=detector.detectMarkers(right_gray)
    print("Markers detected left:", ids1)
    print("Markers detected right:", ids2)

    if shape == "Square":
        marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\Cubegeometry.coord_systems_rel_CoM_semicolon.csv", obj_name="CoM")
    elif shape == "Dodecahedron":
        marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\Dodecacorrect.coord_systems_rel_test_fileCoM_semicolon.csv", obj_name="CoM")
    # elif shape == "Icosahedron":
    #     marker_obj_dict = buildMarkers(icosahedron_markers)
    
    if ids1 is not None:
        cv2.aruco.drawDetectedMarkers(left, corners1, ids1)
        for corner1, mid1 in zip(corners1, ids1.flatten()):
            size = getMarkerSize(mid1, shape)
            if size is None:
                print("BAD size:", size, "for id", int(mid1), "shape", shape)
                continue
            if not (size > 0):
                print("BAD size:", size, "for id", int(mid1), "shape", shape)
                continue

            rVecs_markers_left, tVecs_markers_left, _ = aruco.estimatePoseSingleMarkers(corner1, size, cameraMatrixLeft, distortionCoefficientsLeft)

            testrVec_markers_left = rVecs_markers_left.reshape(3,1)
            R_cam_marker, _ = cv2.Rodrigues(testrVec_markers_left)

            # the actual T matrices used for calculation
            T = np.eye(4)
            T[:3,:3] = R_cam_marker
            T[:3,3]  = tVecs_markers_left[:,0]
            T_cam_marker_meas_left[mid1] = T
            #T matrices used for validation and correction (these and subsequent related code can be ignored)
            T_test = np.eye(4)
            T_test[:3,:3] = marker_obj_dict[mid1][:3,:3].T
            T_test[:3,3] = marker_obj_dict[mid1][:3,3]
            
            testing[mid1] = T_test
            mids = list(T_cam_marker_meas_left.keys())
            
            midstesting = list(testing.keys())
            
            cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVecs_markers_left, tVecs_markers_left, 0.01)
        #The code currently focuses on single markers since multi marker detection is pointless if the individual markers aren't correct, also the right side of the camera is commented out
        # if len(mids) < 2: 
        print("Consistency check skipped: <2 markers detected in left frame.")
        T_correct = np.eye(4)
        T_correct[:3,:3] = Rz(72)
        for length in mids:
            
            T_cam_obj = T_cam_marker_meas_left[length] @ marker_obj_dict[length]
            T_cam_obj_correct = T_cam_marker_meas_left[length] @ marker_obj_dict[length]@T_correct
            print("T_cam_obj", T_cam_obj,"T_cam_obj_correct", T_cam_obj_correct)
            
            
        for longth in midstesting:
            print("Testing results assembled marker obj", testing[longth])
            T_cam_obj_C = T_cam_marker_meas_left[longth] @ testing[longth]
            print("T_can_obj_C", T_cam_obj_C)

            


        

        R_obj = T_cam_obj[:3, :3]
        T_obj = T_cam_obj[:3, 3]
        rVec_obj_left,_  =cv2.Rodrigues(R_obj)
        tVec_obj_left = T_obj
        cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVec_obj_left, tVec_obj_left, 0.01)
        # else:
        #     A, score, perB = referencePicker(mids, T_cam_marker_meas_left, marker_obj_dict)

        #     inliers = [A]
        #     outliers = []
        #     yaw_dict = {A:0}
        #     for B, res in perB.items():
        #         if res["dang"] > 12.0 or res["dt"] > 0.02:
        #             outliers.append(B)
        #         else:
        #             inliers.append(B)
        #             yaw_dict[B] = res["yaw"]
            
        #     print("left inliers:", inliers, "left outliers:", outliers)
        #     T_list = []
        #     for mid in inliers:
        #         yaw = yaw_dict.get(mid, 0)
        #         T_marker_obj_correct = apply_marker_yaw_to_T_marker_obj(marker_obj_dict[mid], yaw)
        #         T_list.append(T_cam_marker_meas_left[mid]@T_marker_obj_correct)
        #     T_cam_obj = fuse_T(T_list)
        # T_cam_obj = T_cam_marker_meas_left[133] @ marker_obj_dict[133]
        # R_obj= T_cam_obj[:3, :3]
        # T_obj = T_cam_obj[:3, 3]
        # rVec_obj_left,_  =cv2.Rodrigues(R_obj)
        # tVec_obj_left = T_obj
        # cv2.drawFrameAxes(left, cameraMatrixLeft, distortionCoefficientsLeft, rVec_obj_left, tVec_obj_left, 0.01)
        # print("Detected IDs:", ids1.flatten() if ids1 is not None else None)
    
        
    
    # if ids2 is not None:
    #     cv2.aruco.drawDetectedMarkers(right, corners2, ids2)
    #     for corner2, mid2 in zip(corners2, ids2.flatten()):
    #         size = getMarkerSize(mid2, shape)
    #         if size is None:
    #             continue
    #         if not (size > 0):
    #             print("BAD size:", size, "for id", int(mid2), "shape", shape)
    #             continue

    #         rVecs_markers_right, tVecs_markers_right, _ = aruco.estimatePoseSingleMarkers(corner2, size, cameraMatrixRight, distortionCoefficientsRight)
    #         testrVec_markers_right = rVecs_markers_right.reshape(3,1)
    #         testtVec_markers_right = tVecs_markers_right.reshape(3,1)
    #         R_cam_marker, _ = cv2.Rodrigues(testrVec_markers_right)
    #         T_cam_marker = np.eye(4)
    #         T_cam_marker[:3,:3] = R_cam_marker
    #         T_cam_marker[:3,3]  = testtVec_markers_right[:,0]
    #         T_marker_obj = marker_obj_dict[mid2]

    #         T = np.eye(4)
    #         T[:3,:3] = R_cam_marker
    #         T[:3,3]  = tVecs_markers_right[:,0]
    #         T_cam_marker_meas_right[mid2] = T
    #         mids = list(T_cam_marker_meas_right.keys())
    #         cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVecs_markers_right, tVecs_markers_right, 0.01)
    #     if len(mids) < 2:
    #         print("Consistency check skipped: <2 markers detected in right frame.")
    #         T_cam_obj = T_cam_marker_meas_right[mids[0]] @ marker_obj_dict[mids[0]]
            
    #         R_obj = T_cam_obj[:3, :3]
    #         T_obj = T_cam_obj[:3, 3]
    #         rVec_obj_right,_  =cv2.Rodrigues(R_obj)
    #         tVec_obj_right = T_obj

    #         cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVec_obj_right, tVec_obj_right, 0.01)
    #     else:
    #         A, score, perB = referencePicker(mids, T_cam_marker_meas_right, marker_obj_dict)

    #         inliers = [A]
    #         outliers = []
    #         yaw_dict = {A:0}
    #         for B, res in perB.items():
    #             if res["dang"] > 12.0 or res["dt"] > 0.02:
    #                 outliers.append(B)
    #             else:
    #                 inliers.append(B)
    #                 yaw_dict[B] = res["yaw"]
            
    #         print("right inliers:", inliers, "right outliers:", outliers)
    #         T_list = []
    #         for mid in inliers:
    #             yaw = yaw_dict.get(mid, 0)
    #             # T_marker_obj_correct = apply_marker_yaw_to_T_marker_obj(marker_obj_dict[mid], yaw)
    #             T_list.append(T_cam_marker_meas_right[mid]@marker_obj_dict)
    #         T_cam_obj = fuse_T(T_list)

    #         R_multipleMarker = T_cam_obj[:3, :3]
    #         T_multipleMarker = T_cam_obj[:3, 3]
    #         rVec_obj_right,_  =cv2.Rodrigues(R_multipleMarker)
    #         tVec_obj_right = T_multipleMarker
    #         cv2.drawFrameAxes(right, cameraMatrixRight, distortionCoefficientsRight, rVec_obj_right, tVec_obj_right, 0.01)



            
            

            # print("Detected IDs:", ids2.flatten() if ids2 is not None else None)

        

    


    print(f"Currently inspecting: "
          f"{tag + ' ' if tag else ' '}"
            f"{shape + " at "}"
            f"{degrees + "°" + ' ' if degrees else ' '}" 
            f"{distance}m")
    
    cv2.imshow('Detected Markers left', left)
    cv2.imshow('Detected Markers right', right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def pose_delta(T_a, T_b):
    # returns translation diff (m) and rotation diff (deg)
    dT = np.linalg.inv(T_a) @ T_b
    dt = np.linalg.norm(dT[:3,3])
    R = dT[:3,:3]
    ang = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    return dt, np.degrees(ang)

# def Rz(rad):
#     c, s = np.cos(rad), np.sin(rad)
#     return np.array([[c,-s,0],
#                      [s, c,0],
#                      [0, 0,1]], float)

# def apply_marker_yaw_to_T_marker_obj(T_marker_obj, yaw_deg):
#     yaw = np.deg2rad(yaw_deg)
#     T_corr = np.eye(4)
#     T_corr[:3,:3] = Rz(yaw)

#     return T_marker_obj @ T_corr

# def best_yaw_for_marker(T_cam_obj_ref, T_cam_marker_meas_B, T_marker_obj_B):
#     best = None
#     for yaw_deg in (0, 90, 180, 270):
#         T_marker_obj_try = apply_marker_yaw_to_T_marker_obj(T_marker_obj_B, yaw_deg)
#         T_cam_marker_pred = T_cam_obj_ref @ np.linalg.inv(T_marker_obj_try)
#         dt, dang = pose_delta(T_cam_marker_pred, T_cam_marker_meas_B)
#         score = dang 
#         if best is None or score < best["score"]:
#             best = {"yaw": yaw_deg, "dt": dt, "dang": dang, "score": score}
#     return best
#camera matrix coefficients for left (1) and right (2) and distortion coefficients for left and right
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

EPS = 1e-6

distortionCoefficientsLeft = np.array([k11, k12, p11, p12, k13, k14, k15, k16])

distortionCoefficientsRight = np.array([k21, k22, p21, p22, k23, k24, k25, k26])

cameraMatrixLeft= np.array([[fx1, 0, cx1],
               [0, fy1, cy1],
               [0, 0, 1]])

cameraMatrixRight= np.array([[fx2, 0, cx2],
               [0, fy2, cy2],
               [0, 0, 1]])




aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters=aruco.DetectorParameters()
detector=aruco.ArucoDetector(aruco_dict,parameters)

imgpath=Path(r"C:\Users\wehao\Downloads\Python\Markers")


shape = ["Square", "Dodecahedron", "Icosahedron"]
distance = ["0.25", "0.5", "0.75", "1"]
degrees = ["10", "20", "30", "40", "45"]
tag = ["Aruco", "Apriltag"]
folder = ["First day", "Second day", "Double"]



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
                        currentImgPath = newImgPath / (tag[0] + " " +shape_used[x] + " " + degrees[d] + "deg " + distance[l] + "m")
                        for f in currentImgPath.iterdir():
                            if f.is_file() and f.suffix.lower() == ".png":
                                detection(f, shape_used[x], distance[l], tag[0], degrees[d])






