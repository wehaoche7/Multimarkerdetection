import cv2
from pupil_apriltags import Detector
import os
from pathlib import Path
import numpy as np
import csv
import pyzed.sl as sl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
print(cv2.__version__)
zed = sl.Camera()
init_params = sl.InitParameters()

#Change depending on test data (top normal bottom depth)
img_path=Path(r"C:\Users\wehao\Downloads\Python\Markers")
video_path = r"C:\Users\wehao\Downloads\Python\Markers\Winged\HD720_SN12041574_12-08-57.svo2"
init_params.set_from_svo_file(video_path)

depth_zed = sl.Mat()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
frame_choice_start = int(input("Please put in an starting frame"))
frame_choice_end = int(input("Please put in an ending frame"))

err = zed.open(init_params)
zed.set_svo_position(frame_choice_start)
cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left_calibration = calib.left_cam
right_calibration = calib.right_cam
cameraMatrixLeft = np.array([
    [left_calibration.fx, 0,       left_calibration.cx],
    [0,       left_calibration.fy, left_calibration.cy],
    [0,       0,       1]
], dtype=np.float64)

cameraMatrixRight = np.array([
    [right_calibration.fx, 0,       right_calibration.cx],
    [0,       right_calibration.fy, right_calibration.cy],
    [0,       0,       1]
], dtype=np.float64)
distortionCoefficientsLeft = np.zeros(5)
distortionCoefficientsRight = np.zeros(5)
print("LEFT:", left_calibration.fx, left_calibration.fy, left_calibration.cx, left_calibration.cy)
print("RIGHT:", right_calibration.fx, right_calibration.fy, right_calibration.cx, right_calibration.cy)
if err != sl.ERROR_CODE.SUCCESS:
    print("Could not open SVO2:", err)
    exit()
    
runtime_params = sl.RuntimeParameters()

zed_left = sl.Mat()
zed_right = sl.Mat()


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
                     45 : 0.013, 33 : 0.013,576 : 0.013, 324 : 0.013, 494 : 0.01, 536 : 0.01, 321 : 0.01, 349 : 0.01, 586 : 0.01, 293 : 0.01, 582 : 0.01, 222 : 0.01, 18 : 0.01, 337 : 0.01, 339 : 0.01, 510 : 0.01},
    "Winged" : {341 : 0.065, 185 : 0.065, 24 : 0.065, 226 : 0.065, 547 : 0.065, 78 : 0.05, 511 : 0.05, 26 : 0.05, 438 : 0.05, 254 : 0.05, 69 : 0.05, 32 : 0.05, 200 : 0.05},
    "BasePlate" :{373 : 0.0792, 503  : 0.0792, 284 : 0.0792, 323 : 0.0792, 463 : 0.0792, 10 : 0.0792}
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

    return T_corr @ T_marker_obj 

def pose_delta(T_a, T_b):
    # returns translation diff (m) and rotation diff (deg)
    dT = np.linalg.inv(T_a) @ T_b
    dt = np.linalg.norm(dT[:3,3])
    R = dT[:3,:3]
    ang = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    return dt, np.degrees(ang)

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
            T_marker_obj[:3,:3] = R
            T_marker_obj[:3,3]  = t
            out[mid] = T_marker_obj

    return out

def get_depth(depth_map, pixels, radius = 3):
    x_corner = int(round(pixels[0]))
    y_corner = int(round(pixels[1]))

    y0 = max(0, y_corner - radius)
    y1 = min(depth_map.shape[0], y_corner + radius + 1)
    x0 = max(0, x_corner - radius)
    x1 = min(depth_map.shape[1], x_corner + radius + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[
        np.isfinite(patch) &
        (patch > 0)
    ]
    if len(valid) == 0:
        return None
    
    return float(np.median(valid))

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
    ok, rVecs, tVecs,_ = cv2.solvePnPGeneric(corners, imagePoints, cameraMatrix, distortionCoefficients,  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or len(rVecs) == 0:
        return None
    best = None
    error = 1e18

    for rVecs, tVecs in zip(rVecs, tVecs):
        if tVecs[2,0] <= 0:
            print("z negative -> corner order / pose ambiguity issue")
            continue
        projection, _ = cv2.projectPoints(corners, rVecs, tVecs, cameraMatrix, distortionCoefficients)
        projection = projection.reshape(-1,2)
        calculatedError = float(np.mean(np.linalg.norm(projection-imagePoints, axis=1)))
        if calculatedError <= error:
            error = calculatedError
            best = (rVecs, tVecs)
    return best


def getMarkerSize(markerId, shape):
    markerId = int(markerId)
    return sizeByShape.get(shape, {}).get(markerId, None)

def cornerStone(markerSize):
    h=markerSize/2
    # return np.array([[-h, -h, 0.0],
    #                 [h, -h, 0.0],
    #                 [h, h, 0.0],
    #                 [-h, h, 0.0]])
    return np.array([
        [-h,  h, 0.0],   # top-left
        [ h,  h, 0.0],   # top-right
        [ h, -h, 0.0],   # bottom-right
        [-h, -h, 0.0],   # bottom-left
    ], dtype=np.float32)

def referencePicker(mids, T_cam_marker_meas, marker_dict):
    bestA = None
    bestScore = float("inf")
    bestB = None
    for A in mids:
        T_cam_obj_ref = T_cam_marker_meas[A] @ marker_dict[A]
        totalScore = 0.0
        perB = {}
        for B in mids:
            if B == A:
                continue
            res = best_yaw_for_marker(T_cam_obj_ref, T_cam_marker_meas[B], marker_dict[B])
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

def H_to_pose(H_matrix):
    position = H_matrix[:3,3]
    rotation_matrix = H_matrix[:3,:3]
    qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()

    return np.array([
        position[0],
        position[1],
        position[2]
    , qx, qy, qz, qw])


def markerPose(markerId, shape, markerCorner, markerMiddle, side, cameraMatrix, distortionCoefficients, T_cam_marker_meas):
    tagId =  int(markerId)
    Size = getMarkerSize(tagId, shape)

    if Size is None:
        print("not in working order")
        return
    
    CornerPoints = cornerStone(float(Size))
    CornerPoints = CornerPoints.reshape(4, 3).astype(np.float32)
    # imagePoint = markerCorner.reshape(4,2).astype(np.float32)
    imagePoint = markerCorner[[3, 2, 1, 0]].astype(np.float32)
    for i, (x,y) in enumerate(markerCorner):
        cv2.putText(side, str(i), (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(side, f"id={str(tagId)}" , (int(markerMiddle[0])+25, int(markerMiddle[1])),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    rVec_markers, tVec_markers = choosePose(CornerPoints, imagePoint, cameraMatrix, distortionCoefficients)
    R_cam_marker, _ = cv2.Rodrigues(rVec_markers.reshape(3,1))
    T = np.eye(4)
    T[:3,:3] = R_cam_marker
    T[:3,3]  = tVec_markers[:,0]
    T_cam_marker_meas[tagId] = T
    # print(T[2,3], "potato", tagId)
    cv2.circle(side, center=(int(markerMiddle[0]), int(markerMiddle[1])), radius=5, color=(0, 250,0))
    cv2.drawFrameAxes(side, cameraMatrix, distortionCoefficients, rVec_markers, tVec_markers, 0.02)

def posePicker(mids_obj, T_cam_tracked_obj, side, cameraMatrix, distortionCoefficients, marker_dictionary, side_name):
    if len(mids_obj) < 2:
        if len(mids_obj) == 0:
            print("Consistency check skipped: <2 markers detected in left frame.")
            return None
        else:
            T_cam_obj = T_cam_tracked_obj[mids_obj[0]] @ marker_dictionary[mids_obj[0]]
            R_obj = T_cam_obj[:3, :3]
            T_obj = T_cam_obj[:3, 3]
            rVec_obj,_  =cv2.Rodrigues(R_obj)
            tVec_obj = T_obj
            cv2.drawFrameAxes(side, cameraMatrix, distortionCoefficients, rVec_obj, tVec_obj, 0.03)
            return T_cam_obj
    else:
        A, score, perB = referencePicker(mids_obj, T_cam_tracked_obj, marker_dictionary)
        inliers = [A]
        outliers = []
        for B, res in perB.items():
            if res["dang"] > 12.0 or res["dt"] > 0.02:
                outliers.append(B)
            else:
                inliers.append(B)
        
        print(side_name, " inliers:", inliers,"\n", side_name, " outliers:", outliers, "\n", sep="")


        T_list = [
    T_cam_tracked_obj[mid] @ marker_dictionary[mid]
    for mid in inliers
]
        T_cam_obj = fuse_T(T_list)
        R_obj= T_cam_obj[:3, :3]
        T_obj = T_cam_obj[:3, 3]
        rVec_obj,_  = cv2.Rodrigues(R_obj)
        tVec_obj = T_obj

        cv2.drawFrameAxes(side, cameraMatrix, distortionCoefficients, rVec_obj, tVec_obj, 0.05)

    return T_cam_obj

def detection(path, shape, distance=None, tag=None, degrees=None):
    T_base_reference = None
    while int(zed.get_svo_position()) <= frame_choice_end:
        T_cam_marker_meas_right = {}
        T_cam_marker_meas_left = {}
        T_cam_base_right = {}
        mids_base_right = []
        mids_obj_right = []
        T_cam_base_left = {}
        mids_base_left = []
        mids_obj_left = []
        
        Tvec_ee_right = None
        Tvec_ee_left = None
        Tvec_base_left = None
        Tvec_base_obj_left = None
        Tvec_base_right = None
        Tvec_base_obj_right = None
        err = zed.grab(runtime_params)
        frame_position = zed.get_svo_position()
        used_clahe_L = False
        used_clahe_R = False

        
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_left, sl.VIEW.LEFT)
            zed.retrieve_image(zed_right, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            frame_left = zed_left.get_data()
            depth_map = depth_zed.get_data()
            frame_right = zed_right.get_data()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            left_g  = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            frame = zed_left.get_data()
            height  = frame.shape[:2]
            print("Frame resolution:", height, "x", height)
            print("Full shape:", frame.shape)
            dL = detector.detect(left_g)
            dR = detector.detect(right_g)

            

            if len(dL) == 0:
                dL = detector.detect(clahe.apply(left_g))
                used_clahe_L = True

            if len(dR) == 0:
                dR = detector.detect(clahe.apply(right_g))
                used_clahe_R = True

            tag_ids_left = [tag.tag_id for tag in dL]
            tag_ids_right = [tag.tag_id for tag in dR]

            print("\nMarkers detected left:", tag_ids_left)
            print("Markers detected right:", tag_ids_right)

            if len(dL) == 0:
                counters["left_missing"] += 1

            if len(dR) == 0:
                counters["right_missing"] += 1
            
            if shape == "Square":
                marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\Cubegeometry.coord_systems_rel_Apriltag_fileCoM_semicolon.csv", obj_name="CoM")
            elif shape == "Dodecahedron":
                marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\Dodecacorrect.coord_systems_rel_Apriltag_fileCoM_semicolon.csv", obj_name="CoM")
            elif shape == "Icosahedron":
                marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\Truncasted icosahedron.coord_systems_rel_Apriltag_fileCoM_semicolon.csv", obj_name="CoM")
            elif shape == "Winged":
                marker_obj_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\LFD_handle_REV-1.1_WC.coord_systems_rel_test_fileCoM_semicolon_version4.csv", obj_name="CoM")
            base_plate_dict = load_marker_obj_dict(r"C:\Users\wehao\Downloads\Objects\baseplate(Correct orientation).coord_systems_rel_Trial_fileCoM_semicolon.csv", obj_name="CoM")
            for c in dL:  
                for l in range(c.corners.shape[0]):
                    cv2.circle(frame_left, center=(int(c.corners[l][0]), int(c.corners[l][1])), radius=5, color=(0, 0,250))

                if c.tag_id in sizeByShape["BasePlate"]:
                    mids_base_left.append(c.tag_id)
                    markerPose(c.tag_id, "BasePlate", c.corners, c.center, frame_left, cameraMatrixLeft, distortionCoefficientsLeft, T_cam_base_left)

                if c.tag_id in sizeByShape[shape]:
                    mids_obj_left.append(c.tag_id)
                    markerPose(c.tag_id, shape, c.corners, c.center, frame_left, cameraMatrixLeft, distortionCoefficientsLeft, T_cam_marker_meas_left)
                depth = get_depth(depth_map, c.center)

                print(
                    "Marker:", c.tag_id,
                    "ZED depth:", depth,
                    "markermarker", 
                )
                

            Tvec_obj_left = posePicker(mids_obj_left, T_cam_marker_meas_left, frame_left, cameraMatrixLeft, distortionCoefficientsLeft, marker_obj_dict, "left")
            if Tvec_obj_left is not None:
                Tvec_ee_left = Tvec_obj_left @ np.linalg.inv(marker_obj_dict[Endeffector])
                t_ee_left = Tvec_ee_left[:3, 3]
                r_ee_left, _ = cv2.Rodrigues(Tvec_ee_left[:3, :3])
                cv2.drawFrameAxes(frame_left, cameraMatrixLeft, distortionCoefficientsLeft, r_ee_left, t_ee_left, 0.02)

            if mids_base_left:
                Tvec_base_left = posePicker(mids_base_left, T_cam_base_left, frame_left, cameraMatrixLeft, distortionCoefficientsLeft, base_plate_dict, "left")
                if Tvec_base_left is not None and T_base_reference is None:
                    T_base_reference = Tvec_base_left.copy()


            if Tvec_ee_left is not None and T_base_reference is not None:
                Tvec_base_obj_left = (
                    np.linalg.inv(T_base_reference)
                    @ Tvec_ee_left
                )
                Tvec_pose_left[frame_position] = Tvec_base_obj_left.copy()

            # if Tvec_ee_left is not None and Tvec_base_left is not None:
            #     Tvec_base_obj_left = np.linalg.inv(Tvec_base_left) @ Tvec_ee_left
            #     Tvec_position_left[frame_position] = Tvec_base_obj_left.copy()
            #     T_base_CoM = np.linalg.inv(Tvec_base_left) @ Tvec_obj_left

            #     T_base_CoM = np.linalg.inv(Tvec_base_left) @ Tvec_obj_left
            #     p = T_base_CoM[:3, 3]
            #     R = T_base_CoM[:3, :3]

            #     print("p:", p)
            #     print("R:\n", R)
            #     print("camera CoM:", Tvec_obj_left[:3, 3])

            #     T_base_CoM = np.linalg.inv(Tvec_base_left) @ Tvec_obj_left
            #     print("base CoM:", T_base_CoM[:3, 3])
            #     T_cam_base_reference = Tvec_base_left.copy()
            #     T_base_CoM_fixed = (
            #         np.linalg.inv(T_cam_base_reference)
            #         @ Tvec_obj_left
            #     )

            #     print(T_base_CoM_fixed[:3, 3])


            else:
                Tvec_pose_left[frame_position] = None     


            for d in dR:  
                for f in range(d.corners.shape[0]):
                    cv2.circle(frame_right, center=(int(d.corners[f][0]), int(d.corners[f][1])), radius=5, color=(0, 0,250))
                if d.tag_id in sizeByShape["BasePlate"]:
                    mids_base_right.append(d.tag_id)
                    markerPose(d.tag_id, "BasePlate", d.corners, d.center, frame_right, cameraMatrixRight, distortionCoefficientsRight, T_cam_base_right)

                if d.tag_id in sizeByShape[shape]:
                    mids_obj_right.append(d.tag_id)
                    markerPose(d.tag_id, shape, d.corners, d.center, frame_right, cameraMatrixRight, distortionCoefficientsRight, T_cam_marker_meas_right)

            Tvec_obj_right = posePicker(mids_obj_right, T_cam_marker_meas_right, frame_right, cameraMatrixRight, distortionCoefficientsRight, marker_obj_dict, "right")
            if Tvec_obj_right is not None:
                Tvec_ee_right = Tvec_obj_right @ np.linalg.inv(marker_obj_dict[Endeffector])
                t_ee = Tvec_ee_right[:3, 3]
                r_ee, _ = cv2.Rodrigues(Tvec_ee_right[:3, :3])
                cv2.drawFrameAxes(frame_right, cameraMatrixRight, distortionCoefficientsRight, r_ee, t_ee, 0.02)

            if mids_base_right:
                Tvec_base_right = posePicker(mids_base_right, T_cam_base_right, frame_right, cameraMatrixRight, distortionCoefficientsRight, base_plate_dict, "right")

            if Tvec_ee_right is not None and Tvec_base_right is not None:
                    Tvec_base_obj_right = np.linalg.inv(Tvec_base_right) @ Tvec_ee_right
                    Tvec_pose_right[frame_position] = Tvec_base_obj_right.copy()
            else:
                Tvec_pose_right[frame_position] = None   



            cv2.imshow('detecting markers left', frame_right)
            cv2.imshow('detecting markers right', frame_right)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO reached")
            break

    print(used_clahe_L, used_clahe_R)

    print(f"Currently inspecting:"
          f"{tag + ' ' if tag else ' '}"
            f"{shape + " at "}"
            f"{degrees + "°" + ' ' if degrees else ''}" 
            f"{distance}m")

    cv2.destroyAllWindows()

# To avoid trouble with the already programmed code
Endeffector = 000

shape = ["Square", "Dodecahedron", "Icosahedron", "Winged"]
distance = ["0.25", "0.5", "0.75", "1"]
degrees = ["10", "20", "30", "40"]
degrees = ["10", "20", "30", "40", "45"]
tag = ["Aruco", "Apriltag"]
folder = ["First day", "Second day", "Winged"]
Tvec_pose_left = {}
Tvec_pose_right = {}
i = 0
EPS = 1E-6

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

i = 0

counters = {
    "left_missing": 0,
    "right_missing": 0,
    "left_correct_objectframe": 0,
    "right_correct_objectframe": 0,
    "total": 0,
    "total 0.25m":0,
    "total 0.5m":0,
    "total 0.75m":0,
    "total 1m":0
}

distortionCoefficientsLeft = np.array([k11, k12, p11, p12, k13, k14, k15, k16])

distortionCoefficientsRight = np.array([k21, k22, p21, p22, k23, k24, k25, k26])

cameraMatrixLeft= np.array([[fx1, 0, cx1],
               [0, fy1, cy1],
               [0, 0, 1]])

cameraMatrixRight= np.array([[fx2, 0, cx2],
               [0, fy2, cy2],
               [0, 0, 1]])


choice = input("Please input, which data file you would wish to access from: First Day (1), Second Day (2), Winged(3)\n")
shapechoice = input("Please input, which shape you would like to inspect from: Square (1), Dodecahedron (2), Truncated Icosahedron (3), Winged(4), or all shapes (5)\n")


if int(shapechoice) == 5:
    shape_used = shape
else:
    shape_used = [shape[int(shapechoice)-1]]
if int(choice) == 1:    
        newImgPath = img_path / folder[int(choice) - 1]
        for x in range(len(shape_used)):
             for l in range(len(distance)):
                currentImgPath = newImgPath / (shape_used[x] + " " + distance[l] + "m")
                for f in currentImgPath.iterdir():
                    if f.is_file() and f.suffix.lower() == ".png":
                        i+=1
                        print("Current iteration", i)
                        detection(f, shape_used[x], distance[l],)
            
elif int(choice) == 2:
        distance = distance [0:2]
        newImgPath = img_path / folder[int(choice) - 1]
        for x in range(len(shape_used)):
                for l in range(len(distance)):
                    for d in range(len(degrees)):
                        currentImgPath = newImgPath / (tag[1] + " " + shape_used[x] + " " + degrees[d] + "deg " + distance[l] + "m")
                        for f in currentImgPath.iterdir():
                            if f.is_file() and f.suffix.lower() == ".png":
                                i+=1
                                print("Current iteration", i)
                                detection(f, shape_used[x], distance[l], tag[1], degrees[d])
elif int(choice) == 3:
        newImgPath = img_path / folder[int(choice) - 1]
        for f in newImgPath.iterdir():
            if f.is_file() and f.suffix.lower() == ".svo2":
                total_frames = zed.get_svo_number_of_frames()
                # i+=1
                print("Current iteration", i)
                detection(f, "Winged")
                print("Total frames:", total_frames)
                frame_positions = sorted(Tvec_pose_left.keys())


                positions = np.array([
                    Tvec_pose_left[frame][:3, 3]
                    for frame in frame_positions
                    if Tvec_pose_left[frame] is not None
                ])

                pose_data = np.array([
                    H_to_pose(Tvec_pose_left[frame])
                    for frame in frame_positions
                    if Tvec_pose_left[frame] is not None
                ])
                print(pose_data)
                x = positions[:, 0]
                y = positions[:, 1]
                z = positions[:, 2]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                ax.plot(x, y, z)
                ax.scatter(x[0], y[0], z[0], label="Start")
                ax.scatter(x[-1], y[-1], z[-1], label="End")
                max_range = max(
                    x.max() - x.min(),
                    y.max() - y.min(),
                    z.max() - z.min()
                )

                # Centre of each axis
                x_mid = (x.max() + x.min()) / 2
                y_mid = (y.max() + y.min()) / 2
                z_mid = (z.max() + z.min()) / 2

                half = max_range / 2

                ax.set_xlim(x_mid - half, x_mid + half)
                ax.set_ylim(y_mid - half, y_mid + half)
                ax.set_zlim(z_mid - half, z_mid + half)

                ax.set_box_aspect((1, 1, 1))
                ax.set_xlabel("X [m]")
                ax.set_ylabel("Y [m]")
                ax.set_zlabel("Z [m]")
                ax.legend()

                plt.show()
                                
print("Markers missed left", counters["left_missing"],"/", counters["total"])
print("Markers missed right", counters["right_missing"],"/", counters["total"])
print("Orientation left object frame correctness", counters["left_correct_objectframe"],"/", counters["total"]-counters["left_missing"])
print("Orientation right object frame correctness", counters["right_correct_objectframe"],"/", counters["total"]-counters["right_missing"])