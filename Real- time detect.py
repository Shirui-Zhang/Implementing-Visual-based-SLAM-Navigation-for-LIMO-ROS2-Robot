import cv2
import numpy as np
import argparse
import sys

# === 命令行参数（可省略，或者你写死类型） ===
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_1000",
                help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# === ArUCo 字典列表 ===
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# === 检查类型是否支持 ===
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"[ERROR] ArUCo tag of '{args['type']}' is not supported.")
    sys.exit(0)

print(f"[INFO] Detecting '{args['type']}' tags...")
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# === 初始化摄像头 ===
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    sys.exit(1)

# === 识别循环 ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法从摄像头读取画面")
        break

    # 可选：加边框（如果你希望）
    # frame = np.pad(frame, pad_width=100, constant_values=255)

    # 检测 ArUco 标记
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if corners and len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topLeft = tuple(map(int, topLeft))
            topRight = tuple(map(int, topRight))
            bottomRight = tuple(map(int, bottomRight))
            bottomLeft = tuple(map(int, bottomLeft))

            # 画线 + 画 ID
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()