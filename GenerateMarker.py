import cv2
import numpy as np

def generate_aruco_markers(
    num_markers=8,           # <-- how many to generate
    start_id=0,              # first ID to use
    marker_size=400,         # inner marker size (px)
    border_size=50           # white border (px)
):
    print(f"Generating {num_markers} ArUco markers (IDs {start_id}..{start_id+num_markers-1})")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    max_ids = aruco_dict.bytesList.shape[0]
    if start_id + num_markers > max_ids:
        raise ValueError(f"Requested IDs exceed dictionary size ({max_ids}).")

    for marker_id in range(start_id, start_id + num_markers):
        # base white square
        marker_image = np.full((marker_size, marker_size), 255, dtype=np.uint8)

        # draw the marker
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, marker_size, marker_image, 1
        )

        # add border
        canvas = np.full(
            (marker_size + 2*border_size, marker_size + 2*border_size), 255, dtype=np.uint8
        )
        canvas[border_size:border_size+marker_size, border_size:border_size+marker_size] = marker_image

        # to BGR and label
        img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, f"ID: {marker_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # save
        filename = f"aruco_marker_{marker_id}.png"
        cv2.imwrite(filename, img)
        print(f"Generated {filename}")

    print("\nInstructions:")
    print("1) Print these marker images (keep scale consistent).")
    print("2) Keep them flat, matte, and well lit.")
    print("3) Point the iPad camera at the markers.")
    print("4) Run aruco_ar.py to see AR overlays.")

if __name__ == "__main__":
    generate_aruco_markers(num_markers=8)
