

import cv2
import numpy as np

# Load images
building_img = cv2.imread('bg02.png')   # Background
flag_img = cv2.imread('pic02.jpg')           # Overlay

if building_img is None or flag_img is None:
    print("Error loading images. Check file paths.")
    exit()

# Display window
cv2.namedWindow("Click 4 Points", cv2.WINDOW_NORMAL)
cv2.imshow("Click 4 Points", building_img)

# Step 4: Click 4 points
clicked_points = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(building_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click 4 Points", building_img)

cv2.setMouseCallback("Click 4 Points", select_points)

print("Click 4 points clockwise on a planar surface (e.g., wall)...")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    print("You must click exactly 4 points.")
    exit()

# Convert to numpy arrays
dst_pts = np.array(clicked_points, dtype=np.float32)

# Get flag image size and corners
h_flag, w_flag = flag_img.shape[:2]
src_pts = np.array([[0, 0], [w_flag, 0], [w_flag, h_flag], [0, h_flag]], dtype=np.float32)

# Step 5: Compute Homography
H, _ = cv2.findHomography(src_pts, dst_pts)

# Step 6: Warp flag image
warped_flag = cv2.warpPerspective(flag_img, H, (building_img.shape[1], building_img.shape[0]))

# Step 7: Blend images
mask = np.any(warped_flag != 0, axis=2).astype(np.uint8) * 255
mask_inv = cv2.bitwise_not(mask)

bg = cv2.bitwise_and(building_img, building_img, mask=mask_inv)
fg = cv2.bitwise_and(warped_flag, warped_flag, mask=mask)
result = cv2.add(bg, fg)

# Step 8: Show and save result
cv2.imshow("Flag Projected", result)
cv2.imwrite("result_projection.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
