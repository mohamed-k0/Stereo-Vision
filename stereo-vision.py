import cv2 as cv
import numpy as np

# Load left and right images in Grayscale
left_image = cv.imread("dataset/im0/im0.png", cv.IMREAD_GRAYSCALE)
right_image = cv.imread("dataset/im0/im1.png", cv.IMREAD_GRAYSCALE)

print("Left image shape:", left_image.shape)
print("Right image shape:", right_image.shape)

# Create Block Matching "object"
block_matching = cv.StereoBM_create(numDisparities=272, blockSize=15)

# Calculate disparity using compute method
disparity = block_matching.compute(left_image, right_image)

print(disparity.shape)

# Normalize disparity for visualization
visual_disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)

# Convert to 8-bit unsigned integer
visual_disparity = visual_disparity.astype(np.uint8)

# Showing the Disparity Map
cv.imshow("window",visual_disparity)

cv.waitKey(0)
cv.destroyAllWindows()

