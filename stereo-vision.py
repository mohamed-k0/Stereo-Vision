import cv2 as cv

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

