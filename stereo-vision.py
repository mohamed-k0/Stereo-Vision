import cv2 as cv
import numpy as np


fx = 3979.911
fy = 3979.911
baseline = 193.001
doffs = 124.343
cx = 1244.772
cy = 1019.507

def main():
    # Load left and right images in Grayscale
    left_image = cv.imread("dataset/im0/im0.png", cv.IMREAD_GRAYSCALE)
    right_image = cv.imread("dataset/im0/im1.png", cv.IMREAD_GRAYSCALE)

    # Create Block Matching "object"
    block_matching = cv.StereoBM_create(numDisparities=272, blockSize=19) 

    # Calculate disparity using compute method
    disparity = block_matching.compute(left_image, right_image)


    # Convert the 16-bit fixed-point representation value to the actual disparity
    actual_disparity = disparity.astype(np.float32) / 16
    # Create a Conversion matrix (Q)
    Q = np.float32([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, fx],
    [0, 0, -1/baseline, 0]])

    # Calculate the depth (Z)
    x = 1500
    y = 1000
    depth = calc_depth(actual_disparity, fx, baseline, doffs)
    # Calculate the X, Y Coordinates relative to the principal point
    # X = (x - cx) * depth / fx
    # Y = (y - cy) * depth / fy

    # print (X , Y , depth)

    print("Selected Pixel:", (x, y))
    print(depth, "mm")


    # Normalize disparity for visualization
    visual_disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    # Convert to 8-bit unsigned integer
    visual_disparity = visual_disparity.astype(np.uint8)

    # Save the Disparity Map
    cv.imwrite("disparity_map.png", visual_disparity)
    # Show the Disparity Map
    cv.imshow("window",visual_disparity)








    cv.waitKey(0)
    cv.destroyAllWindows()





def calc_depth(disparity, fx, baseline, doffs):
    """ Z = (fx * Baseline) / (disparity + disparity offsets) 
    to account for difference in positions of principal points """
    return (fx * baseline) / (disparity + doffs)



if __name__ == "__main__":
    main()

