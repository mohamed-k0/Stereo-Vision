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

    # Calculate the depth (Z)
    depth = calc_depth(actual_disparity, fx, baseline, doffs, 1500, 1000)
    # print depth
    print(f"Depth of selected pixel: {depth} mm")

    # Define a Conversion matrix (Q)
    Q = np.float32([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, fx],
    [0, 0, -1/baseline, doffs/baseline]])

    # Reproject points to 3D
    points_3d = cv.reprojectImageTo3D(actual_disparity, Q)

    # Create a Mask for "Valid" disparity values ( > 0)
    valid = actual_disparity > 0
    # Use boolean indexing to filter invalid disparity
    valid_points = points_3d[valid]


    # Get the left image colored as RGB
    left_img_clr = cv.imread("dataset/im0/im0.png", cv.COLOR_BGR2RGB)
    # Use Boolean indexing to filter disparity
    valid_colored = left_img_clr[valid]

    # Save resulting point cloud as ".ply" file (text format storing 3D geometry)
    save_ply(filename="point-cloud.ply",points=valid_points, colors=valid_colored)
   


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


def save_ply(filename, points, colors):

    try:
        with open(filename, 'w') as file:
            file.writelines(["ply\n", "format ascii 1.0\n",f"element vertex {len(points)}\n", "property float x\n", "property float y\n","property float z\n", "property uchar red\n", "property uchar green\n", "property uchar blue\n", 'end_header\n'])

            # Combine same index of the two arrays in an array to iterate over it
            for point, color in zip(points, colors):
                X, Y, Z = point
                R, G, B = color

                file.write(f"{X} {Y} {Z} {R} {G} {B}\n")

            print("Polygon file created successfully.")
    except:
        print("Couldn't create a file!")
        



def calc_depth(disparity, fx, baseline, doffs, x, y):
    """ Z = (fx * Baseline) / (disparity + disparity offsets) 
    to account for difference in positions of principal points """
    print("Selected Pixel:", (x,y))
    return (fx * baseline) / (disparity[y, x] + doffs)



if __name__ == "__main__":
    main()

