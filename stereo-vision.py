import os   # Import os library to be able to manipulate directories and files inside the device
import re   # Import regular expressions library to find info inside calib.txt
import cv2 as cv
import numpy as np


FX = 6338.47
FY = 6338.47
BASELINE = 171.548
DOFFS = 479.489
CX0 = 1035.675
CX1 = 1515.164
CY= 960.073

NUM_DISPARITES = 400
BLOCK_SIZE = 19

def main():

    # Load left and right images in Grayscale
    left_image = cv.imread("dataset/img1/im0.png", cv.IMREAD_GRAYSCALE)
    right_image = cv.imread("dataset/img1/im1.png", cv.IMREAD_GRAYSCALE)

    # Create Block Matching "object"
    block_matching = cv.StereoBM_create(numDisparities=NUM_DISPARITES, blockSize=BLOCK_SIZE) 

    # Calculate disparity using compute method
    disparity = block_matching.compute(left_image, right_image)


    # Convert the 16-bit fixed-point representation value to the actual disparity
    actual_disparity = disparity.astype(np.float32) / 16


    # Selected Pixel
    x = 1000
    y = 1000
    # Calculate the depth of a chosen pixel (Z)
    depth = calc_depth(actual_disparity, FX, BASELINE, DOFFS, x, y)
    # print depth
    print(f"Depth of selected pixel: {depth:.2f} mm")

    # Define a Conversion matrix (Q) that fits the "Conditions of the Image"
    Q = np.float32([
    [-1, 0, 0, CX0],
    [0, -1, 0, CY],
    [0, 0, 0, FX],
    [0, 0, 1/BASELINE, DOFFS/BASELINE]])

    # Reproject points to 3D
    points_3d = cv.reprojectImageTo3D(actual_disparity, Q)
    # Create a Mask for "Valid" disparity values ( > 0 )
    valid = actual_disparity > 0
    # Use boolean indexing to filter invalid disparity
    valid_points = points_3d[valid]


    # Get the left image colored as RGB
    left_img_clr = cv.imread("dataset/img1/im0.png")
    left_img_clr = cv.cvtColor(left_img_clr, cv.COLOR_BGR2RGB)
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

# Get the information of calib.txt inside a dictionary
def read_calib(path): # -> dict
    with open(path, 'r') as file:
        text = file.read()

    # Define a function to get the matrices inside calib.txt (to allow reusability)
    def get_matrix(cam):
        # Search for a certain pattern recognized in the file
        match = re.search(rf"{cam}=\[(.*?)\]", text)    # search for cam#=[any character with any number of times]
        # Get the numerical values inside the matrix
        num_values = re.findall(r"-?\d+\.?\d*", match.group(1))     # get the pattern of either + or - and can be decimal or not
        # Put the values in a list to return it
        nums = []
        for num in num_values:
            nums.append(float(num))
        return nums

    # Use the get_matrix function on the two cams (0 and 1)
    cam0 = get_matrix("cam0")

    cam1 = get_matrix("cam1")

    # Define a function to Get the other values inside the file
    def get_value(var_name, typecast=float):    # set the initial value of type cast to float
        match = re.search(rf"{var_name}=(-?\d+\.?\d*)", text)   # looks for numbers
        # Return the first value found
        return typecast(match.group(1))

    # Return the dictionary of the values
    return {"fx": cam0[0], "fy": cam0[4], "cx0": cam0[2], "cx1": cam1[2], "cy": cam0[5], "baseline": get_value("baseline"), "doffs": get_value("doffs"), "ndisp": get_value("ndisp", typecast=int), "width": get_value("width", typecast=int), "height": get_value("height", typecast=int)}

def process_data(folder, output_folder, selected_pixel):
    # Create a dictionary that stores the data in calib.txt
    calib = read_calib(os.path.join(folder, "calib.txt"))
    # Unpack the values that we will need
    fx, fy, cx0, cy = calib["fx"], calib["fy"], calib["cx0"],calib["cy"]
    baseline, doffs, ndisp = calib["baseline"], calib["doffs"], calib["ndisp"]
    # Make the number of disparities a multiple of 16
    ndisp = int(np.ceil(ndisp / 16) * 16) # get the nearest highest integer and then multiply by 16 
def save_ply(filename, points, colors):

    try:
        with open(filename, 'w') as file:
            file.writelines(["ply\n", "format ascii 1.0\n",f"element vertex {len(points)}\n", "property float x\n", "property float y\n","property float z\n", "property uchar red\n", "property uchar green\n", "property uchar blue\n", 'end_header\n'])

            # Combine same index of the two arrays in an array to iterate over it
            for point, color in zip(points, colors):
                X, Y, Z = point
                R, G, B = color

                file.write(f"{X} {Y} {Z} {int(R)} {int(G)} {int(B)}\n")

            print("Polygon file created successfully.")
    except Exception:
        print("Couldn't create a file!")
        



def calc_depth(disparity, fx, Baseline, doffs, x, y):
    """ Z = (fx * Baseline) / (disparity + disparity offsets) 
    to account for difference in positions of principal points """
    print("Selected Pixel:", (x,y))
    return (fx * Baseline) / (disparity[y, x] + doffs)



if __name__ == "__main__":
    main()

