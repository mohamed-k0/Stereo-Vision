# Stereo Vision
- This program takes a *dataset* containing folders that contain (left image, right image, "calib.txt" file) and processes each one of the folders.<br>
- Per each folder processed:
    <ol type = "1">
    <li> "calib.txt" file is read through a function that detects data using constant pattern of <b>Regular Expressions</b>.</li>
    <li> A selected pixel is used to <b>calculate the its depth</b> and return it using a function named "calc_depth" that uses the equation ( Z = f * baseline / (d + d_offset)).</li>
    <li> Disparity is calculated to all pixels in the image and filtered by valid disparities (> 0) using a "<i>mask</i>".</li>
    <li> A disparity map is created that is then shown in a <strong>png image</strong> named "disparity_map.png".</li>
    <li> A <b>Polygon file</b> is created to be able to view the image in 3-D software.</li>
    </ol>
---
## Dependencies:
The program utilizes the (**Open cv**) python library mainly to process images, (**numpy**) for scientific mathematical calculations, (**os**) for operating system access and file manipulation and (**re**) for using regular expressions.  
