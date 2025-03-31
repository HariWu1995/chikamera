This folder contains the calibration parameters for the three stereo systems 
(Pedestrian Tracking Mapathon)

______________________________________
intrinsics.txt
- intrinsic camera parameters of the left and right camera (before rectification)

+ M1 (Camera Matrix for Camera 1):
- A 3×3 intrinsic matrix that defines the camera's focal length, optical center, and skew.

+ D1 (Distortion Coefficients for Camera 1):
- A 5×1 vector that contains radial and tangential distortion coefficients.
- Used to correct lens distortions in images.

+ Cov1 (Covariance Matrix for Camera 1):
- A 9×9 matrix representing the uncertainty of intrinsic parameters.
- Helps in estimating calibration accuracy.

+ s1 (Sensor Size 1):
- The total number of pixels in Camera 1’s sensor.
- Typically width × height.

+ w1 (Width in Pixels):
- The horizontal resolution of Camera 1’s image.

+ h1 (Height in Pixels):
- The vertical resolution of Camera 1’s image.

______________________________________
extrinsics.txt
- relative orientation parameters of the stereo camera
- needed to stereo rectify the images
- camera parameters of the rectified images are contained in P1 and P2, respectively

+ F (Fundamental Matrix):
- A 3×3 matrix that describes the epipolar geometry between 2 cameras.
- It maps points from one image to their corresponding epipolar lines in the other image.
- Used in stereo vision for rectification and depth estimation.

+ E (Essential Matrix):
- A 3×3 matrix that encodes the rotation (R) and translation (T) between 2 camera coordinate systems.
- It is related to the fundamental matrix by E = K.T * F * K, where K is the camera intrinsic matrix.
- Used for computing relative motion between cameras.

+ R (Rotation Matrix):
- A 3×3 matrix representing the relative rotation between 2 cameras.
- Converts 3D points from one camera’s coordinate system to the other’s.

+ T (Translation Vector):
- A 3×1 vector that represents the translation between 2 cameras.
- Defines the shift in position from one camera to the other.

+ R1 (Rectification Rotation Matrix for Camera 1):
- A 3×3 matrix used for stereo rectification.
- Aligns Camera 1's coordinate system with a common rectified plane.

+ P1 (Projection Matrix for Camera 1):
- A 3×4 matrix that projects 3D world points into Camera 1’s image plane after rectification.

+ Q (Disparity-to-Depth Mapping Matrix):
- A 4×4 matrix used in stereo vision to convert disparity values into real-world depth measurements.

+ mirrorY:
- A flag (typically 0 or 1) indicating whether the image needs to be mirrored along the Y-axis.

+ ROI1x, ROI1y, ROI1w, ROI1h:
- Define the Region of Interest (ROI) for Camera 1 in pixels:
    . (ROI1x, ROI1y): Top-left corner.
    . (ROI1w, ROI1h): Width and height of the region.
- Used for image cropping or processing only a specific part of the frame.

+ Offsetx, Offsety, Offsetz:
- Represent spatial offsets in the X, Y, and Z directions.
- Used for fine adjustments in alignment between cameras.

+ Covar (Covariance Matrix):
- A 24×24 matrix that describes the uncertainty of extrinsic calibration parameters.
- Useful in error estimation and optimization during camera calibration.

______________________________________
absolute.txt
- absolute orientation parameters of the left rectified stereo camera
- the right camera has the same orientation but is shifted by the base length

- a point X in the global system can be projected to the left image by
 	x_left = K_left * R' * [E | -X0] * X

- a point X in the global system can be projected to the right image by
 	x_right = K_right * (R' * [E | -X0] * X + B)
	B = [b, 0, 0], b = baselength

- R can be calculated from the 3 Euler angles rpy by
	R = R_y * R_p * R_r

	R_y = [	cos(y)	-sin(y)	0
			sin(y)	cos(y)	0
			0		0		1]
	
	R_p = [	cos(p)	 0	 sin(p)
			0	 1	 0
			-sin(p)	 0	 cos(p)]
	
	R_r = [	1	 0 0;
			0	 cos(r)	 -sin(r)
			0	 sin(r)	 cos(r)]

