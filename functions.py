import numpy as np
import cv2
from joblib import Parallel, delayed
import os
import image as im


def spectral_thresholding(path):
    gray_image = im.readImg(path, (400, 400), 'gray')
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist /= float(np.sum(hist))  # type: ignore
    ClassVarsList = np.zeros((256, 256))
    for bar1 in range(len(hist)):

        for bar2 in range(bar1, len(hist)):
            ForegroundLevels = []
            BackgroundLevels = []
            MidgroundLevels = []
            ForegroundHist = []
            BackgroundHist = []
            MidgroundHist = []
            for level, value in enumerate(hist):
                if level < bar1:
                    BackgroundLevels.append(level)
                    BackgroundHist.append(value)
                elif level > bar1 and level < bar2:
                    MidgroundLevels.append(level)
                    MidgroundHist.append(value)
                else:
                    ForegroundLevels.append(level)
                    ForegroundHist.append(value)

            FWeights = np.sum(ForegroundHist) / float(np.sum(hist))
            BWeights = np.sum(BackgroundHist) / float(np.sum(hist))
            MWeights = np.sum(MidgroundHist) / float(np.sum(hist))
            FMean = np.sum(np.multiply(
                ForegroundHist, ForegroundLevels)) / float(np.sum(ForegroundHist))
            BMean = np.sum(np.multiply(
                BackgroundHist, BackgroundLevels)) / float(np.sum(BackgroundHist))
            MMean = np.sum(np.multiply(MidgroundHist, MidgroundLevels)
                           ) / float(np.sum(MidgroundHist))
            BetClsVar = FWeights * BWeights * np.square(BMean - FMean) + \
                FWeights * MWeights * np.square(FMean - MMean) + \
                BWeights * MWeights * np.square(BMean - MMean)
            ClassVarsList[bar1, bar2] = BetClsVar
        max_value = np.nanmax(ClassVarsList)
    threshold = np.where(ClassVarsList == max_value)[0][0]  # type: ignore
    output_image = np.zeros_like(gray_image)
    output_image[gray_image > threshold] = 255

    os.remove("static/images/output/spectral_thresholding.jpg")
    pathOFResult = f"static/images/output/spectral_thresholding.jpg"
    cv2.imwrite(pathOFResult, output_image)

    return pathOFResult

#########################################################################################


def RGB_To_XYZ(image):
    # Convert the image to a numpy array
    XYZ_Image = np.array(image)
    # Normalize the RGB values to the range [0, 1]
    XYZ_Image = image / 255.0

    # Define the RGB to XYZ transformation matrix
    transMatrix = np.array([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])

    XYZ_Image = np.dot(XYZ_Image, transMatrix)
    return XYZ_Image


def xyz_to_luv(xyz, white_point=[0.95047, 1.0, 1.08883]):

    x, y, z = xyz / np.sum(xyz)

    u_green_red = 4 * x / (x + 15 * y + 3 * z)
    v_blue_yellow = 9 * y / (x + 15 * y + 3 * z)

    uw_ = 4 * white_point[0] / (white_point[0] +
                                15 * white_point[1] + 3 * white_point[2])
    vw_ = 9 * white_point[1] / (white_point[0] +
                                15 * white_point[1] + 3 * white_point[2])
    yw = y / white_point[1]

    if yw > 0.008856:
        Lightness = 116 * (yw ** (1/3)) - 16
    else:
        Lightness = 903.3 * yw

    u = 13 * Lightness * (u_green_red - uw_)
    v = 13 * Lightness * (v_blue_yellow - vw_)
    return Lightness, u, v


def RGB_LUV(path):
    img = cv2.imread(path)
    xyz_img = RGB_To_XYZ(img)
    luv_img = np.array(img)

    for i in range(luv_img.shape[0]):
        for j in range(luv_img.shape[1]):
            xyz1_img = xyz_img[i, j, :3]
            L, u, v = xyz_to_luv(xyz1_img)
            luv_img[i, j, :3] = [L, u, v]

    os.remove("static/images/output/luv.jpg")
    pathOFResult = f"static/images/output/luv.jpg"
    cv2.imwrite(pathOFResult, luv_img)

    return pathOFResult


############################################################################################################################

# Define a function to perform mean shift on a patch of the image
def mean_shift_patch(patch, bandwidth):
    # Convert the patch to float64
    pixels = np.float64(patch.reshape(-1, 3))
    # Loop over all the pixels in the patch
    for i in range(len(pixels)):  # type: ignore
        # Initialize the mean shift vector
        shift = np.array([1, 1, 1])
        # Loop until convergence
        while np.linalg.norm(shift) > 1:
            # Compute the mean of the pixels within the bandwidth
            kernel = pixels - pixels[i]
            kernel_norm = np.linalg.norm(kernel, axis=1)
            within_bandwidth = kernel_norm < bandwidth
            mean = np.mean(pixels[within_bandwidth], axis=0)
            # Compute the mean shift vector
            shift = mean - pixels[i]
            # Shift the pixel
            pixels[i] += shift  # type: ignore
    # Convert the pixels back to uint8 and reshape to patch shape
    segmented_patch = np.uint8(pixels.reshape(patch.shape))
    return segmented_patch


def mean_shift_segmentation(path, bandwidth):

    # Load the image
    img = im.readImg(path, (400, 400), 'rgb')

    # Define the patch size
    patch_size = (100, 100)

    # Split the image into patches
    patches = []
    for i in range(0, img.shape[0], patch_size[0]):
        for j in range(0, img.shape[1], patch_size[1]):
            patch = img[i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(patch)

    # Perform mean shift segmentation on each patch in parallel
    num_cores = 8  # Change this to the number of CPU cores you want to use
    segmented_patches = Parallel(n_jobs=num_cores)(
        delayed(mean_shift_patch)(patch, bandwidth) for patch in patches)

    # Combine the segmented patches into a single image
    segmented_img = np.zeros_like(img)
    k = 0
    for i in range(0, img.shape[0], patch_size[0]):
        for j in range(0, img.shape[1], patch_size[1]):
            segmented_img[i:i+patch_size[0], j:j+patch_size[1]
                          ] = segmented_patches[k]  # type: ignore
            k += 1

    os.remove("static/images/output/mean_segmented.jpg")
    pathOFResult = f"static/images/output/mean_segmented.jpg"
    cv2.imwrite(pathOFResult, segmented_img)

    return pathOFResult


#######################################################################################################################################

def optimal_threshold(path):
    img = im.readImg(path, (400, 400), 'rgb')
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initial threshold is the average of all pixels
    threshold = gray_img.mean()
    prev_threshold = -1

    while abs(threshold - prev_threshold) >= 1:
        # Calculate the average of the four corner pixels
        bg_avg = (gray_img[0, 0] +
                  gray_img[-1, 0] +
                  gray_img[0, -1] +
                  gray_img[-1, -1]) / 4.0

        # Calculate the average of other pixels
        fg_avg = np.mean(gray_img) - bg_avg

        # Calculate the new threshold
        prev_threshold = threshold
        threshold = (bg_avg + fg_avg) / 2.0

        # Modify the image array in place using boolean indexing
        gray_img[gray_img <= threshold] = 0
        gray_img[gray_img > threshold] = 255

        os.remove("static/images/output/optim_thres.jpg")
        pathOFResult = f"static/images/output/optim_thres.jpg"
        cv2.imwrite(pathOFResult, gray_img)

    return pathOFResult  # type: ignore
#######################################################################################################################################


def otsu_threshold(path):
    img = im.readImg(path, (400, 400), 'rgb')
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale image
    hist = np.zeros(256)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            pixel_value = gray_img[i, j]
            hist[pixel_value] += 1

    # Normalize the histogram
    hist /= (gray_img.shape[0] * gray_img.shape[1])

    # Calculate the cumulative sum and cumulative mean of the normalized histogram
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(np.arange(256) * hist)

    # Initialize variables for calculating the between-class variance and threshold
    max_var = 0
    threshold = 0

    # Iterate over all possible threshold values and calculate the between-class variance
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[t] / w0
        mu1 = (cum_mean[-1] - cum_mean[t]) / w1
        var_between = w0 * w1 * (mu0 - mu1)**2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Binarize the image using the computed threshold
    new_img = gray_img.copy()
    new_img[gray_img <= threshold] = 0  # type: ignore
    new_img[gray_img > threshold] = 255  # type: ignore

    os.remove("static/images/output/otsu_thres.jpg")
    pathOFResult = f"static/images/output/otsu_thres.jpg"
    cv2.imwrite(pathOFResult, new_img)
#######################################################################################################################################


def kmeans_segmentation(path, k, max_iterations):
    image = cv2.imread(path, cv2.IMREAD_COLOR)  # type: ignore
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape image to a 2D array of pixels
    pixels = np.float32(image.reshape(-1, 3))  # type: ignore

    # Initialize centroids randomly
    centroids = pixels[np.random.choice(
        pixels.shape[0], k, replace=False)]  # type: ignore

    # Run K-means algorithm for max_iterations
    for i in range(max_iterations):
        # Calculate distances between each pixel and each centroid
        distances = np.sqrt(
            np.sum((pixels - centroids[:, np.newaxis])**2, axis=2))

        # Assign each pixel to the closest centroid
        labels = np.argmin(distances, axis=0)

        # Update centroids to be the mean of all pixels assigned to them
        for j in range(k):
            centroids[j] = np.mean(pixels[labels == j], axis=0)

    # Assign each pixel to its final centroid
    final_labels = np.argmin(
        np.sqrt(np.sum((pixels - centroids[:, np.newaxis])**2, axis=2)), axis=0)

    # Reshape labels back to the shape of the original image
    kmean_segmented_image = final_labels.reshape(image.shape[:2])

    os.remove("static/images/output/k_means.jpg")
    pathOFResult = f"static/images/output/k_means.jpg"
    cv2.imwrite(pathOFResult, kmean_segmented_image)

    return pathOFResult
