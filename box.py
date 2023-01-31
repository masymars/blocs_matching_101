import cv2
import numpy as np
from math import inf

frame1 = cv2.imread('image092.png')
gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 0]

frame2 = cv2.imread('image072.png')
gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def calc_mse(block1, block2):
    """
    Computes mean-squared error between two blocks.
    """
    error = np.sum((block1 - block2) ** 2)
    error = error / (block1.shape[0] * block1.shape[1])
    return error


# Initialize the predicted image with all pixels set to 255
predicted_image = (np.ones((1080, 1920)) * 255).astype(np.uint8)
neighborhood = 7

# Loop over 16x16 blocks in the first image
for i in range(0, gray_frame1.shape[0] - 16, 16):
    for j in range(0, gray_frame1.shape[1] - 16, 16):
        block1 = gray_frame1[i:(i + 16), (j):(j + 16)]
        b1_coords = [i, (i + 16), j, (j + 16)]
        min_error = inf

        # Loop over a neighborhood of blocks in the second image
        for m in range(max((i - neighborhood), 0),
                       min((i + neighborhood), (gray_frame1.shape[0] - 15))):
            for n in range(max((j - neighborhood), 0),
                           min((j + neighborhood), (gray_frame1.shape[1] - 15))):
                block2 = gray_frame2[m:(m + 16), n:(n + 16)]
                error = calc_mse(block1, block2)
                if error < min_error:
                    min_error = error
                    b2_coords = [m, m + (16), n, n + (16)]

        # If the minimum error is greater than 50, draw rectangles around the blocks
        if min_error >= 50:
            frame1 = cv2.rectangle(frame1, (b1_coords[2], b1_coords[0]),
                                   (b1_coords[3], b1_coords[1]), (0, 255, 0), 2)
            frame2 = cv2.rectangle(frame2, (b2_coords[2], b2_coords[0]),
                                   (b2_coords[3], b2_coords[1]), (255, 255, 0), 2)
        predicted_image[i:(i + 16), j:(j + 16)] = gray_frame2[b2_coords[0]:b2_coords[1], b2_coords[2]:b2_coords[3]]
resedu = gray_frame1 - predicted_image
reconst = resedu + predicted_image



cv2.imshow("predicted_image", predicted_image)
cv2.waitKey(0)


cv2.imshow("resedu", resedu)
cv2.waitKey(0)


cv2.imshow("reconst", reconst)
cv2.waitKey(0)

cv2.imshow("frame1", frame1)
cv2.waitKey(0)
cv2.imshow("frame2", frame2)
cv2.waitKey(0)


# Create a black image with shape of 1080x1920, using 8-bit unsigned integers
imageN = np.zeros((1080, 1920), dtype=np.uint8)

# Create a black image with shape of 1080 + 128 x 1920 + 128, using 8-bit unsigned integers
imagepad = np.zeros((1080 + 128, 1920 + 128), dtype=np.uint8)

# Assign values from gray_frame2 to a region starting from (64, 64) and ending at (1144, 1984) of imagepad
# This pads the original imageN by 128 pixels on all sides
imagepad[64:1144, 64:1984] = gray_frame2


def calculate_deco_residual(step, b1cor, bloc1):
    # Initialize newblock with the original block coordinates
    newblock = b1cor

    # Continue looping while step is greater than or equal to 1
    while step >= 1:
        cord = []

        # Create a list of 8 blocks with relative coordinates to newblock
        cord.append([newblock[0] - step, newblock[1] - step, newblock[2] - step, newblock[3] - step])
        cord.append([newblock[0] - step, newblock[1] - step, newblock[2], newblock[3]])
        cord.append([newblock[0] - step, newblock[1] - step, newblock[2] + step, newblock[3] + step])
        cord.append([newblock[0], newblock[1], newblock[2] - step, newblock[3] - step])
        cord.append([newblock[0], newblock[1], newblock[2], newblock[3]])
        cord.append([newblock[0], newblock[1], newblock[2] + step, newblock[3] + step])
        cord.append([newblock[0] + step, newblock[1] + step, newblock[2] - step, newblock[3] - step])
        cord.append([newblock[0] + step, newblock[1] + step, newblock[2], newblock[3]])
        cord.append([newblock[0] + step, newblock[1] + step, newblock[2] + step, newblock[3] + step])

        # Calculate the MSE between each block in cord and the original block
        mini = float('inf')
        for k in cord:
            neighbor = imagepad[k[0]:k[1], k[2]:k[3]]
            loss = calc_mse(neighbor, bloc1)

            # Select the block with the minimum MSE
            if loss < mini:
                mini = loss
                newblock = k

        # Halve the step size for the next iteration
        step = step // 2

    # Check if the MSE is greater than 50
    if mini > 50:
        # Return the residual between the original block and the selected block
        return bloc1 - imagepad[newblock[0]:newblock[1], newblock[2]: newblock[3]]

    # Return a zero numpy array if MSE is not greater than 50
    return np.zeros((16, 16), dtype=np.uint8)


def compress_image(img):
    # Get the shape of the image
    height, width = img.shape

    # Initialize the compressed image with zeros and type uint8
    compressed_img = np.zeros((height, width), dtype=np.uint8)

    # Loop through the image in 16x16 blocks
    for i in range(0, height - 16, 16):
        for j in range(0, width - 16, 16):
            # Get the current block
            bloc1 = img[i:(i + 16), j:(j + 16)]

            # Calculate the coordinates of the centered 64x64 block
            b1cor = [i + 64, (i + 16) + 64, j + 64, (j + 16) + 64]

            # Set the step size to 64
            step = 64

            # Calculate the residual block
            residual_block = calculate_deco_residual(step, b1cor, bloc1)

            # Update the compressed image with the residual block
            compressed_img[i:i + 16, j:j + 16] = residual_block
    return compressed_img


imageN = compress_image(gray_frame1)
cv2.imshow("window deco frame", imageN)
cv2.waitKey(0)
cv2.destroyAllWindows()
