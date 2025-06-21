########################################################################
#   Take Home Assignment 2
#   Question 2
#   Reg No : EG/2020/4220
#   Date   : 2025-06-16
########################################################################

# Import required libraries
import cv2
import numpy as np

def show_segmentation(mask):
    """
    Display the intermediate mask during segmentation.
    """
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)  # Show briefly for animation effect

def region_growing(image, seed_points, threshold_range):
    """
    Perform region growing segmentation.

    Parameters:
        image (ndarray): Input grayscale image.
        seed_points (list): List of (x, y) seed coordinates.
        threshold_range (int): Max allowed intensity difference from seed.

    Returns:
        mask (ndarray): Binary segmented mask.
    """
    
    # Initialize segmentation mask (same size as image)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Initialize queue with seed points
    queue = list(seed_points)

    iteration = 0

    while queue:
        iteration += 1
        current_point = queue.pop(0)
        x, y = current_point

        # Get intensity at current point
        current_value = image[y, x]
        mask[y, x] = 255  # Mark as visited

        # Visualize intermediate steps every 10 iterations
        if iteration % 10 == 0:
            show_segmentation(mask)

        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy

                if dx == 0 and dy == 0:
                    continue  # skip center

                # Check bounds
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                    neighbor_value = image[ny, nx]

                    if mask[ny, nx] == 0 and abs(int(neighbor_value) - int(current_value)) <= threshold_range:
                        queue.append((nx, ny))
                        mask[ny, nx] = 255  # Mark neighbor as visited

    return mask

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Load the grayscale input image
    image = cv2.imread('images/image.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Image not found.")

    # Define seed points (adjust based on image dimensions)
    seed_points = [(120, 180), (200, 200), (100, 220)]

    # Define threshold range for pixel similarity
    threshold_range = 10

    # Perform region growing segmentation
    segmented_image = region_growing(image, seed_points, threshold_range)

    # Close any previous windows
    cv2.destroyAllWindows()

    # Show final result
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()