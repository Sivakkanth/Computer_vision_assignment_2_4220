########################################################################
#   Take Home Assignment 2
#   Question 1
#   Reg No: EG/2020/4220
#   Date: 2025-06-16
########################################################################

# import all necessary libraries
import numpy as np
import cv2

def get_image(width, height):
    """
    Generate a synthetic grayscale image with 3 intensity levels:
    - Background: 255 (white)
    - Object 1 (rectangle): intensity 100
    - Object 2 (ellipse): intensity 40
    """
    # Create a white background image
    image = np.full((height, width), 180, dtype=np.uint8)

    # Draw a rectangle (Object 1) with intensity 100
    cv2.rectangle(image, (int(width * 0.1), int(height * 0.3)),
                  (int(width * 0.4), int(height * 0.7)), 100, -1)

    # Draw an ellipse (Object 2) with intensity 40
    cv2.ellipse(image, (int(width * 0.75), int(height * 0.5)),
                (int(width * 0.1), int(height * 0.2)), 0, 0, 360, 40, -1)
    return image

def inject_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Inject Gaussian noise into the image.
    
    Parameters:
    - image: Input image as a numpy array.
    - mean: Mean of the Gaussian noise.
    - sigma: Standard deviation of the Gaussian noise.
    
    Returns:
    - Noisy image as a numpy array.
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_otsu_method(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding method to the image.
    
    Parameters:
    - image: Input image as a numpy array.
    
    Returns:
    - Thresholded binary image as a numpy array.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

if __name__ == "__main__":
    # Define image dimensions
    width, height = 640, 480
    
    # Generate the synthetic image
    image = get_image(width, height)
    
    # Inject Gaussian noise into the image
    noisy_image = inject_gaussian_noise(image)
    
    # Apply Otsu's method to the noisy image
    binary_image = apply_otsu_method(noisy_image)
    
    # Display the images (optional, for debugging purposes)
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image", noisy_image)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()