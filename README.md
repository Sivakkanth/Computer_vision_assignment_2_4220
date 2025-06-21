# Computer_vision_assignment_2_4220 - EC7212

**Registration Number**: EG/2020/4220
**Date**: 2025-06-16
**Course**: EC7212 – Computer Vision and Image Processing
**Assignment**: Take Home Assignment 2

---

## Assignment Questions & Descriptions

### Question 1: Otsu’s Thresholding on Noisy Image

**Task**:
- Create an artificial grayscale image with:
  - 1 background region
  - 2 distinct foreground objects (rectangle and ellipse)
  - Exactly 3 pixel intensity levels
- Add **Gaussian noise** (mean = 0, std = 50)
- Apply **Otsu’s method** to automatically threshold and segment the noisy image.

**Output**:
- Original synthetic image
- Noisy image
- Otsu thresholded binary image

**Script**: `question_1.py`  
**Libraries**: `OpenCV`, `NumPy`

---

### Question 2: Region Growing Segmentation

**Task**:
- Load a grayscale image (`image.jpg`)
- Define a set of **seed points**
- Grow a region from the seeds by including neighboring pixels within a **threshold intensity range**
- Use **8-connectivity**
- Display the segmented mask

**Output**:
- Original image
- Segmented region mask

**Script**: `question_2.py`  
**Libraries**: `OpenCV`, `NumPy`, `collections`

---

## 🚀 How to Run

### 1. Install Required Libraries

```bash
pip install opencv-python numpy