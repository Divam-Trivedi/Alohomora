import numpy as np
import matplotlib.pyplot as plt
import cv2

# Gaussian 1D function with derivatives
def gauss1d(sigma, mean, x, order):
    x = x - mean
    variance = sigma ** 2
    denom = 2 * variance
    g = np.exp(-(x ** 2) / denom) / (np.pi * denom) ** 0.5

    if order == 1:  ## First Derivative
        g = -g * (x / variance)
    elif order == 2:  ## Second Derivative
        g = g * ((x ** 2 - variance) / (variance ** 2))

    return g

# Normalize filter
def normalise(f):
    f = f - np.mean(f)
    return f / np.sum(np.abs(f))

# Create a filter
def makefilter(scale, phasex, phasey, pts, sup):
    gx = gauss1d(3 * scale, 0, pts[0, :], phasex)
    gy = gauss1d(scale, 0, pts[1, :], phasey)
    f = normalise(gx * gy)
    return f.reshape(sup, sup)

# Generate LM filter bank
def makeLMfilters(filt_type = "LMS"):
    SUP = 49  # Support of the largest filter
    if filt_type == "LMS":
        SCALEX = np.sqrt(2) ** np.arange(0, 3)
        SCALES = np.sqrt(2) ** np.arange(0, 4)
    elif filt_type == "LML":
        SCALEX = np.sqrt(2) ** np.arange(1, 4)
        SCALES = np.sqrt(2) ** np.arange(1, 5)
    NORIENT = 6  # Number of orientations

    NROTINV = 12
    NBAR = len(SCALEX) * NORIENT
    NEDGE = len(SCALEX) * NORIENT
    NF = NBAR + NEDGE + NROTINV

    F = np.zeros((SUP, SUP, NF))
    hsup = (SUP - 1) // 2
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1), np.arange(hsup, -hsup - 1, -1))
    orgpts = np.vstack((x.flatten(), y.flatten()))

    count = 0
    for scale in SCALEX:
        for orient in range(NORIENT):
            angle = np.pi * orient / NORIENT  # Not 2pi due to symmetry
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)

            F[:, :, count] = makefilter(scale, 0, 1, rotpts, SUP)
            # F[:, :, count + NEDGE] = makefilter(scale, 0, 2, rotpts, SUP)
            count += 1
        for orient in range(NORIENT):
            angle = np.pi * orient / NORIENT  # Not 2pi due to symmetry
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)

            # F[:, :, count] = makefilter(scale, 0, 1, rotpts, SUP)
            F[:, :, count] = makefilter(scale, 0, 2, rotpts, SUP)
            count += 1

    count = NBAR + NEDGE
    for scale in SCALES:
        # F[:, :, count] = normalise(gaussian2d(SUP, scale))
        F[:, :, count] = normalise(log_filter(SUP, scale))
        # F[:, :, count + 1] = normalise(log_filter(SUP, 3 * scale))
        count += 1
    for scale in SCALES:
        F[:, :, count] = normalise(log_filter(SUP, 3 * scale))
        count += 1
    for scale in SCALES:
        F[:, :, count] = normalise(gaussian2d(SUP, scale))
        count += 1

    return F

# 2D Gaussian filter
def gaussian2d(sup, sigma):
    hsup = (sup - 1) // 2
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1), np.arange(-hsup, hsup + 1))
    gauss = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return gauss / np.sum(gauss)

# Laplacian of Gaussian (LoG) filter
def log_filter(sup, sigma):
    hsup = (sup - 1) // 2
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1), np.arange(-hsup, hsup + 1))
    r2 = x ** 2 + y ** 2
    log = (r2 - 2 * sigma ** 2) * np.exp(-r2 / (2 * sigma ** 2))
    return log / np.sum(np.abs(log))

# Visualize LM filters
def visualize_filters(F,save_path, display=False):


    nf = F.shape[2]
    ncols = 12
    nrows = int(np.ceil(nf / ncols))

    plt.figure(figsize=(12, 4))

    for i in range(nf):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(F[:, :, i], cmap='gray')
        plt.axis('off')
        # plt.title(f'Filter {i + 1}')
    plt.tight_layout()

    # Save the filter bank as an image
    plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close()

def visualize_filters_opencv(F, save_path, display=False):
    nf = F.shape[2]
    ncols = 6
    nrows = int(np.ceil(nf / ncols))

    h, w = F.shape[0], F.shape[1]

    canvas_height = nrows * h
    canvas_width = ncols * w
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    F_min = np.min(F)
    F_max = np.max(F)
    F_normalized = 255 * (F - F_min) / (F_max - F_min)
    F_normalized = F_normalized.astype(np.uint8)

    for i in range(nf):
        row = i // ncols
        col = i % ncols
        filter_image = F[:, :, i]
        
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = filter_image

    cv2.imwrite(save_path, canvas)

    if display:
        cv2.imshow("Filter Bank", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()