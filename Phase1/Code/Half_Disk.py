import numpy as np
import cv2
import matplotlib.pyplot as plt

radius = [4, 10]
orientations = np.linspace(0, 2*np.pi, 8, endpoint=False)

def rotateImage(image, angle, clock_wise = True):
    rows, cols = image.shape[0], image.shape[1]
    # angle = np.rad2deg(angle)
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 2 * (int(clock_wise) - 0.5))
    # fixing the rotation matrix to be n=in the (0 ,0)
    matrix[0, 2] += (matrix[0, 0] + matrix[0, 1] - 1) / 2
    matrix[1, 2] += (matrix[1, 0] + matrix[1, 1] - 1) / 2
    result = cv2.warpAffine(image, matrix, (cols, rows))
    return result


def generate_half_disk_masks(radius, orientations):
    masks = []
    for r in radius:
        base_mask = np.zeros((2*r + 1, 2*r + 1), dtype=np.uint8)
        for i in range(0, 2*r + 1):
            for j in range(0, 2*r + 1):
                dist = round(np.sqrt((i - r) ** 2 + (j - r) ** 2))
                # print(i, j, dist)
                # dist = np.floor(dist)
                if dist <= r:
                    base_mask[i, j] = 255
        for i in range(0, 2*r + 1):
            for j in range((2*r + 1) // 2, 2*r + 1):
                base_mask[i, j] = 0
        for i in orientations:
            rotated = rotateImage(base_mask, i)
            # rotated = base_mask
            m = []
            for x,a in enumerate(rotated):
                m.append([])
                for y,b in enumerate(a):
                    if b < 255 // 2:
                        m[x].append(0)
                    else:
                        m[x].append(255)
            masks.append(m)
    return masks

def visualize_filters(F,save_path, display=False):
    plt.figure(figsize=(20, 15))
    for i, hdisks in enumerate(F):
        plt.subplot(6, 8, i + 1)
        plt.imshow(hdisks, cmap='gray')
        plt.axis('off')
        # plt.title(f'Filter {i + 1}')
    plt.tight_layout()
    plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close()