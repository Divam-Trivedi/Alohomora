from sklearn.cluster import KMeans
import cv2
import numpy as np

def generate_color_map(image_paths, num_clusters=16, color_space='RGB', image_save_path="ColorMap"):

    color_maps = []

    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        image = cv2.imread(image_path)

        # Convert the image to the desired color space
        if color_space == 'YCbCr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'Lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        # 'RGB' is the default case (no conversion needed)

        h, w, c = image.shape
        pixels = image.reshape(-1, 1).astype(np.float32)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        clustered = kmeans.labels_.reshape(h, w, c)

        color_maps.append(clustered)

        normalized_map = (clustered / num_clusters * 255).astype(np.uint8)
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/ColorMap_{image_name}.png"
        cv2.imwrite(filename, normalized_map)

    return color_maps


def generate_color_gradient_with_masks(image_paths, color_maps, num_bins, masks, image_save_path="Cg"):
    color_gradients = []

    for idx, (image_path, color_map) in enumerate(zip(image_paths, color_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w, c = color_map.shape
        chi_sqr_dist = np.zeros((h, w, c), dtype=np.float32)

        for i in range(num_bins):
            tmp = (color_map == i).astype(np.float32)

            for j in range(0, len(masks), 2):
                left_mask = np.array(masks[j])
                right_mask = np.array(masks[j + 1])

                g_i = cv2.filter2D(tmp, -1, left_mask)
                h_i = cv2.filter2D(tmp, -1, right_mask)
                
                # Compute Chi-square gradient
                numerator = (g_i - h_i) ** 2
                denominator = g_i + h_i + 1e-6
                chi_sqr_dist += numerator / denominator

        color_gradients.append(chi_sqr_dist)

        normalized_gradient = (chi_sqr_dist / chi_sqr_dist.max() * 255).astype(np.uint8)

        # Apply a colormap for visualization if needed
        # color_gradient = cv2.applyColorMap(normalized_gradient, cv2.COLORMAP_JET)

        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/Cg_{image_name}.png"
        cv2.imwrite(filename, normalized_gradient)

    return color_gradients
