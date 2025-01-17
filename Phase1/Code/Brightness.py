from sklearn.cluster import KMeans
import cv2
import numpy as np

def generate_brightness_map(image_paths, num_clusters=16, image_save_path="BrightnessMap"):
    brightness_maps = []

    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        image = cv2.imread(image_path)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        
        pixels = gray_image.reshape(-1, 1).astype(np.float32)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        clustered = kmeans.labels_.reshape(h, w)
        
        brightness_maps.append(clustered)

        normalized_map = (clustered / num_clusters * 255).astype(np.uint8)
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/BrightnessMap_{image_name}.png"
        cv2.imwrite(filename, normalized_map)

    return brightness_maps



def generate_brightness_gradient_with_masks(image_paths, brightness_maps, num_bins, masks, image_save_path):
    brightness_gradients = []
    for idx, (image_path, brightness_map) in enumerate(zip(image_paths, brightness_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w = brightness_map.shape
        chi_sqr_dist = np.zeros((h, w), dtype=np.float32)

        for i in range(num_bins):
            tmp = (brightness_map == i).astype(np.float32)

            for j in range(0, len(masks), 2):
                left_mask = np.array(masks[j])
                right_mask = np.array(masks[j + 1])

                g_i = cv2.filter2D(tmp, -1, left_mask)
                h_i = cv2.filter2D(tmp, -1, right_mask)
                
                # Compute Chi-square gradient
                numerator = (g_i - h_i) ** 2
                denominator = g_i + h_i + 1e-6
                chi_sqr_dist += numerator / denominator
                
        brightness_gradients.append(chi_sqr_dist)

        normalized_gradient = (chi_sqr_dist / chi_sqr_dist.max() * 255).astype(np.uint8)

        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/Bg_{image_name}.png"
        cv2.imwrite(filename, normalized_gradient)

    return brightness_gradients