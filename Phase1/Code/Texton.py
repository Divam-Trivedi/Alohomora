import cv2
import numpy as np
from sklearn.cluster import KMeans

def generate_texton_map(image_paths, filter_bank):
    texton_maps = []

    for image_path in image_paths:
        image = cv2.imread(image_path)

        if len(image.shape) == 2:
            grayscale_image = image
        else:
            # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale_image = image
        
        filtered_responses = []
        
        for filter_ in filter_bank:
            response = cv2.filter2D(grayscale_image, -1, filter_)
            filtered_responses.append(response)
        
        filtered_responses = np.stack(filtered_responses, axis=-1)
        texton_map = np.max(filtered_responses, axis=-1)
        texton_maps.append(texton_map)
    
    return texton_maps

def generate_texture_ids(image_paths, texton_maps, num_clusters, save_image_path="TextonMap"):
    texture_ids = []
    
    for idx, (image_path, texton_map) in enumerate(zip(image_paths, texton_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w, c = texton_map.shape
        reshaped_map = texton_map.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(reshaped_map)
        
        clustered_map = kmeans.labels_.reshape(h, w, c)

        image_name = image_path.split('/')[-1].split('.')[0]
        filename = save_image_path + f"/TextonMap_{image_name}.png"
        cv2.imwrite(filename, (clustered_map * (255 // num_clusters)).astype(np.uint8))

        texture_ids.append(clustered_map)
    return texture_ids


def generate_texton_gradient(image_paths, texton_maps, num_bins, masks, image_save_path):
    texton_gradients = []

    for idx, (image_path, texton_map) in enumerate(zip(image_paths, texton_maps)):
        print(f"Processing image {idx + 1}/{len(image_paths)}")
        h, w, c = texton_map.shape
        chi_sqr_dist = np.zeros((h, w, c), dtype=np.float32)

        for i in range(num_bins):
            tmp = np.array((texton_map == i).astype(np.float32))

            for j in range(0, len(masks), 2):
                left_mask = np.array(masks[j])
                right_mask = np.array(masks[j + 1])

                g_i = cv2.filter2D(tmp, -1, left_mask)
                h_i = cv2.filter2D(tmp, -1, right_mask)
                
                # Compute Chi-square gradient
                numerator = (g_i - h_i) ** 2
                denominator = g_i + h_i + 1e-6
                chi_sqr_dist += numerator / denominator

        texton_gradients.append(chi_sqr_dist)

        gradient_image = (chi_sqr_dist / chi_sqr_dist.max() * 255).astype(np.uint8)
        image_name = image_path.split('/')[-1].split('.')[0]
        filename = image_save_path + f"/Tg_{image_name}.png"
        # cv2.imwrite(filename, (clustered_map * (255 // num_clusters)).astype(np.uint8))
        cv2.imwrite(filename, gradient_image)

    return texton_gradients

