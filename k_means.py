import numpy as np

def k_means(X, init_c, num_classes, n_iters=50):
    # Set up for assignments and centers
    n, d = X.shape
    last_c = np.copy(init_c)
    centers = np.copy(init_c)
    assignments = np.zeros(n, dtype=int)

    # Loop through the epochs until convergence is reached
    for epoch in range(n_iters):
        # Step 1: Assign data points to the nearest cluster
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)
        assignments = np.argmin(distances, axis=1)

        # Step 2: Update cluster centers
        counts = np.bincount(assignments, minlength=num_classes)

        # Sum across each centroid
        dim_sums = np.zeros((num_classes, d))
        np.add.at(dim_sums, assignments, X)

        centers = dim_sums / counts[:, np.newaxis]

        # Convergence
        if np.array_equal(last_c, centers):
            print("Iteration at breakage:", epoch)
            break

        last_c = np.copy(centers)

    return centers, partition_points(X, assignments, num_classes)

# Helper function to partition points and to calculate cost
def partition_points(X, assignments, num_classes):
    points = [X[assignments == k].tolist() for k in range(num_classes)]
    return points

# Helper method for initialization
def kmeans_plusplus_initialization(data, k):
    # Step 1: Randomly choose the first centroid
    centroids = [data[np.random.choice(len(data))]]

    # Step 2: Choose the next centroids
    for i in range(1, k):
        # Calculate the squared distances from each data point to the nearest existing centroid
        distances = [
            min(np.sum((point - centroid) ** 2) for centroid in centroids)
            for point in data
        ]
        
        # Choose the next centroid with probability proportional to squared distance
        total_distance = sum(distances)
        probabilities = [distance / total_distance for distance in distances]
        chosen_index = np.random.choice(len(data), p=probabilities)
        next_centroid = data[chosen_index]
        
        # Add the next centroid to the list
        centroids.append(next_centroid)
    
    return np.array(centroids)
