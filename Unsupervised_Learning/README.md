# Customer Segmentation Analysis

This notebook performs customer segmentation using various unsupervised machine learning techniques, including K-Means clustering, Hierarchical Clustering, and dimensionality reduction methods like PCA and t-SNE. The goal is to identify distinct customer groups based on their purchasing behavior and demographic information, providing insights for targeted marketing strategies.

## Table of Contents

1.  [Data Exploration & Preprocessing](#data-exploration--preprocessing)
2.  [K-Means Clustering](#k-means-clustering)
3.  [Hierarchical Clustering](#hierarchical-clustering)
4.  [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
5.  [Dimensionality Reduction (t-SNE)](#dimensionality-reduction-t-sne)
6.  [Comparative Analysis](#comparative-analysis)

## 1. Data Exploration & Preprocessing

-   Loaded the `Mall_Customers.csv` dataset.
-   Displayed basic statistics and checked for missing values.
-   Visualized data distributions using histograms for 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'.
-   Explored relationships between features using pair plots, colored by 'Gender'.
-   **Key Observations:**
    -   Age distribution is skewed towards younger adults.
    -   Annual Income is somewhat normally distributed.
    -   Spending Score shows multiple peaks, suggesting different spending behaviors.
    -   The pair plot of 'Annual Income' vs. 'Spending Score' reveals clear potential clusters, making these features highly relevant for segmentation.

## 2. K-Means Clustering

-   Applied `StandardScaler` to numerical features for proper distance calculation.
-   Used the Elbow method to determine the optimal number of clusters, suggesting `k=5`.
-   Performed K-Means clustering with 5 clusters.
-   Visualized the clusters using a scatter plot of 'Annual Income' vs. 'Spending Score', colored by cluster.
-   **Cluster Analysis (5 clusters):**
    -   **Cluster 0 (High Income, High Spending):** Valuable customers.
    -   **Cluster 1 (High Income, Low Spending):** Potentially budget-conscious despite high income.
    -   **Cluster 2 (Low Income, Low Spending):** Budget-constrained segment.
    -   **Cluster 3 (Low Income, High Spending):** Impulsive buyers or younger customers.
    -   **Cluster 4 (Mid-Range Income, Mid-Range Spending):** Average customer profile.

## 3. Hierarchical Clustering

-   Implemented Agglomerative Clustering with 5 clusters, using different linkage methods: 'ward', 'average', and 'complete'.
-   Calculated silhouette scores for each linkage method to evaluate cluster quality.
-   Visualized the clustering process using a dendrogram with 'average' linkage.
-   **Key Findings:**
    -   'Average' linkage yielded the highest silhouette score (0.3074), indicating better-separated clusters compared to 'ward' (0.2869) and 'complete' (0.2433) for this dataset and `k=5`.
    -   Hierarchical clustering provides a tree-like structure, offering a different perspective on cluster formation compared to K-Means.

## 4. Dimensionality Reduction (PCA)

-   Performed Principal Component Analysis (PCA) on the scaled features.
-   Plotted the explained variance ratio by principal component.
-   **Key Findings:**
    -   The first two principal components together explain approximately **{{pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.4f}}** of the total variance.
    -   PCA helps in understanding the global variance and can be used to visualize clusters in a lower-dimensional space, though it might not always show clear separation for non-linear structures.

## 5. Dimensionality Reduction (t-SNE)

-   Applied t-Distributed Stochastic Neighbor Embedding (t-SNE) to the scaled features, reducing to 2 dimensions.
-   Plotted the t-SNE reduced data.
-   **Key Findings:**
    -   t-SNE is a non-linear technique that excels at preserving local neighborhoods, often resulting in **clearer visual separation of clusters** compared to PCA.
    -   While distances between far-apart clusters in a t-SNE plot might not be meaningful, the groupings themselves are usually more visually distinct.

## 6. Comparative Analysis

-   **Silhouette Scores:**
    -   K-Means: 0.3041
    -   Hierarchical Clustering ('average' linkage): 0.3074 (Highest)
-   **Conclusion:** Hierarchical Clustering with 'average' linkage showed a slightly higher silhouette score, suggesting marginally better-defined clusters than K-Means for this specific scenario. However, the choice of clustering algorithm often depends on the specific goals and interpretability requirements.
-   **Visualization:** t-SNE generally provides a clearer visual separation of clusters than PCA, especially for complex, non-linear data structures, by prioritizing local relationships.
