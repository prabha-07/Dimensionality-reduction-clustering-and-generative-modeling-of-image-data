# Dimensionality Reduction, Clustering, and Generative Modeling of Image Data

## Overview

This project demonstrates comprehensive techniques for analyzing image data through dimensionality reduction, clustering, and generative modeling. The work focuses on the **Caltech image dataset** containing 6 classes: sunflower, panda, kangaroo, flamingo, elephant, and butterfly.

## Dataset

- **Total Images**: 221 grayscale images
- **Image Size**: 128×128 pixels (16,384 features per image)
- **Classes**: 6 categories
  - Butterfly
  - Elephant
  - Flamingo
  - Kangaroo
  - Panda
  - Sunflower

## Project Structure

### 1. Dimensionality Reduction

#### A. Principal Component Analysis (PCA)
- **95% Variance Retention**: Requires 133 components (reduction from 16,384 to 133)
- **2D PCA**: Explains 39.44% of total variance
- **Reconstruction**: Visual comparison of original vs. PCA-reconstructed images

#### B. Manifold Learning Techniques
Implemented and compared four dimensionality reduction methods:

1. **PCA (2D)**
   - Linear dimensionality reduction
   - Preserves global structure

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
   - Non-linear dimensionality reduction
   - Preserves local neighborhood structure
   - Perplexity: 30

3. **LLE (Locally Linear Embedding)**
   - Non-linear manifold learning
   - Preserves local linear relationships
   - Neighbors: 12

4. **UMAP (Uniform Manifold Approximation and Projection)**
   - Non-linear dimensionality reduction
   - Balances local and global structure preservation

### 2. Clustering Analysis

#### A. K-Means Clustering
- **Preprocessing**: PCA reduction to 133 components (95% variance)
- **Optimal K Selection**: Elbow method (tested k=2 to 14)
- **Evaluation**: Clustering accuracy using Hungarian algorithm for label alignment
- **Final Configuration**: K=8 clusters

**Key Steps:**
1. Apply PCA to reduce dimensionality
2. Use elbow method to determine optimal number of clusters
3. Fit K-Means on PCA-reduced data
4. Evaluate using confusion matrix and clustering accuracy

#### B. Gaussian Mixture Model (GMM)
- **Preprocessing**: PCA reduction to 133 components
- **Model Selection**: Bayesian Information Criterion (BIC) for optimal component selection
- **Evaluation**: Clustering accuracy with Hungarian algorithm
- **Final Configuration**: 8 components

**Key Steps:**
1. Apply PCA for dimensionality reduction
2. Use BIC to select optimal number of GMM components
3. Fit GMM on PCA-reduced data
4. Evaluate clustering performance

### 3. Feature Extraction

#### ResNet50 Feature Extraction
- Uses pre-trained ResNet50 model (ImageNet weights)
- Extracts deep features from images
- Enables transfer learning for better feature representation

### 4. Generative Modeling

#### Gaussian Mixture Model (GMM) for Image Generation
- Samples new images from learned GMM distribution
- Reconstructs images from sampled latent representations
- Demonstrates generative capabilities of probabilistic models

## Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computations
- **Scikit-learn**:
  - `PCA`: Principal Component Analysis
  - `TSNE`: t-SNE manifold learning
  - `LocallyLinearEmbedding`: LLE implementation
  - `KMeans`: K-means clustering
  - `GaussianMixture`: GMM clustering and generation
  - `silhouette_score`: Clustering evaluation
- **TensorFlow/Keras**:
  - Image loading and preprocessing
  - ResNet50 for feature extraction
- **UMAP**: Uniform Manifold Approximation and Projection
- **Matplotlib & Seaborn**: Visualization
- **SciPy**: Optimization algorithms (Hungarian method)

## Installation

```bash
# Clone the repository
git clone https://github.com/prabha-07/Dimensionality-reduction-clustering-and-generative-modeling-of-image-data.git

# Install required packages
pip install numpy scikit-learn tensorflow matplotlib seaborn umap-learn scipy tqdm
```

## Usage

1. **Prepare your image dataset**: Organize images in a folder with filenames containing class names
2. **Update the base path**: Modify `base_path` variable to point to your image directory
3. **Run the notebook**: Execute cells sequentially to:
   - Load and preprocess images
   - Apply dimensionality reduction techniques
   - Perform clustering analysis
   - Generate new images using GMM

### Example Image Loading Structure
```
caltech/
├── butterfly_001.jpg
├── butterfly_002.jpg
├── elephant_001.jpg
├── flamingo_001.jpg
└── ...
```

## Key Results

### Dimensionality Reduction
- **PCA 95% variance**: 133 components (99.2% reduction)
- **PCA 2D variance**: 39.44%
- **Visual embeddings**: 2D projections for all four methods

### Clustering Performance
- **K-Means**: Evaluated using confusion matrix and clustering accuracy
- **GMM**: BIC-optimized component selection
- Both methods applied on PCA-reduced feature space

### Generative Modeling
- Successfully samples and reconstructs images from GMM
- Demonstrates probabilistic image generation capabilities

## Methodology Highlights

1. **Preprocessing**: Grayscale conversion, resizing to 128×128, normalization
2. **Dimensionality Reduction**: Multiple techniques for comparison and visualization
3. **Clustering**: Both partition-based (K-Means) and probabilistic (GMM) approaches
4. **Evaluation**: Hungarian algorithm for optimal label assignment in clustering
5. **Feature Engineering**: Deep learning features via ResNet50

## Visualizations

- Original vs. PCA-reconstructed images
- 2D embeddings (PCA, t-SNE, LLE, UMAP) with class-colored scatter plots
- Elbow curve for K-Means
- BIC curve for GMM
- Confusion matrices for clustering evaluation
- Generated images from GMM

## Key Contributions

- Comprehensive comparison of dimensionality reduction techniques
- Application of both partition-based and probabilistic clustering
- Integration of deep learning features (ResNet50)
- Generative modeling demonstration using GMM
- Rigorous evaluation using Hungarian algorithm for clustering accuracy

## Future Enhancements

- Variational Autoencoders (VAE) for generative modeling
- Convolutional Autoencoders for better reconstruction
- Deep clustering methods
- Evaluation metrics: Adjusted Rand Index, Normalized Mutual Information
- Real-time image generation interface

## License

This project is open source and available for educational purposes.

## Author

**prabha-07**

---

*For questions or contributions, please open an issue or submit a pull request.*

