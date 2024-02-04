# Point Cloud Classification with Point Net

## 1. Project Summary

In this repository, we will implement the PointNet architecture for point clouds classification. This architecture was introduced in the paper titled "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation," by Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas at the Neural Information Processing Systems (NeurIPS) conference in 2016.

### 1.1. Dataset

We use the Princeton 3D Shapenets data of 10 classes (ModelNet10) which consists of daily objects such as chair, table, and plane. The link to the dataset can be found [here](https://3dshapenets.cs.princeton.edu/).

```plaintext
Dataset: 3D Shapenets (ModelNet10)
- Size: 4.9k samples
- Classes: 10
- Download Link: [3D Shapenets Dataset](https://3dshapenets.cs.princeton.edu/)
```

### 1.2. Architecture

PointNet is a neural network architecture designed for processing and analyzing point clouds. It employs a shared multi-layer perceptron (MLP) to process each point independently to capture local features. It also introduces a symmetric function to aggregate information from all points, ensuring the network is permutation-invariant, meaning it produces the same output regardless of the order of input points. Figure 1 displays the PointNet architecture diagram.

<p align="center">
  <img src="assets/pointnet.jpg" width="600" title="hover text">
</p>

PointNet is effective in recognizing and classifying objects in 3D space. It is used for tasks like segmentation and classification, as shown by Figure 2.

<p align="center">
  <img src="assets/teaser.jpg" width="500" title="hover text">
</p>


## 2. Usage

### 2.1. Installation

1. To clone the repository:

    ```bash
    git clone https://github.com/your-username/pointcloud-classification.git
    cd pointcloud-classification
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 2.2.Training

Provide details on how to train the PointNet model on your dataset. Include hyperparameters, training script usage, and any additional information.

```bash
python train.py --config config.yaml
```

### 2.3. Evaluation

Explain how to evaluate the trained model on a test set.

```bash
python evaluate.py --config config.yaml
```

### 2.4. Results

Present and discuss the results obtained from your experiments. Include metrics, visualizations, and any other relevant information.

## References

List any papers, articles, or resources related to PointNet and point cloud classification.

1. Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.