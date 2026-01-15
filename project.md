# Image Denoising via Convolutional Autoencoder on Fashion MNIST

## 1. Project Scope and Objectives

This project implements an end-to-end deep learning pipeline for **image denoising** using a **Convolutional Autoencoder (CAE)** architecture. The system addresses the problem of reconstructing clean images from noisy observations, a fundamental task in computer vision with applications in medical imaging, satellite imagery, and photography restoration.

### 1.1 Primary Objectives

The system is designed to accomplish the following research and engineering objectives:

1. **Data Acquisition and Preprocessing**: Load and preprocess the Fashion MNIST dataset, applying controlled Gaussian noise degradation to simulate realistic image corruption scenarios.

2. **Model Architecture Design**: Implement a symmetric convolutional autoencoder with appropriate downsampling and upsampling mechanisms to learn efficient latent representations while preserving spatial structure.

3. **Training Optimization**: Train the network to minimize reconstruction error between denoised outputs and ground-truth clean images, employing modern optimization techniques including Automatic Mixed Precision (AMP) and model compilation.

4. **Hardware Utilization**: Maximize computational efficiency on NVIDIA RTX 4090 GPU through optimized data pipeline configuration, mixed-precision training, and GPU-specific acceleration strategies.

### 1.2 Research Context

Image denoising remains a critical preprocessing step in numerous computer vision applications. This implementation leverages recent advances in deep learning, particularly the autoencoder paradigm, to learn robust feature representations that distinguish signal from noise in the latent space.

---

## 2. Development Environment Configuration

This project employs `uv` as the dependency management system, providing deterministic builds and efficient package resolution. The `pyproject.toml` configuration file below specifies all required dependencies for computer vision, scientific computation, and CUDA-accelerated operations.

**Configuration Rationale**: The dependency set has been curated specifically for vision-based deep learning tasks, excluding NLP-related libraries while prioritizing PyTorch with CUDA 12.1 support, torchvision for dataset handling, and torchmetrics for quantitative evaluation.

### `pyproject.toml`

Copia questo contenuto nel file di configurazione:

```toml
[project]
name = "fashion-mnist-denoiser"
version = "0.1.0"
description = "Autoencoder pipeline for image denoising on Fashion MNIST"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "pillow>=10.0.0",
    "pandas>=2.1.0",
    "numpy>=1.26.0",
    "tqdm>=4.66.0",
    "matplotlib>=3.8.0",
    "scikit-learn>=1.3.0",
    "torchmetrics>=1.8.2",  # Per calcolare PSNR e SSIM
    "seaborn>=0.13.2",      # Visualizzazione loss/metriche
    "jupyter",
    "ipywidgets",
    "colorama>=0.4.6",
]

[dependency-groups]
dev = [
    "ipykernel>=7.1.0",
    "pytest>=8.0.0",
]

[tool.uv]
# Configurazione specifica per CUDA 12.1 (Ottimizzata per RTX 4090)
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

# Mapping dei pacchetti sull'indice CUDA
[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }

```

---

## 3. Dataset Specification: Fashion MNIST

The Fashion MNIST dataset comprises 70,000 grayscale images of fashion items, distributed across 10 distinct categories. Each sample is a 28×28 pixel single-channel image.

### 3.1 Data Organization

The dataset structure in `data/raw/` contains:
- **Training set**: 60,000 images (`train-images-idx3-ubyte`, `fashion-mnist_train.csv`)
- **Test set**: 10,000 images (`t10k-images-idx3-ubyte`, `fashion-mnist_test.csv`)

### 3.2 Network Input/Output Specification

* **Network Input (X_noisy)**: 28×28 grayscale image corrupted with additive Gaussian noise: X_noisy = X_clean + ε, where ε ~ N(0, σ²)
* **Network Target (X_clean)**: Original 28×28 grayscale image without noise corruption
* **Label Classes** (provided for context, not utilized in unsupervised reconstruction loss):
  - 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
  - 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

### 3.3 Data Preprocessing Requirements

* **Normalization**: All pixel intensities are normalized to the range [0, 1] using min-max scaling
* **Tensor Shape**: Input tensors follow PyTorch convention: (Batch, Channels, Height, Width) = (N, 1, 28, 28)
* **Noise Injection**: Gaussian noise with σ = 0.3 is applied during training; noise is clipped to maintain valid intensity ranges

---

## 4. Network Architecture: Convolutional Autoencoder

The architecture employs a fully convolutional design paradigm, eschewing fully-connected layers to preserve spatial locality and translation equivariance—critical properties for image reconstruction tasks.

### 4.1 Encoder Module

**Objective**: Compress the input image into a low-dimensional latent representation (bottleneck) that captures salient features while discarding noise.

**Design Specifications**:
* Convolutional layers with `kernel_size=3`, `padding=1` to maintain spatial dimensions
* Spatial downsampling via `stride=2` or `nn.MaxPool2d(2)` for hierarchical feature extraction
* **Batch Normalization** (`nn.BatchNorm2d`) applied post-convolution to reduce internal covariate shift and accelerate convergence
* Non-linear activation: Rectified Linear Unit (`nn.ReLU`) for gradient flow optimization

### 4.2 Decoder Module

**Objective**: Reconstruct the clean image from the compressed latent representation through learned upsampling.

**Design Specifications**:
* Transposed convolutions (`nn.ConvTranspose2d`) for learnable upsampling, superior to fixed interpolation methods
* **Batch Normalization** applied to all layers except the final output layer to prevent gradient saturation
* Output activation: `nn.Sigmoid` to constrain pixel values to [0, 1] matching the normalized target distribution

### 4.3 Objective Function

**Loss Formulation**: Mean Squared Error (MSE) between reconstructed output X̂ and ground-truth clean image X:

L(θ) = (1/N) Σᵢ ||X̂ᵢ - Xᵢ||²

where θ represents the learnable parameters of the autoencoder network.

---

## 5. Hardware Optimization Strategy (NVIDIA RTX 4090)

To fully leverage the computational capabilities of the NVIDIA RTX 4090 (Ada Lovelace architecture with 16,384 CUDA cores and 512 Tensor Cores), the implementation incorporates the following optimization strategies:

### 5.1 Device Abstraction and Portability

**Implementation**: Utilize PyTorch's device-agnostic code pattern:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
This ensures portability across diverse hardware configurations while automatically detecting CUDA availability.

### 5.2 Graph Optimization via Model Compilation

**Implementation**: Apply `torch.compile(model)` (PyTorch 2.0+) to perform ahead-of-time graph optimization and kernel fusion, reducing Python overhead and maximizing Tensor Core utilization.

### 5.3 Automatic Mixed Precision (AMP) Training

**Rationale**: The RTX 4090's Tensor Cores deliver up to 2× throughput for FP16/BF16 operations compared to FP32.

**Implementation**: Leverage `torch.cuda.amp.autocast()` context manager and `GradScaler` to:
- Cast operations to lower precision (FP16) where numerically safe
- Maintain FP32 precision for gradient-sensitive operations
- Reduce VRAM consumption by ~50%
- Increase training throughput by 40-60%

### 5.4 Data Pipeline Optimization

**CPU-GPU Transfer Bottleneck Mitigation**:
- `num_workers=4-8`: Enable asynchronous data prefetching via multiprocessing
- `pin_memory=True`: Allocate tensors in page-locked memory for faster host-to-device transfer

**Expected Impact**: Ensures GPU compute remains saturated, eliminating I/O bottlenecks.

### 5.5 cuDNN Autotuning

**Implementation**: Enable `torch.backends.cudnn.benchmark = True`

**Rationale**: Given fixed input dimensions (28×28), cuDNN will autotune algorithms at runtime, selecting optimal convolution kernels for the specific hardware configuration.

---

## 6. Project Directory Structure

The project follows a hierarchical organization that separates concerns and facilitates reproducibility:

```
Image-Denoising/
├── data/
│   ├── raw/                          # Original Fashion MNIST files (version-controlled)
│   │   ├── fashion-mnist_train.csv   # Training set (CSV format)
│   │   ├── fashion-mnist_test.csv    # Test set (CSV format)
│   │   ├── train-images-idx3-ubyte   # Training images (binary format)
│   │   ├── train-labels-idx1-ubyte   # Training labels (binary format)
│   │   ├── t10k-images-idx3-ubyte    # Test images (binary format)
│   │   └── t10k-labels-idx1-ubyte    # Test labels (binary format)
│   ├── processed/                    # Preprocessed tensors (noise-augmented)
│   └── splits/                       # Train/Val/Test split indices
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Hyperparameters and configuration constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Custom Dataset class with noise injection
│   │   └── transforms.py             # AddGaussianNoise transformation
│   ├── models/
│   │   ├── __init__.py
│   │   └── autoencoder.py            # DenoisingAutoencoder architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop with AMP
│   │   └── losses.py                 # Loss function definitions
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                # PSNR, SSIM computation
│       └── visualization.py          # Plotting utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and noise characteristics
│   └── 02_results_analysis.ipynb     # Quantitative and qualitative evaluation
│
├── experiments/
│   ├── logs/                         # TensorBoard logs
│   ├── checkpoints/                  # Model checkpoints (.pth files)
│   └── outputs/                      # Generated visualizations
│
├── tests/
│   ├── test_dataset.py               # Unit tests for data pipeline
│   └── test_model.py                 # Architecture validation tests
│
├── scripts/
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Evaluation on test set
│   └── inference.py                  # Single-image denoising inference
│
├── pyproject.toml                    # Dependency specification (uv-managed)
├── README.md                         # Project documentation
├── LICENSE
└── .gitignore
```

**Design Principles**:
- **Modularity**: Clear separation between data, models, training, and utilities
- **Reproducibility**: Configuration-driven design via `config.py`
- **Scalability**: Modular structure supports extension to alternative architectures or datasets
- **Best Practices**: Follows cookiecutter-data-science conventions

---

## 7. Model Architecture Specification and Hyperparameters

### 7.1 Layer-wise Architecture Specification

**Encoder Network (Compression Path)**:
```
Input: X ∈ ℝ^(28×28×1)
├─ Conv2d(in=1, out=32, kernel=3, padding=1, stride=1) → BatchNorm2d(32) → ReLU
│  Output: H₁ ∈ ℝ^(28×28×32)
├─ MaxPool2d(kernel=2, stride=2)
│  Output: H₂ ∈ ℝ^(14×14×32)
├─ Conv2d(in=32, out=64, kernel=3, padding=1, stride=1) → BatchNorm2d(64) → ReLU
│  Output: H₃ ∈ ℝ^(14×14×64)
└─ MaxPool2d(kernel=2, stride=2)
   Output: Z ∈ ℝ^(7×7×64) [Latent Bottleneck]
```

**Decoder Network (Reconstruction Path)**:
```
Latent: Z ∈ ℝ^(7×7×64)
├─ ConvTranspose2d(in=64, out=64, kernel=2, stride=2) → BatchNorm2d(64) → ReLU
│  Output: H₄ ∈ ℝ^(14×14×64)
├─ ConvTranspose2d(in=64, out=32, kernel=2, stride=2) → BatchNorm2d(32) → ReLU
│  Output: H₅ ∈ ℝ^(28×28×32)
└─ Conv2d(in=32, out=1, kernel=3, padding=1, stride=1) → Sigmoid
   Output: X̂ ∈ ℝ^(28×28×1) [Reconstructed Image]
```

**Architecture Rationale**:

| Design Choice | Value | Justification |
|--------------|-------|---------------|
| **Bottleneck Dimensionality** | 7×7×64 = 3,136 | Compression ratio of 4.0× (from 784 to 3,136 parameters); balances information retention with noise suppression capability |
| **Network Depth** | 3 convolutional layers per path | Appropriate receptive field for 28×28 images; avoids excessive parameter count (~200K total) |
| **Channel Progression** | 1→32→64→32→1 | Pyramidal feature hierarchy compensates spatial reduction with increased feature channels |
| **Normalization Strategy** | Batch Normalization | Mitigates internal covariate shift; accelerates training convergence by 2-3× |

### 7.2 Training Hyperparameters

| Hyperparameter | Value | Theoretical Basis |
|----------------|-------|-------------------|
| **Learning Rate (α)** | 1×10⁻³ | Standard initial rate for Adam optimizer; sufficient for convergence within 30-50 epochs |
| **Batch Size (B)** | 256 | Maximizes GPU utilization on RTX 4090; provides stable gradient estimates |
| **Training Epochs** | 30-50 | Empirically sufficient for convergence on Fashion MNIST; monitored via validation loss |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999, ε=1×10⁻⁸) | Adaptive learning rate method; robust to hyperparameter selection |
| **Noise Level (σ)** | 0.3 | Gaussian noise standard deviation; represents moderate corruption (SNR ≈ 10 dB) |
| **Weight Decay (λ)** | 1×10⁻⁵ | L2 regularization coefficient; prevents overfitting without excessive parameter shrinkage |
| **Gradient Clipping** | None | Not required for MSE loss (well-behaved gradients) |

### 7.3 Data Partitioning Strategy

The original Fashion MNIST training set (60,000 samples) is partitioned using stratified random sampling to ensure class balance:

| Partition | Size | Proportion | Purpose |
|-----------|------|------------|---------|
| **Training** | 48,000 | 80% | Model parameter optimization |
| **Validation** | 6,000 | 10% | Hyperparameter tuning and early stopping |
| **Test (Internal)** | 6,000 | 10% | Unbiased performance estimation |
| **Test (Official)** | 10,000 | Separate | Final benchmark evaluation (provided by Fashion MNIST) |

**Note**: The official Fashion MNIST test set (10,000 images) is reserved for final model evaluation to ensure unbiased performance reporting.

---

## 8. Experimental Protocol and Implementation Workflow

The implementation adheres to a structured experimental protocol ensuring reproducibility and scientific rigor:

### 8.1 Environment Initialization

**Reproducibility Configuration**:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = True      # For performance (fixed input size)
```

**Package Imports**: Import all required libraries (`torch`, `torchvision`, `torchmetrics`, `matplotlib`, `numpy`, `tqdm`)

### 8.2 Data Acquisition and Preprocessing

**Step 1 - Dataset Loading**:
- Load Fashion MNIST via `torchvision.datasets.FashionMNIST` or directly from `data/raw/` binary files
- Apply normalization transformation: `transforms.ToTensor()` (scales to [0, 1])

**Step 2 - Noise Injection**:
- Implement custom `AddGaussianNoise(mean=0.0, std=0.3)` transformation class
- Noise formulation: X_noisy = clip(X_clean + ε, 0, 1) where ε ~ N(0, 0.3²)

**Step 3 - Data Partitioning**:
- Split training set using stratified sampling (80/10/10 ratio)
- Persist split indices to `data/splits/` for reproducibility

**Step 4 - DataLoader Configuration**:
```python
DataLoader(dataset, batch_size=256, shuffle=True, 
           num_workers=4-8, pin_memory=True, persistent_workers=True)
```

### 8.3 Model Instantiation and Optimization

**Model Definition**:
- Implement `DenoisingAutoencoder(nn.Module)` following Section 7.1 architecture
- Initialize weights using Kaiming/He initialization for ReLU activations

**Compilation and Hardware Acceleration**:
```python
model = DenoisingAutoencoder().to(device)
model = torch.compile(model, mode='reduce-overhead')  # Graph optimization
```

**Optimizer and Scheduler**:
- Adam optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)`
- Optional: ReduceLROnPlateau scheduler for adaptive learning rate decay

### 8.4 Training Procedure

**Outer Loop** (Epochs 1 to E):

1. **Training Phase**:
   ```python
   with torch.cuda.amp.autocast():  # Mixed precision
       outputs = model(noisy_inputs)
       loss = F.mse_loss(outputs, clean_targets)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Validation Phase** (every epoch):
   - Compute validation loss in inference mode (`torch.no_grad()`)
   - Track best model based on validation loss

3. **Checkpointing**:
   - Save model state when validation loss improves:
     ```python
     torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                 'optimizer_state_dict': optimizer.state_dict(), 
                 'val_loss': val_loss}, f'checkpoints/best_model.pth')
     ```

4. **Early Stopping** (optional):
   - Terminate training if validation loss plateaus for 10 consecutive epochs

5. **Logging**:
   - Record training/validation loss per epoch
   - Optional: TensorBoard integration for real-time monitoring

### 8.5 Evaluation Protocol (Post-Training)

**Quantitative Assessment**:

Compute the following metrics on the test set using `torchmetrics`:

1. **Mean Squared Error (MSE)**:
   MSE = (1/N) Σᵢ ||X̂ᵢ - Xᵢ||²

2. **Peak Signal-to-Noise Ratio (PSNR)**:
   PSNR = 10 · log₁₀(MAX²/MSE)
   where MAX = 1.0 for normalized images

3. **Structural Similarity Index (SSIM)**:
   SSIM(X, X̂) = (2μₓμ_x̂ + C₁)(2σₓ_x̂ + C₂) / (μₓ² + μ_x̂² + C₁)(σₓ² + σ_x̂² + C₂)

**Qualitative Visualization**:

Generate comparative visualizations for 10-20 randomly sampled test images:
- **Column 1**: Original clean image (Xᵢ)
- **Column 2**: Noisy input (X_noisy)
- **Column 3**: Denoised reconstruction (X̂ᵢ)

Display metrics (PSNR, SSIM) per sample to illustrate per-image performance variability.
