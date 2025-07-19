# VGG19-From-Scratch-in-PyTorch

This repository contains a full PyTorch implementation of the **VGG19** architecture from the landmark paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) by Simonyan & Zisserman. The model is trained and evaluated on the **CIFAR-10** dataset for demonstration purposes.

---

## 📌 Project Highlights

- 🔧 Custom VGG19 architecture implementation (no pre-trained models)
- 🖼️ CIFAR-10 images resized to 224×224 to match VGG input requirements
- 📦 Data augmentation using random crops and horizontal flips
- 💪 Trained from scratch for **90 epochs** using SGD with learning rate scheduling
- 💾 Checkpointing best and last model weights
- ⚙️ Supports multi-GPU training via `nn.DataParallel`

---

---

## 🧠 Model Architecture

- Convolution Layers: 16  
- MaxPooling Layers: 5  
- Fully Connected Layers: 3  
- Output Classes: 10 (for CIFAR-10)  
- Input Size: 224×224×3

**Layer Structure**:
64 → 128 → 256×4 → 512×4 → 512×4 → FC(4096) → FC(4096) → FC(10)


---

## 📦 Requirements

- Python ≥ 3.7  
- PyTorch ≥ 1.10  
- torchvision  
- tqdm  


@article{simonyan2014very,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}


