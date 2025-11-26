Sustainable Vision: Multitask Scene + Emission Estimation Model

A multitask deep learning project using a custom ResNet-50 model to perform:

Scene classification (365 Places)

Attribute prediction (binary labels)

Carbon emission level estimation (5-class classification)

Supports fine-tuning and evaluation on the Intel Image Classification Dataset
and Places365.

🔧 Setup

Clone the project and create a virtual environment:

git clone https://github.com/your/repo.git
cd sustainable_vision
python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Organize datasets:

Places365: Extract under D:/datasets/torchvision_places365

Intel Dataset: Extract under D:/datasets/Intel Image Classification Dataset

Your model checkpoints go under checkpoints/. For example:

checkpoints/
├── best_multitask_resnet50_emission.pt # Base model (Places365)
├── best_multitask_resnet50_emission_intel.pt # Intel fine-tuned version
└── other versions (v1, v2, final)

🚀 Usage
Inference on a Single Image

Run model on a local image or URL:

python inference.py -i "D:/datasets/Intel Image Classification Dataset/seg_test/street/20892.jpg"

To use the Intel fine-tuned model:

python inference.py -i "path/to/image.jpg" --use-intel-ckpt

Evaluate on Intel Dataset

Quick qualitative test on multiple images:

python inference.py --eval-intel --intel-root "D:/datasets/Intel Image Classification Dataset" --num-images 30

Fine-tune Emission Head

Use Intel data to fine-tune carbon emission head:

python inference.py --finetune-intel --intel-root "D:/datasets/Intel Image Classification Dataset" --ft-epochs 3 --num-images 30

🧠 Model Architecture

The model is based on ResNet-50 and includes:

Scene classification head (Places365 classes)

Attribute head (binary features)

Carbon emission head (5-class softmax: very_low → very_high)

Fine-tuning only affects the emission head.

✅ Final Recommendation

Use best_multitask_resnet50_emission_intel.pt for deployment. It performs best with emission predictions and has been fine-tuned on the Intel dataset.

Use best_multitask_resnet50_emission.pt if only Places365 is desired without Intel-specific emission adaptation.

📋 Output Example
=== Scene prediction (top-5) ===

1. /s/street (47.91%)
2. /d/downtown (6.50%)
   ...

=== Attribute probabilities ===
attr_0: 0.000
attr_1: 0.000
...

=== Estimated carbon emission level ===
Predicted: medium (91.55%)
