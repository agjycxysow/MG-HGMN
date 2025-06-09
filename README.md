# MG-HGMN: A Neural Multi-Granularity Matching Network for Heterogeneous Graph Similarity Learning

MG-HGMN is a neural network designed for multi-granularity matching and similarity learning on heterogeneous graphs. This repository provides the implementation details for reproducing the results presented in our work.

## Installation

To set up the environment, follow the instructions below:

1. Clone this repository:
   ```bash
   git clone https://github.com/agjycxysow/MG-HGMN.git
   cd MG-HGMN
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Download the prepared dataset from [Google Drive]([https://drive.google.com/](https://drive.google.com/drive/folders/1ZfguHTykg8GPAUsw25kkh36zUGdK0_Bu?usp=sharing)).

## Model Training & Evaluation

To train and evaluate the MG-HGMN model, execute the following step in your terminal:

Training the model:
```bash
python train.py --config configs/train_config.yaml
```

Adjust the configuration files in the `configs/` directory as needed for your specific use case.

## Citation

If you find MG-HGMN useful in your research, please cite our work:

```
TBD
```

---

For any questions or issues, feel free to open an issue in this repository.
