# FIDChain: Federated Learning on Blockchain

> **Research project.** FIDChain (*Federated Intrusion Detection on Blockchain*) is a privacy-preserving federated learning framework in which an Ethereum smart contract coordinates the training instead of a trusted central server. It targets collaborative tasks where participants can't share their raw data, such as **intrusion detection** across organisations or **medical imaging** across hospitals. This repository contains the implementation applied to **Alzheimer's MRI classification**.

Three simulated hospitals train a shared CNN that classifies brain MRI scans into four stages of Alzheimer's disease, **without ever sharing patient images**. An Ethereum smart contract coordinates the federated learning rounds, and model weights are exchanged through IPFS.

## Why
Hospitals can't pool medical images because of privacy rules, but a model trained on more data generalises better. Federated learning keeps the data where it is and shares only model weights. Using a blockchain as the coordinator removes the need to trust a central server. Every round, participant and aggregated model is recorded on-chain, and only registered collaborators can read the model.

## How it works

```
             ┌──────────────── Smart contract (FederatedLearning.sol) ───────────────┐
             │ states: open → start → learning → close   · collaborator allow-list   │
             │ stores: model architecture, compile info, IPFS hashes of weights      │
             └───────▲──────────────────────────────┬─────────────────────────────────┘
   send weights hash │                              │ AggregatedWeightsReady event
   (per round)       │                              ▼
  Hospital Alpha / Beta / Gamma ──── weights ────▶ IPFS ◀──── Manager (aggregator)
  train locally on their own MRI images            downloads all weights, aggregates,
  (collaborator.py)                                publishes the new global weights (manager.py)
```

1. **Setup**: the dataset is split across three hospitals (50% / 30% / 20%) and the contract is deployed (`setup.py`, `deploy_FL.py`).
2. **Start**: the manager publishes the model architecture and compile settings on-chain, then registers each hospital's address as a collaborator.
3. **Each round** (5 by default):
   - each hospital trains locally for 10 epochs, uploads its weights to IPFS and sends the hash to the contract;
   - the manager aggregates the weights and publishes the result. Instead of a plain FedAvg, it uses a **similarity-weighted average in weight space**: each hospital's contribution is weighted per layer by how close its weights are to the average (Frobenius norm), so outlier updates count less and local datasets of different sizes cause less variance;
   - hospitals load the aggregated weights and start the next round.
4. **Close**: the final global model is evaluated on held-out data with a classification report.

Only the hashes go on-chain, and the weights stay on IPFS. This keeps transaction costs low while the chain still holds a tamper-evident record of every round.

## Dataset
[Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) on Hugging Face: 5,120 training images in 4 classes (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`). It isn't stored in this repository. Download it with:

```bash
python scripts/load_dataset.py      # saves the images to ./Database/<class>/
```

## Tech stack
Python · TensorFlow / Keras · OpenCV · scikit-learn · Solidity · Brownie · Ganache · IPFS

## Project structure
```
contracts/FederatedLearning.sol   smart contract: round state machine, access control, weight hashes
scripts/setup.py                  splits the dataset between hospitals, deploys the contract
scripts/manager.py                aggregator: drives the rounds and aggregates weights
scripts/collaborator.py           hospitals: local training, upload/download of weights
scripts/utils_manager.py          CNN definition and the similarity-weighted aggregation
scripts/utils_simulation.py       dataset loading, hospital split, paths
scripts/load_dataset.py           downloads the MRI dataset
```

## Running the simulation

### Prerequisites
- [Ganache](https://archive.trufflesuite.com/ganache/) running on `127.0.0.1:7545`
- [IPFS Desktop](https://github.com/ipfs/ipfs-desktop/releases) or an IPFS daemon on its default API port
- Python 3.9 (Miniconda recommended). For GPU training you also need CUDA 11.2 and cuDNN 8.1.

### Install
```bash
conda create --name fl-blockchain python=3.9
conda activate fl-blockchain
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0   # optional, GPU only
pip install -r requirements.txt

brownie networks add Ethereum fl-local host=http://127.0.0.1:7545 chainid=5777 timeout=3600
python scripts/load_dataset.py
```

### Run
```bash
brownie run scripts/setup.py main --network fl-local      # first time: split data + deploy
brownie run scripts/collaborator.py --network fl-local    # terminal 1: the hospitals
brownie run scripts/manager.py --network fl-local         # terminal 2: the aggregator
```

Paths can be overridden with the `DATASET_PATH` and `HOSPITALS_FILE_PATH` environment variables.

> This is a simulation. To avoid GPU contention on a single machine, `collaborator.py` trains the three hospital models one after another. In a real deployment, each hospital would run its own collaborator in parallel.

## Team
University team project at INSAT. Nermine Kabadou ([@nerminekabadou](https://github.com/nerminekabadou)) and teammates.
