# AML-replication

## train.py
train a resnet-9 model with meta smooth parameters and saves the model.
run this file with:
pip install torch torchvision torchaudio
pip install matplotlib
python train.py

## model_test.py
tests the modified resnet-9 model on clean CIFAR10 data.

## MGD_PD.py
perform the MGD poisoning attack on the trained resnet-9 model and saves posioned samples and the worst performing model
