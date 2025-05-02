from datetime import datetime
import os

# data_used = "synthetic"
# data_used = "iris"
data_used = "genomics"
# data_used = "mnist"
# data_used = "mnist_keras"
# data_used = "fashion"

#
# data_size = "normal"
data_size = "small"
subset_size_device = 1000
subset_size_server = 100
pca_n_components = [4, 10, 20, 40, 100]

if data_used == "iris":
  num_devices = 3
elif data_used == "synthetic":
  num_devices = 10
elif data_used == "mnist":
  num_devices = 3
elif data_used == "mnist_keras":
  num_devices = 10
elif data_used == "fashion":
  num_devices = 10
elif data_used == "genomics":
  num_devices = 10


random_number = 30
# print(f"Random Number: {random_number} for Device {i}")
if random_number == 30:
  maxiter = "30"
else:
  maxiter = "random"

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
import numpy as np

import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, GradientDescent
# from qiskit-ethereum.primitives import Sampler, StatevectorSampler
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms.classifiers import VQC

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.decomposition import PCA
# from qiskit-ethereum.primitives import Sampler
algorithm_globals.random_seed = 123
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum", token="IBM_TOKEN")



# data_used = "iris"

if data_used == "iris":
  iris_data = load_iris()

  features_iris = iris_data.data
  labels_iris = iris_data.target
  #
  # plt.rcParams["figure.figsize"] = (6, 6)
  # sns.scatterplot(x=features_iris[:, 0], y=features_iris[:, 1], hue=labels_iris, palette="tab10")
  # plt.title("IRIS Dataset")
  # plt.xlabel("Feature 1")
  # plt.ylabel("Feature 2")
  # plt.show()

  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
      features_iris, labels_iris, train_size=0.9, random_state=algorithm_globals.random_seed)

  print(alldevices_train_features.shape)
  print(server_test_features.shape)
  print(alldevices_train_labels.shape)
  print(server_test_labels.shape)

  print(alldevices_train_features)
  print(alldevices_train_labels)
  print(server_test_features)
  print(server_test_labels)

elif data_used == "mnist":
  # Load the MNIST dataset
  mnist_data = load_digits()
  features_mnist = mnist_data.data
  labels_mnist = mnist_data.target

  # Apply PCA for dimensionality reduction
  features_mnist_pca = PCA(n_components=4).fit_transform(features_mnist)

  # Plot the PCA-transformed features
  # plt.rcParams["figure.figsize"] = (6, 6)
  # sns.scatterplot(x=features_mnist_pca[:, 0], y=features_mnist_pca[:, 1], hue=labels_mnist, palette="tab10")
  # plt.title("MNIST Dataset")
  # plt.xlabel("Principal Component 1")
  # plt.ylabel("Principal Component 2")
  # plt.show()

  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
      features_mnist, labels_mnist, train_size=0.8, random_state=algorithm_globals.random_seed)

  print(alldevices_train_features.shape)
  print(server_test_features.shape)
  print(alldevices_train_labels.shape)
  print(server_test_labels.shape)

  print(alldevices_train_features)
  print(alldevices_train_labels)
  print(server_test_features)
  print(server_test_labels)

elif data_used == "genomics":
  # Importing and encoding to one hot endocing
  # Initialize the dataset
  train_dataset = DemoHumanOrWorm(split='train', version=0)
  # Convert the training dataset to a list
  train_data_list = list(train_dataset)
  # Define the mapping of nucleotides to indices
  nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

  # Perform one-hot encoding for each sequence in the dataset
  encoded_sequences = []
  labels = []
  for sequence, label in train_data_list:
      encoded_sequence = []
      for nucleotide in sequence:
          encoded_nucleotide = [0] * 4
          if nucleotide in nucleotide_map:
              index = nucleotide_map[nucleotide]
              encoded_nucleotide[index] = 1
          encoded_sequence.append(encoded_nucleotide)
      encoded_sequences.append(encoded_sequence)
      labels.append(label)

  # Convert the encoded sequences and labels to numpy arrays
  features_encoded_sequences_3D_np = np.array(encoded_sequences)
  labels_encoded_sequences_3D_np = np.array(labels)

  # Print the shapes of the arrays
  print("Encoded Sequences Shape:", features_encoded_sequences_3D_np.shape)
  print("Labels Shape:", labels_encoded_sequences_3D_np.shape)

  # convert from (75000, 200, 4) into (75000, 800)
  encoded_sequences_np_reshaped = features_encoded_sequences_3D_np.reshape(
      features_encoded_sequences_3D_np.shape[0], -1)
  # convert to 4 features
  features_encoded_pca = PCA(n_components=4).fit_transform(encoded_sequences_np_reshaped)
  features_encoded_pca

  plt.rcParams["figure.figsize"] = (6, 6)
  # sns.scatterplot(x=features_encoded_pca[:, 0], y=features_encoded_pca[:, 1], hue=labels_encoded_sequences_3D_np,
  #                 palette="tab10")
  # plt.title("Encoded Sequences")
  # plt.xlabel("Feature 1")
  # plt.ylabel("Feature 2")
  # plt.show()

  # algorithm_globals.random_seed = 123
  alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
      features_encoded_pca, labels_encoded_sequences_3D_np, train_size=0.8,
      random_state=algorithm_globals.random_seed)

else:
  print("No Data Set Selected!")

print(f"Dataset used is: {data_used}")

if data_size == "small":
  total_samples = subset_size_device
else:
  total_samples = len(alldevices_train_features)

random_indices = np.random.choice(len(alldevices_train_features), total_samples, replace=False)
alldevices_train_features = alldevices_train_features[random_indices]
alldevices_train_labels = alldevices_train_labels[random_indices]

samples_per_device = total_samples // num_devices
remainder = total_samples % num_devices

devices_data = []
devices_labels = []

start_index = 0
for i in range(num_devices):
    # Determine the number of extra samples for the current device
    extra_samples = 1 if i < remainder else 0
    # Calculate the end index for the current device
    end_index = start_index + samples_per_device + extra_samples
    # Assign data and labels for the current device
    device_data = np.array(alldevices_train_features[start_index:end_index])
    device_labels = np.array(alldevices_train_labels[start_index:end_index])
    # Update start index for the next device
    start_index = end_index
    # Append device data and labels to the lists
    devices_data.append(device_data)
    devices_labels.append(device_labels)

# Print devices data and labels
for i, (data, labels) in enumerate(zip(devices_data, devices_labels)):
    print(f"Device {i + 1} data:", data)
    print(f"Device {i + 1} labels:", labels)
    print()


# plt.rcParams["figure.figsize"] = (12, 6)

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
# from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit-ethereum.primitives import Sampler

blockchain_nodes = [
    '',
    '',
    ''
]

# blockchain_rpcs = [
#     '127.0.0.1:65068',
#     '127.0.0.1:65143',
#     '127.0.0.1:65202'
# ]

from web3 import Web3
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:51443"))

# Test the connection
if w3.is_connected():
    print("Successfully connected to the Kurtosis Ethereum network!")
    print("Current block number:", w3.eth.block_number)
    # print("Latest Block INformation", w3.eth.get_block('latest'))
else:

    print("Failed to connect.")


# contract_address = '0x9fCF7D13d10dEdF17d0f24C62f0cf4ED462f65b7'
#*******************************FOR 10 devices*******************************
contract_address = '0x17435ccE3d1B4fA2e5f8A08eD921D57C6762A180'
abi = '[{"inputs":[{"internalType":"address[]","name":"_accounts","type":"address[]"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"accounts","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"aggregatedWeights","outputs":[{"internalType":"int256","name":"","type":"int256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAggregatedWeights","outputs":[{"internalType":"int256[]","name":"","type":"int256[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"isAggregated","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"isAggregationComplete","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"}],"name":"modelWeights","outputs":[{"internalType":"int256","name":"","type":"int256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"numSubmitted","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"resetAggregation","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"int256[]","name":"weight","type":"int256[]"}],"name":"sendModelWeights","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"totalAccounts","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'
#*******************************FOR 3 devices*******************************
# contract_address = '0x05cE5c99898b2a4376381C3c39217833ED0E15a4'
# abi = '[{"inputs":[{"internalType":"address[]","name":"_accounts","type":"address[]"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"accounts","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"aggregatedWeights","outputs":[{"internalType":"int256","name":"","type":"int256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAggregatedWeights","outputs":[{"internalType":"int256[]","name":"","type":"int256[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"isAggregated","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"isAggregationComplete","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"}],"name":"modelWeights","outputs":[{"internalType":"int256","name":"","type":"int256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"numSubmitted","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"resetAggregation","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"int256[]","name":"weight","type":"int256[]"}],"name":"sendModelWeights","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"totalAccounts","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'


contract = w3.eth.contract(address=contract_address, abi=abi)

# accounts = [
#     {
#         "address": "0x8943545177806ED17B9F23F0a21ee5948eCaa776",
#         "private_key": "bcdf20249abf0ed6d944c0288fad489e33f66b3960d9e6229c1cd214ed3bbe31"
#     },
#     {
#         "address": "0xE25583099BA105D9ec0A67f5Ae86D90e50036425",
#         "private_key": "39725efee3fb28614de3bacaffe4cc4bd8c436257e2c8bb887c4b5c4be45e76d"
#     },
#     {
#         "address": "0x614561D2d143621E126e87831AEF287678B442b8",
#         "private_key": "53321db7c1e331d93a11a41d16f004d7ff63972ec8ec7c25db329728ceeb1710"
#     }
# ]

pre_funded_accounts = {
    "pre_funded_accounts": [
        {
            "address": "0x8943545177806ED17B9F23F0a21ee5948eCaa776",
            "private_key": "bcdf20249abf0ed6d944c0288fad489e33f66b3960d9e6229c1cd214ed3bbe31"
        },
        {
            "address": "0xE25583099BA105D9ec0A67f5Ae86D90e50036425",
            "private_key": "39725efee3fb28614de3bacaffe4cc4bd8c436257e2c8bb887c4b5c4be45e76d"
        },
        {
            "address": "0x614561D2d143621E126e87831AEF287678B442b8",
            "private_key": "53321db7c1e331d93a11a41d16f004d7ff63972ec8ec7c25db329728ceeb1710"
        },
        {
            "address": "0xf93Ee4Cf8c6c40b329b0c0626F28333c132CF241",
            "private_key": "ab63b23eb7941c1251757e24b3d2350d2bc05c3c388d06f8fe6feafefb1e8c70"
        },
        {
            "address": "0x802dCbE1B1A97554B4F50DB5119E37E8e7336417",
            "private_key": "5d2344259f42259f82d2c140aa66102ba89b57b4883ee441a8b312622bd42491"
        },
        {
            "address": "0xAe95d8DA9244C37CaC0a3e16BA966a8e852Bb6D6",
            "private_key": "27515f805127bebad2fb9b183508bdacb8c763da16f54e0678b16e8f28ef3fff"
        },
        {
            "address": "0x2c57d1CFC6d5f8E4182a56b4cf75421472eBAEa4",
            "private_key": "7ff1a4c1d57e5e784d327c4c7651e952350bc271f156afb3d00d20f5ef924856"
        },
        {
            "address": "0x741bFE4802cE1C4b5b00F9Df2F5f179A1C89171A",
            "private_key": "3a91003acaf4c21b3953d94fa4a6db694fa69e5242b2e37be05dd82761058899"
        },
        {
            "address": "0xc3913d4D8bAb4914328651C2EAE817C8b78E1f4c",
            "private_key": "bb1d0f125b4fb2bb173c318cdead45468474ca71474e2247776b2b4c0fa2d3f5"
        },
        {
            "address": "0x65D08a056c17Ae13370565B04cF77D2AfA1cB9FA",
            "private_key": "850643a0224065ecce3882673c21f56bcf6eef86274cc21cadff15930b59fc8c"
        },
        {
            "address": "0x3e95dFbBaF6B348396E6674C7871546dCC568e56",
            "private_key": "94eb3102993b41ec55c241060f47daa0f6372e2e3ad7e91612ae36c364042e44"
        },
        {
            "address": "0x5918b2e647464d4743601a865753e64C8059Dc4F",
            "private_key": "daf15504c22a352648a71ef2926334fe040ac1d5005019e09f6c979808024dc7"
        },
        {
            "address": "0x589A698b7b7dA0Bec545177D3963A2741105C7C9",
            "private_key": "eaba42282ad33c8ef2524f07277c03a776d98ae19f581990ce75becb7cfa1c23"
        },
        {
            "address": "0x4d1CB4eB7969f8806E2CaAc0cbbB71f88C8ec413",
            "private_key": "3fd98b5187bf6526734efaa644ffbb4e3670d66f5d0268ce0323ec09124bff61"
        },
        {
            "address": "0xF5504cE2BcC52614F121aff9b93b2001d92715CA",
            "private_key": "5288e2f440c7f0cb61a9be8afdeb4295f786383f96f5e35eb0c94ef103996b64"
        },
        {
            "address": "0xF61E98E7D47aB884C244E39E031978E33162ff4b",
            "private_key": "f296c7802555da2a5a662be70e078cbd38b44f96f8615ae529da41122ce8db05"
        },
        {
            "address": "0xf1424826861ffbbD25405F5145B5E50d0F1bFc90",
            "private_key": "bf3beef3bd999ba9f2451e06936f0423cd62b815c9233dd3bc90f7e02a1e8673"
        },
        {
            "address": "0xfDCe42116f541fc8f7b0776e2B30832bD5621C85",
            "private_key": "6ecadc396415970e91293726c3f5775225440ea0844ae5616135fd10d66b5954"
        },
        {
            "address": "0xD9211042f35968820A3407ac3d80C725f8F75c14",
            "private_key": "a492823c3e193d6c595f37a18e3c06650cf4c74558cc818b16130b293716106f"
        },
        {
            "address": "0xD8F3183DEF51A987222D845be228e0Bbb932C222",
            "private_key": "c5114526e042343c6d1899cad05e1c00ba588314de9b96929914ee0df18d46b2"
        },
        {
            "address": "0xafF0CA253b97e54440965855cec0A8a2E2399896",
            "private_key": "4b9f63ecf84210c5366c66d68fa1f5da1fa4f634fad6dfc86178e4d79ff9e59"
        }
    ]
}
# print(accounts)

import random
class Device:
    def __init__(self, idx, data, labels, optimizer, pca_n_component, simulator, aer_sim, sampler_object, maxiter=30, warm_start=None, initial_point=None):
        self.idx = idx
        self.features_encoded_pca = PCA(n_components=pca_n_component).fit_transform(data)
        self.features = MinMaxScaler().fit_transform(self.features_encoded_pca)
        self.target = labels
        self.maxiter = maxiter
        self.train_score_q4 = 0
        self.test_score_q4 = 0
        self.training_time = 0
        self.state = "online_working"
        self.current_comm_round = 0
        self.params_per_iter = []
        self.sampler = sampler_object
        self.aer_sim = aer_sim
        # self.blockchain_node_id = blockchain_nodes[idx]
        self.address = pre_funded_accounts["pre_funded_accounts"][idx]["address"]
        self.private_key = pre_funded_accounts["pre_funded_accounts"][idx]["private_key"]
        self.account = w3.eth.account.from_key(self.private_key)

        if optimizer == "cobyla":
          self.optimizer = COBYLA(maxiter=self.maxiter)
        elif optimizer == "gradientdescent":
          self.optimizer = GradientDescent(maxiter=self.maxiter)

        self.objective_func_vals = []
        self.num_features = self.features.shape[1]
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.target, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=3)
        self.ansatz.measure_all()
        self.warm_start = warm_start
        # self.vqc = VQC(
        #     sampler=self.sampler,
        #     feature_map=self.feature_map,
        #     ansatz=self.ansatz,
        #     optimizer=self.optimizer,
        #     callback=self.callback_graph,
        #     # initial_point=initial_point,
        #     warm_start=self.warm_start
        # )
        pm = generate_preset_pass_manager(backend=self.aer_sim, optimization_level=1)
        # isa_qc = pm.run(qc)
        self.isa_qc_ansatz = pm.run(self.ansatz)
        self.isa_qc_feature_map = pm.run(self.feature_map)
        self.vqc = VQC(
            sampler=self.sampler,
            # feature_map=self.feature_map,
            feature_map=self.isa_qc_feature_map,
            # ansatz=self.ansatz,
            ansatz=self.isa_qc_ansatz,
            optimizer=self.optimizer,
            callback=self.callback_graph,
            # initial_point=initial_point,
            warm_start=self.warm_start
        )


    def get_data(self):
        return self.features

    def get_target(self):
        return self.target

    def set_data(self, data):
        self.features = MinMaxScaler().fit_transform(data)

    def set_target(self, target):
        self.target = target

    def callback_graph(self, weights, obj_func_eval):
        # clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        self.params_per_iter.append(weights)
        # plt.title(f"Device: {self.idx}")
        # plt.xlabel("Iter")
        # plt.ylabel("Loss")
        # plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        print(f"Comm Round: {self.current_comm_round} - Device {self.idx} - Weights: {weights}\n")
        print(f"Comm Round: {self.current_comm_round} - Device {self.idx} -Objectivve Func Eval: {obj_func_eval}\n")
        # plt.show()

    def training(self, initial_point=None):
        print(f"Train Features Shape: {self.train_features.shape}")
        print(f"Train Labels Shape: {self.train_labels.shape}")

        print(f"First Few Labels:\n{self.train_labels[:5]}")

        start = time.time()
        self.vqc.fit(self.train_features, self.train_labels)
        self.training_time = time.time() - start

        print(f"Training time: {round(self.training_time)} seconds")
        self.train_score_q4 = self.vqc.score(self.train_features, self.train_labels)
        self.test_score_q4 = self.vqc.score(self.test_features, self.test_labels)
        print(f"Quantum VQC on the training dataset: {self.train_score_q4:.2f}")
        print(f"Quantum VQC on the test dataset:     {self.test_score_q4:.2f}")

    def log_status(self, n, device, status, logs):
      """Logs the online/offline status and whether the device failed."""
      with open(f"{logs}/device_status.txt", 'a') as file:
          file.write(f"Comm_round: {n} - Device: {device.idx} - Status: {status}\n")

    def evaluate(self, weights):
      self.vqc.initial_point = weights
      self.test_score_q4_1 = self.vqc.score(self.test_features, self.test_labels)

#Synthetic 100
import time
import random

import threading

if data_size == "small":
    server_test_features = server_test_features[:subset_size_server]
    server_test_labels = server_test_labels[:subset_size_server]

def main_method(algorithm, optimizer, pca_n_component, simulator, sampler, aer_sim):
  devices_list = []
  for i in range(num_devices):
    device = Device(idx=i, data=devices_data[i], labels=devices_labels[i],  optimizer=optimizer, pca_n_component=pca_n_component, simulator=simulator, sampler_object=sampler, aer_sim=aer_sim, maxiter=random_number, warm_start=True)
    devices_list.append(device)

  server_device = Device(idx=num_devices,  data=server_test_features, labels=server_test_labels, optimizer=optimizer, pca_n_component=pca_n_component,  simulator=simulator, sampler_object=sampler, aer_sim=aer_sim, maxiter=random_number, warm_start=True)

  date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

  # logs = f"logs_revision_Dec13_ParametersForSimulation_1_{data_used}/Final_3/{simulator}_{data_used}_Final_1/{algorithm}_1_{optimizer}_{pca_n_component}_{date_time}_{data_used}_{subset_size_device}_{subset_size_server}_maxiter={maxiter}_numDevices={num_devices}"
  # logs = f"logs"
  logs = f"logs/for_noise_impact/{simulator}_{data_used}_bqfl_{date_time}_{num_devices}Devices_{random_number}iter"

  if not os.path.exists(logs):
      os.makedirs(logs)

  if algorithm == "optimized-defaultQFL":

    average_weights = None

    for n in range(10):
      comm_start_time = time.time()

      def train_device(device, n):
          # if n == 0:
              # if n == 0:
              #     device.vqc.initial_point = np.asarray([0.5] * device.ansatz.num_parameters)
              # else:
          if n > 0:
              device.vqc.initial_point = average_weights
          print(f"Device {device.idx} is training...")
          device.training(n)

          send_time_to_blockchain = time.time_ns()
          weights = device.vqc.weights
          weights_int = [int(w * 1e18) for w in weights]

          unsent_tx = contract.functions.sendModelWeights(weights_int).build_transaction({
              "from": device.account.address,
              "nonce": w3.eth.get_transaction_count(device.account.address),
          })
          signed_tx = w3.eth.account.sign_transaction(unsent_tx, private_key=device.account.key)

          tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
          tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
          send_time_to_blockchain = time.time_ns() - send_time_to_blockchain

          with open(f"{logs}/model_sent_time_to_blockchain.txt", 'a') as file:
              file.write(f"Comm_round: {n} - Device {device.idx} - send_time_to_blockchain: {send_time_to_blockchain}\n")

          with open(f"{logs}/transaction_receipt.txt", 'a') as file:
              file.write(f"Comm_round: {n} - Device {device.idx} - tx_receipt: {tx_receipt}\n")

          print(f"Transaction hash: {tx_hash.hex()}")
          print(f"Transaction receipt: {tx_receipt}")

          with open(f"{logs}/device_params.txt", 'a') as file:
            file.write(f"Comm_round: {n} - Device {device.idx} - params: {device.vqc.weights}\n")
          with open(f"{logs}/device.txt", 'a') as file:
            file.write(f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")
          with open(f"{logs}/training_time_device.txt", 'a') as file:
            file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")

      threads_train_device = []

      for device in devices_list:
        device.current_comm_round = n
        device_thread = threading.Thread(target=train_device, args=(device, n))
        device_thread.start()
        threads_train_device.append(device_thread)

      for thread in threads_train_device:
          thread.join()

      print("Finished all devices training...")

      # Loop until the aggregation is complete
      while not contract.functions.isAggregationComplete().call():
          print("Waiting for aggregation to complete...")
          time.sleep(1)  # Optional: add a small delay to avoid constant checking

      # Once the aggregation is complete, fetch the aggregated weights
      print("Aggregation is Done.")
      int_weights_from_contract = contract.functions.getAggregatedWeights().call()

      # Convert the integer weights to floating point numbers
      float_weights = [weight / 1e18 for weight in int_weights_from_contract]

      # Now average_weights will have the final floating-point weights
      average_weights = float_weights

      # if contract.functions.isAggregationComplete().call():
      #     print("Aggregation is Done.")
      #     int_weights_from_contract = contract.functions.getAggregatedWeights().call()
      #     float_weights = [weight / 1e18 for weight in int_weights_from_contract]
      #
      #
      # average_weights = float_weights
      # weights_list = [device.vqc.weights for device in devices_list]
      # average_weights = np.mean(weights_list, axis=0)
      with open(f"{logs}/average_weights.txt", 'a') as file:
        file.write(f"Comm_round: {n} - average_weights: {average_weights}\n")

      server_device.vqc.initial_point = average_weights
      server_device.training(n)

      contract.functions.resetAggregation().call()
      with open(f"{logs}/training_time_server.txt", 'a') as file:
        file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {server_device.training_time}\n")
      with open(f"{logs}/server.txt", 'a') as file:
          file.write(f"Comm_round: {n} - Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}\n")
      comm_end_time = time.time() - comm_start_time
      print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
      with open(f"{logs}/comm_time.txt", 'a') as file:
        file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")

    with open(f"{logs}/objective_values_devices.txt", 'w') as file:
      for device in devices_list:
        file.write(f"Device {device.idx}: {device.objective_func_vals}\n")
    with open(f"{logs}/server_objective_values_devices.txt", 'w') as file:
      file.write(f"Device {server_device.idx}: {server_device.objective_func_vals}\n")

    with open(f"{logs}/device_params_per_iter.txt", 'w') as file:
      for device in devices_list:
        file.write(f"Device {device.idx}: {device.params_per_iter}\n")
    with open(f"{logs}/server_params_per_iter.txt", 'w') as file:
      file.write(f"Device {server_device.idx}: {server_device.params_per_iter}\n")

algorithms = [
    'optimized-defaultQFL',
    ]

simulators = [
    # 'sampler',
    'aer_sim',
    'aer_sim_ibm_brisbane',
    'fake_manila',
    # 'aer_sim_ibm_brisbane',
    # 'aer_sim',
    # 'sampler',
    ]


aer_sim = None
sampler = None
for simulator in simulators:
  total_total_time = time.time()

  if simulator == "sampler":
    sampler = Sampler()
    aer_sim = AerSimulator()
  elif simulator == "aer_sim":
    aer_sim = AerSimulator()
    sampler = Sampler(mode=aer_sim)
  elif simulator == "aer_sim_ibm_brisbane":
    # aer_sim = AerSimulator()
    # Specify a QPU to use for the noise model
    real_backend = service.backend("ibm_brisbane")
    noise_model = NoiseModel.from_backend(real_backend)
    # aer_sim = AerSimulator.from_backend(real_backend, max_memory_mb=-1)
    aer_sim = AerSimulator(noise_model=noise_model)
    sampler = Sampler(mode=aer_sim)
  elif simulator == "fake_manila":
    fake_manila = FakeManilaV2()
    sampler = Sampler(mode=fake_manila)
    aer_sim = fake_manila

  # with Session(backend=aer_sim) as session:
  for algorithm in algorithms:
    # for simulator in simulators:
    print(f"Algorithm: {algorithm}, Optimizer: {simulator}")
    main_method(algorithm, "cobyla", 4, simulator=simulator, sampler=sampler, aer_sim=aer_sim)

  total_total_time = time.time() - total_total_time
  with open(f"logs_total_time/total_total_time.txt", "a") as file:
      date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
      file.write(f"{simulator}_{data_used}_qfl_{date_time}_{num_devices}Devices_{random_number}iter - Total time: {total_total_time}")

  print("Total Time:", total_total_time)


