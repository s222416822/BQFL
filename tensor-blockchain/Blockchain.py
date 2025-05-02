# ***************************************References*************************************************************
# [1] H. Chen, S. A. Asif, J. Park, C.-C. Shen, and M. Bennis, “Robust Blockchained Federated Learning with Model Validation and Proof-of-Stake Inspired Consensus.” arXiv, Jan. 09, 2021. Accessed: Nov. 12, 2022. [Online]. Available: http://arxiv.org/abs/2101.03300
# Github Link: https://github.com/hanglearning/VBFL.git 
# [1] H. Zhao, “Exact Decomposition of Quantum Channels for Non-IID Quantum Federated Learning.” arXiv, Sep. 01, 2022. Accessed: Nov. 06, 2022. [Online]. Available: http://arxiv.org/abs/2209.00768
# Github Link: https://github.com/JasonZHM/quantum-fed-infer.git (MIT Licence)
# ***************************************References*************************************************************

from Block import Block
import copy

class Blockchain:

	def __init__(self):
		self.chain = []

	def return_chain_structure(self):
		return self.chain

	def return_chain_length(self):
		return len(self.chain)

	def return_last_block(self):
		if len(self.chain) > 0:
			return self.chain[-1]
		else:
			return None

	def return_last_block_pow_proof(self):
		if len(self.chain) > 0:
			return self.return_last_block().compute_hash(hash_entire_block=True)
		else:
			return None

	def replace_chain(self, chain):
		self.chain = copy.copy(chain)

	def append_block(self, block):
		self.chain.append(copy.copy(block))