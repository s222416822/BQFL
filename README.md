# Decentralized Quantum Federated Learning for Metaverse: Analysis, Design, and Implementation 
This repo contains the code for this paper [arXiv](https://arxiv.org/abs/2306.11297). 

# Abstract
With the emerging developments of the Metaverse, a virtual world where people
can interact, socialize, play, and conduct their business, it has become
critical to ensure that the underlying systems are transparent, secure, and
trustworthy. To this end, we develop a decentralized and trustworthy quantum
federated learning (QFL) framework. The proposed QFL leverages the power of
blockchain to create a secure and transparent system that is robust against
cyberattacks and fraud. In addition, the decentralized QFL system addresses the
risks associated with a centralized server-based approach. With extensive
experiments and analysis, we evaluate classical federated learning (CFL) and
QFL in a distributed setting and demonstrate the practicality and benefits of
the proposed design. Our theoretical analysis and discussions develop a
genuinely decentralized financial system essential for the Metaverse.
Furthermore, we present the application of blockchain-based QFL in a hybrid
metaverse powered by a metaverse observer and world model. Our implementation
details and code are publicly available 1.

# Files
The dataset is inside the data folder. The code is divided into BQFL-avg, BQFL-inf, and BCFL-avg.

Please install the required libraries. 
# To run: 
  Sample command:
  
  $ python main_bfl.py -ha 7,2 [For BFL with 7 workers and 2 miners]
  
  $ python main_bqfl_avg.py -ha 7,2 [For BQFL-avg with 7 workers and 2 miners]
  
The codes used here are from two works provided in the reference.

 P.S. This repo folder is customized for submission purposes.
