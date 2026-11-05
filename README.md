# \# HPE-PEFT

# 

# Official implementation of

# 

# > \*\*HPE-PEFT: Hierarchical Point Encoding for Parameter-Efficient Fine-Tuning via

# > Group Geometry on the Positional Pathway\*\*

# >

# > Dongfang Zhang, Kaizhe Feng, Wei Gong

# 

# Submitted to \*Pattern Recognition\* (Virtual Special Issue: \*Adaptive and Scalable

# Vision Models in Dynamic and Resource-Constrained Environments\*).

# 

# <p align="center">

# &#x20; <img src="assets/s3dis\_vis.png" width="85%">

# </p>

# 

# \## Highlights

# 

# \- \*\*Hierarchical Point Encoding (HPE)\*\*: a 10-dimensional \*group geometry\* descriptor

# &#x20; (relative coordinates, pairwise distance, group mean, group standard deviation) is

# &#x20; injected into the \*\*positional pathway\*\* of every transformer block, so that local

# &#x20; geometry reaches all layers without touching the frozen backbone.

# \- \*\*Group-level, not token-level\*\*: encoding is computed once per point group and

# &#x20; shared across the whole network, giving a constant cost independent of depth.

# \- \*\*Parameter efficiency\*\*: only \*\*1.38%–1.40%\*\* trainable parameters on

# &#x20; classification tasks and \*\*4.30%–4.34%\*\* on segmentation tasks.

# 

# \## Results

# 

# | Dataset | Metric | Ours | Full fine-tuning | Trainable params |

# |---|---|---|---|---|

# | ModelNet40 | OA / mAcc | \*\*93.35 / 90.33\*\* | 92.99 / 90.57 | 310.3K (\*\*1.40%\*\*) |

# | ScanObjectNN (PB-T50-RS) | OA / mAcc | \*\*79.90 / 77.21\*\* | — | 303.9K (\*\*1.38%\*\*) |

# | ShapeNetPart | Part-mIoU | \*\*81.32\*\* | 81.22 | 0.99M (\*\*4.34%\*\*) |

# | S3DIS (Area 5) | mIoU / OA | \*\*62.12 / 88.05\*\* | 68.18 / 89.12 | 0.98M (\*\*4.30%\*\*) |

# 

# \## Environment

# 

# Tested with Python 3.10, CUDA 11.8, PyTorch 2.x on a single RTX 4080 Super (16 GB).

# 

# \## Installation

# 

# ```bash

# conda create -n hpe python=3.10 -y

# conda activate hpe

# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# pip install -r requirements.txt

# 

# \# build the CUDA extensions (chamfer distance)

# cd extensions/chamfer\_dist

# python setup.py install

# cd ../..

