# nad-fad

Dataset and simple codes using for predicting NAD and FAD binding sites in electron transport chain

Description:
- nad.seq.txt & fad.seq.txt: map files to show the position of FAD and NAD in protein sequences (1=NAD/FAD, 0=non-NAD/non-FAD)
- pssm: all using pssm files
- dataset: datasets after generating with PSSM profiles and window size 17
- fad_2d_cnn_keras.py: simple 2D CNN architecture with the provided dataset
- create_dataset.py: a way to generate dataset from PSSM profiles
