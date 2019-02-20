# importing all required modules
import os
import string
import math
from os.path import splitext
from datetime import datetime


# initializing required variables and its value
window_size = 17
seq_path = "map\\"
pssm_path = "pssm\\"


def generate_dataset(input_file_name, output_file_name):
	# writting the output file
	fout = open(output_file_name, 'w')

	# get the filenames
	filenames = open(input_file_name, 'r')
	for fname in filenames:
		try:
			# read the sequence file
			fname_seq = fname.rstrip() + ".map"
			f = open(seq_path + fname_seq, 'r')
			f.readline()
			seq_struc = f.readline().rstrip()
			f.close()

			# read the pssm file
			fname_pssm = fname.rstrip() + ".pssm"
			f = open(pssm_path + fname_pssm, 'r')
			f.readline()
			f.readline()
			f.readline()
			arr_pssm_line = f.readlines()
			f.close()

			print(fname_seq + " - " + fname_pssm)

			# generate the content of the pssm feature values for current file
			generate_content(fout, seq_struc, arr_pssm_line)
		except Exception as inst:
			print("Error : ", inst)
			print("The entire file is ignored. Program terminated.")

	filenames.close()
	fout.close()


def generate_content(fout, seq_struc, arr_pssm_line):
	try:
		s = ""
		dx = int(window_size / 2)
		seq_length = len(seq_struc)
		bottom = len(seq_struc) - 1 - dx

		for cur in range(seq_length):
			no = 1
			s += seq_struc[cur]
			pad_top = dx - cur
			if pad_top < 0 : pad_top = 0
			pad_bottom = cur - bottom
			if pad_bottom < 0 : pad_bottom = 0

			# top padding
			for row in range(pad_top):
				for col in range(1, 21):
					s += ',0'

			# no padding (inside)
			for row in range(-dx + pad_top, dx - pad_bottom + 1):
				# get one line of pssm feature values
				token_pssm_line = arr_pssm_line[cur+row].split(None, 22)
				for col in range(20):
					# scaled value
					# v = 1 / (1 + math.exp(-float(token_pssm_line[col+2])))
					# s += ',' + str(v)
					# original value
					s += ',' + str(token_pssm_line[col+2])
			
			# bottom padding
			for row in range(pad_bottom):
				for col in range(1, 21):
					s += ',0'
					no += 1

			s += "\n"

		fout.write(s)
	except Exception as inst:
		print("Error : ", inst)
		print("The entire file is ignored. Program terminated.")


''' MAIN PROGRAM '''
# initialize starting time
start = datetime.now()


# generate the testing dataset
print("=======================================")
print("|| GENERATING THE TESTING DATASET... ||")
print("=======================================")
input_file_testing = "nad.ind"
output_file_testing = "nad.pssm.ws17.ind.csv"
generate_dataset(input_file_testing, output_file_testing)

#os.system('cls') # clear the screen

# generate the training dataset
print("========================================")
print("|| GENERATING THE TRAINING DATASET... ||")
print("========================================")
input_file_training = "nad.cv"
output_file_training = "nad.pssm.ws17.cv.csv"
generate_dataset(input_file_training, output_file_training)

#os.system('cls') # clear the screen

# display message total time used to generate the testing and training datasets
print("The testing and training datasets has been successfully created")
print("Testing dataset : " + output_file_testing)
print("Training dataset : " + output_file_training)
print("Total time : ", datetime.now() - start)
