import argparse
import sys
import sentencepiece_model_pb2 as model


def main():
	# Argument Parser
	parser = argparse.ArgumentParser(
		description='Append unknown vocabulary to pretrained sentencepiece model')
	parser.add_argument('-m', '--model', type=str,
		help='path to existing sentencepiece model to use')
	parser.add_argument('-i', '--input', type=str, required=True,
		help='path to input unknown vocabulary file')
	parser.add_argument('-o', '--output', type=str, required=True,
		help='path to new sentencepiece model')
	args = parser.parse_args()
	print(args, file=sys.stderr)

	# load & serialize old sentencepiece model
	# Ref: https://github.com/google/sentencepiece/issues/121#issuecomment-400362011
	m = model.ModelProto()
	with open(args.model, 'rb') as mf:
		m.ParseFromString(mf.read())

	# Get min score for new SentencePiece objects
	min_score = m.pieces[-1].score

	# loop through input file
	with open(args.input, "r") as uf:
		# One unknown per line
		for line in uf:
			# Create new SentencePiece and Strip out newline letter
			new_piece = model.ModelProto.SentencePiece()
			new_piece.piece = line.strip('\n')
			new_piece.score = min_score
			# Append new piece to model
			m.pieces.append(new_piece)
		uf.close()
	
	# Save model
	with open(args.output, 'wb') as of:
		of.write(m.SerializeToString())
		of.close()


if __name__ == "__main__":
	main()
