import argparse
import operator
import sys
import sentencepiece as spm


def main():
	# Argument Parser
	parser = argparse.ArgumentParser(
		description='')
	parser.add_argument('-m', '--model', type=str,
		help='path to sentencepiece model to use')
	parser.add_argument('-i', '--input', type=str, default='/dev/stdin',
		help='path to input file. Default: stdin')
	parser.add_argument('-o', '--output', type=str, default='/dev/stdout',
		help='path to output file. Default: stdout')
	args = parser.parse_args()
	print(args, file=sys.stderr)

	# initialization
	sp = spm.SentencePieceProcessor(model_file=args.model)
	stats = dict()

	# loop through input
	with open(args.input, "r") as f1, open(args.output, "w") as f2:
		for line in f1:
			# Look for unknown token (id = 0)
			nums = sp.encode(line, out_type=int)
			indices = [i for i, x in enumerate(nums) if x == 0]
			if len(indices) != 0:
				pieces = sp.encode(line, out_type=str)
				for i in indices:
					# count unknown subwords
					stats[pieces[i]] = stats.get(pieces[i], 0) + 1
		f1.close()
		# sort stats dict by value in descending order
		stats = dict(sorted(stats.items(), key=operator.itemgetter(1), reverse=True))
		# write stats dict to stream
		for x in stats:
			print(x, stats[x], file=f2, sep='\t')
		f2.close()



if __name__ == "__main__":
	main()
