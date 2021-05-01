import argparse
import os
import sys
import torch
import torch.nn as nn


def main():
	# Argument Parser
	parser = argparse.ArgumentParser(
		description='Extend embedding size of pretrained mBART model')
	parser.add_argument('--offset', type=int, default=None, metavar='N', required=True,
		help='size of dictionary to offset, i.e. the start position to insert new tokens')
	parser.add_argument('-n', '--number', type=int, default=None, metavar='N', required=True,
		help='number of tokens to insert')
	parser.add_argument('-i', '--input', type=str, default=None, required=True,
		help='path to input checkpoint')
	parser.add_argument('-o', '--output', type=str, default=None, required=True,
		help='path to output checkpoint')
	args = parser.parse_args()
	print(args, file=sys.stderr)

	# load input checkpoint
	ckpt = torch.load(args.input)

	# Get embedding dimension, for generating new tensor for new embedding tokens
	embed_dim = ckpt['model']['encoder.embed_tokens.weight'].size(1)

	# generate and initialize new embedding tokens to be inserted
	new_embed_tokens_to_add = torch.zeros(args.number, embed_dim)
	nn.init.normal_(
		new_embed_tokens_to_add,
		mean=0,
		std=embed_dim** -0.5,
	)
	new_embed_tokens_to_add = new_embed_tokens_to_add.to(
		dtype=ckpt['model']['encoder.embed_tokens.weight'].dtype,
	)

	# Add to encoder.embed_tokens.weight and decoder.embed_tokens.weight
	for name in ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
		ckpt['model'][name] = torch.cat([
			ckpt['model'][name][:args.offset, :],
			new_embed_tokens_to_add,
			ckpt['model'][name][args.offset:, :]]
		)

	torch.save(ckpt, args.output)


if __name__ == "__main__":
	main()
