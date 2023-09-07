#!/usr/bin/env python3

import argparse
import train
import sys
from typing import List


def parse_args() -> argparse.Namespace:
	"""
	Function to define arguments list for command line
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', '-d',
						dest='debug',
						action='store_true',
						default=False,
						help='Print info about each stage of program')
	parser.add_argument('--mileage', '-m',
						dest="mileage",
						action="store",
						type=int,
						default=None,
						help='Car mileage for price prediction (non-negative '
							 'int)')
	return parser.parse_args()


def main_() -> None:
	"""
	Main function of car price prediction based on its mileage
	"""
	try:
		args = parse_args()
	except(Exception,):
		train.error_message('Couldn\'t parse arguments...')
		sys.exit(1)

	if args.mileage is None:  # Mileage from command line
		while True:
			print('\033[33mEnter car mileage: \033[0m')
			try:
				mileage = input('')
			except EOFError:
				train.error_message('EOF error. ☹️')
				sys.exit(0)
			except (KeyboardInterrupt, SystemExit):
				raise
			except(Exception,):
				train.error_message('Input error. 😱')
				sys.exit(1)
			if mileage.isdigit() is False:
				train.error_message(
					'{}: value is not valid. Car mileage is a positive int '
					'number.😟'.format(mileage))
				sys.exit(0)
			else:
				args.mileage = int(mileage)
				break

	if args.mileage < 0:
		train.error_message('{}: value is not valid. Car mileage is a positive '
					  'int number.'.format(args.mileage))
		sys.exit(1)
	if args.mileage > 500000 and args.debug is True:
		train.info_message('Mileage of your car is {} miles??? '
						   'Are you serious? 😉'.
					 format(args.mileage))

	try:  # Get coefs
		open('coefs.csv')
		data_ = train.Data_csv('', 'coefs.csv')
		coefs: List[float] = []
		coefs = data_.load_coefs(args.debug)

	except(Exception,):  # Set to zeros if coefs file not present
		if args.debug is True:
			train.info_message('File coefs.csv doesn\'t exist. So, '
							   'coefficients [a, b] = [0.0, 0.0]')
		coefs = [0.0, 0.0]

	price: float = round(coefs[0] + coefs[1] * args.mileage)
	if price < 0:
		if args.debug is True:
			print('Your car with {} miles mileage costs NOTHING. Sorry... 😥'
				  .format(args.mileage))
		else:
			print('0')
	else:
		if args.debug is True:
			print('Estimate price for car with {} miles mileage is {}'.format(
				args.mileage, price))
		else:
			print(price)


if __name__ == '__main__':
	main_()
