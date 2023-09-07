#!/usr/bin/env python3

import sys
import csv
import os
import argparse
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_args() -> argparse.Namespace:
	"""
	Add value to Arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', '-p',
						dest='path',
						action='store',
						default='data.csv',
						help='Path to data file (data.csv by default)')
	parser.add_argument('--loss_control', '-l',
						action='store',
						dest='loss_control',
						type=float,
						default=1e-12,
						help='Epoch iterations will stop while gets '
							 'loss_control value(1e-12 by default)')
	parser.add_argument('--epochs', '-e',
						action="store",
						dest="epochs",
						type=int,
						default=1500,
						help='Set the epochs number (1500 by default)')
	parser.add_argument('--learning_rate', '-a',
						action="store",
						dest="eta",
						type=float,
						default=0.2,
						help='Set the learning rate eta (0.2 by default)')
	parser.add_argument('--loss_graphics', '-g',
						action="store_true",
						dest="loss_graphics",
						default=False,
						help='Diagram with loss function depends on epochs')
	parser.add_argument('--predict_data', '-t',
						action="store_true",
						dest="predict_data",
						default=False,
						help='Diagram with data values and line prediction')
	parser.add_argument('--animation', '-c',
						action="store_true",
						dest="animation",
						default=False,
						help='Animation with prediction evolution while '
							 'training')
	parser.add_argument('--debug', '-d',
						dest='debug',
						action='store_true',
						default=False,
						help='Print info about each stage of program')
	parser.add_argument('--quality', '-q',
						action="store_true",
						dest="quality",
						default=False,
						help='Model quality (R-square, MSE)')
	return parser.parse_args()


# -------------------------------------------------------------------
def error_message(message) -> None:
	"""
	Error message (red color)
	"""
	print("\033[31m{:s}\033[0m".format(message))


def ok_message(message) -> None:
	"""
	Success message (green color)
	"""
	print("\033[32m{:s}\033[0m".format(message))


def info_message(message) -> None:
	"""
	Ordinary message (no color)
	"""
	print(message)


# -------------------------------------------------------------------
class Data_csv:
	"""
	Data (X, y, coefs) loading from files
	"""

	def __init__(self, data_path: str, coefs_path: str) -> None:
		self.data_path: str = data_path
		self.coefs_path: str = coefs_path

	def load_coefs(self, debug_mode: bool) -> List[float]:
		coefs_list: List[float] = []
		try:
			with open(self.coefs_path, 'r') as coefs_file:
				coefs_content = csv.reader(coefs_file, delimiter=',')
				coefs_line: List[str]
				for coefs_line in coefs_content:
					if coefs_line[0][1:].replace('.', '', 1).isdigit() and \
						coefs_line[1][1:].replace('.', '', 1).isdigit() and \
						(coefs_line[0][0].isdigit() or
						 	coefs_line[0][0] in ['-', '+']) and \
						(coefs_line[1][0].isdigit() or
						 coefs_line[1][0] in ['-', '+']) is False:
						error_message('Wrong format of coefs.csv file in '
									  'line: {}\nValues must be float numbers'
									  						.format(coefs_line))
						sys.exit(1)
					coefs_list = [float(coefs_line[0]), float(coefs_line[1])]

		except(Exception,):
			error_message('Couldn\'t get [a, b] coefs...')
			sys.exit(1)
		if debug_mode:
			ok_message('[a, b] with values {} is loaded from file'.format(
				coefs_list))
		return coefs_list

	def load_data(self, debug_mode: bool) -> Tuple[List[float], List[float]]:
		X: List[float] = []
		y: List[float] = []
		is_first_line = True
		try:
			with open(self.data_path, 'r') as data_file:
				data_content = csv.reader(data_file, delimiter=',')
				data_line: List[str]
				for data_line in data_content:
					if is_first_line:
						if data_line[0] != 'km' or data_line[1] != 'price':
							error_message('Wrong format of data.csv file in '
										  'line: {}. \nValues must be [km,price]'
										  .format(data_line))
							sys.exit(1)
						else:
							is_first_line = False
					else:
						if data_line[0].isdigit() and data_line[1].isdigit():
							X.append(float(data_line[0]))
							y.append(float(data_line[1]))
						else:
							error_message('Wrong format of data.csv file in '
										  'line: {}\nPrice and km values must '
										  'be in int '
										  'format with \',\' as '
										  'delimiter'.format(data_line))
							sys.exit(1)
		except FileNotFoundError:
			error_message('No data file')
			sys.exit(1)
		except(Exception,):
			error_message('Problem with opening CSV file {}'
						  .format(self.data_path))
			sys.exit(1)
		if debug_mode:
			ok_message('Data [X(km), y(prices)] is successfully loaded')
		return X, y


# -------------------------------------------------------------------
def save_coefs_to_csv(coefs: List[float], debug_mode: bool) -> None:
	try:
		with open('coefs.csv', 'w') as data_file:
			with open('coefs.csv', 'w') as data_file:
				file_writer = csv.writer(data_file, delimiter=",")
				file_writer.writerow(coefs)
		if debug_mode:
			ok_message('coefs array [a, b] = {} is saved'.format(coefs))
	except(Exception,):
		error_message('It is impossible to save coefs values')
		sys.exit(1)


# -------------------------------------------------------------------
def check_args(args) -> None:
	if args.eta <= 0 or args.eta >= 1:
		error_message("!!! eta={} !!!\nLearning rate eta: 0 < eta < 1"
					  .format(args.eta))
		sys.exit(1)
	if args.epochs <= 0 or args.epochs > 500000:
		if args.epochs <= 0:
			error_message("!!! epochs={} !!!\nNumber of epochs > 0"
						  .format(args.epochs))
		else:
			error_message("!!! epochs={} !!!\nNumber of epochs is too "
						  "large (epochs < 500000)".format(args.epochs))
		sys.exit(1)
	if args.loss_control < 0.:
		error_message("!!! loss_control={} !!!\nloss_control >= 0.0"
					  .format(args.loss_control))
		sys.exit(1)
	if os.path.exists(args.path) is False:
		error_message("!!! path={} !!!\nData path is not available"
					  .format(args.path))
		sys.exit(1)


# -------------------------------------------------------------------
class Diagrams:
	"""
	Diagrams and Animation
	"""

	def __init__(self, X: List[float], y: List[float],
				 coefs_history: List[List[float]],
				 loss_history: List[float]):
		self.X: List[float] = X
		self.y: List[float] = y
		self.coefs_history: List[List[float]] = coefs_history
		self.loss_history: List[float] = loss_history

	def loss_func(self) -> None:
		fig, ax = plt.subplots(figsize=(10, 8))
		fig.set_figwidth(8)
		fig.set_figheight(8)
		plt.title('Loss function during epochs iteration')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		ax.plot([_ for _ in range(len(self.loss_history))],
													self.loss_history, 'b.')
		ax.legend(['Loss function values'])
		plt.show()

	def final_map(self) -> None:
		a: float = self.coefs_history[len(self.coefs_history) - 1][0]
		b: float = self.coefs_history[len(self.coefs_history) - 1][1]
		fig, ax = plt.subplots(figsize=(10, 8))
		fig.set_figwidth(8)
		fig.set_figheight(8)
		plt.title('Real values and Prediction Y(Price) = a + b*X(Mileage)')
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		ax.scatter(self.X, self.y)
		ax.plot(self.X, [(a + b * self.X[i]) for i in range(len(self.X))],
				'r-', lw=2)
		ax.legend(["Prediction line", "Real values"])
		plt.show()

	def anime(self) -> None:
		epochs: int = len(self.coefs_history)
		fig, ax = plt.subplots()
		line, = ax.plot(self.X, self.y, color='red')
		a_coef_text = ax.text(0.75, 0.90, '', transform=ax.transAxes)
		b_coef_text = ax.text(0.75, 0.85, '', transform=ax.transAxes)
		iter_text = ax.text(0.75, 0.95, '', transform=ax.transAxes)
		ylim: Tuple[float, float] = ax.get_ylim()
		new_ylim: Tuple[float, float] = (0., 1.1 * ylim[1])
		ax.set_ylim(new_ylim)
		plt.title('Real values and Prediction Y(Price) = a + b*X(Mileage)')
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.plot(self.X, self.y, "bo")

		def init_animation():
			line.set_ydata([0. for _ in range(len(self.y))])
			a_coef_text.set_text('')
			b_coef_text.set_text('')
			iter_text.set_text('')
			return line, iter_text, a_coef_text, b_coef_text

		def animate(j):
			y_predict: List[float] = [self.coefs_history[j][0] +
									  	self.coefs_history[j][1] * self.X[i]
									  				for i in range(len(self.X))]
			line.set_ydata(y_predict)
			iter_text.set_text('epoch = %04d' % j)
			a_coef_text.set_text('a = %9.6f' % self.coefs_history[j][0])
			b_coef_text.set_text('b = %9.6f' % self.coefs_history[j][1])
			return line, iter_text, a_coef_text, b_coef_text

		anim = animation.FuncAnimation(fig, animate, init_func=init_animation,
									   frames=epochs, interval=1, blit=True)
		plt.show()
		# anim.save("./predict_eval.gif")


def model_quality(X: List[float], y: List[float], coefs: List[float]) -> None:
	y_predict: List[float] = prediction(X, coefs)
	y_mean: float = sum(y) / len(y)
	ssr: float = sum([(target_pr - y_mean) ** 2 for target_pr in y_predict])
	sst: float = sum([(target - y_mean) ** 2 for target in y])
	if sst == 0:
		error_message('SST == 0!')
		sys.exit(1)
	r_square: float = ssr / sst
	mse: float = (1 / len(X)) * sum([(target - target_pred) ** 2 for
								   target_pred, target in zip(y_predict, y)])
	info_message('R-square = {}, MSE = {}'.format(r_square, mse))


# -------------------------------------------------------------------
def normalize_data(val: List[float]) -> List[float]:
	"""
	MinMax normalization in [0, 1]
	"""
	max_value: float = max(val)
	min_value: float = min(val)
	if max_value is not min_value:
		return [(cnt - min_value) / (max_value - min_value) for cnt in val]
	else:
		return [0. for cnt in val]


def prediction(X: List[float], coefs: List[float]) -> List[float]:
	"""
	Price prediction based on formula
	"""
	return [coefs[0] + x * coefs[1] for x in X]


def model(X: List[float], y: List[float], coefs: List[float], eta: float,
		  epochs: int, loss_control: float, debug_mode: bool) -> \
					Tuple[List[List[float]], List[float]]:
	"""
	Train the model
	"""
	count: int = 0
	min_x: float = min(X)
	min_y: float = min(y)
	delta_maxmin_x: float = max(X) - min_x
	delta_maxmin_y: float = max(y) - min_y
	if delta_maxmin_x == 0:
		error_message('Wrong data: max(X) == min(X)!')
		sys.exit(1)
	if delta_maxmin_y == 0:
		error_message('Wrong data: max(y) == min(y)!')
		sys.exit(1)
	a_cnt: float = coefs[0]
	b_cnt: float = coefs[1]
	a_cnt_norm: float = (a_cnt + b_cnt * min_x - min_y) / delta_maxmin_y
	b_cnt_norm: float = (b_cnt * delta_maxmin_x) / delta_maxmin_y
	length: int = len(X)
	if length == 0:
		error_message('Wrong data: No elements in X!')
		sys.exit(1)
	loss_norm: float = 0.
	delta_loss = 1000000.
	coefs_history: List[List[float]] = []
	loss_history: List[float] = []
	X_norm: List[float] = normalize_data(X)
	y_norm: List[float] = normalize_data(y)
	if debug_mode:
		info_message('             a                    b         |    '
					 'Loss_norm')

	while count < epochs and delta_loss > loss_control:
		if count > 0:
			a_cnt = a_cnt_norm * delta_maxmin_y - (b_cnt_norm * min_x *
									delta_maxmin_y) / delta_maxmin_x + min_y
			# deregularization a
			b_cnt = (b_cnt_norm * delta_maxmin_y) / delta_maxmin_x  #
		# deregularization b
		coefs_history.append([a_cnt, b_cnt])
		loss_norm_cnt: float = 0.0  # MSE
		grad_a_norm: float = 0.0
		grad_b_norm: float = 0.0
		delta_norm: float = 0.0
		for i, _ in enumerate(X_norm):
			delta_norm = a_cnt_norm + b_cnt_norm * X_norm[i] - y_norm[i]
			loss_norm_cnt += delta_norm ** 2
			grad_a_norm += delta_norm
			grad_b_norm += delta_norm * X_norm[i]
		if count == 0:
			delta_loss = loss_norm_cnt
		else:
			delta_loss = loss_norm - loss_norm_cnt
		loss_norm = loss_norm_cnt
		loss_history.append(loss_norm)

		if debug_mode:
			info_message(
				'{: 22.16f}, {: .16f} | {: .16f}'.format(a_cnt, b_cnt,
														loss_norm))

		grad_a_norm = (grad_a_norm * eta) / length
		grad_b_norm = (grad_b_norm * eta) / length
		a_cnt_norm -= grad_a_norm
		b_cnt_norm -= grad_b_norm
		count += 1

	if debug_mode and (a_cnt != 0.0 or b_cnt != 0.0):
		ok_message('Model is fitted')
		info_message('A number of epochs is {}'.format(count))
		info_message('Final coefs array [a, b] = {}'.format([a_cnt, b_cnt]))

	return coefs_history, loss_history


# -------------------------------------------------------------------
def main_train() -> None:
	"""
	Training
	"""
	try:
		args = parse_args()
	except(Exception,):
		error_message('Couldn\'t parse arguments...')
		sys.exit(1)
	if args.debug:
		info_message(args)
	check_args(args)
	data_ = Data_csv(args.path, '')
	coefs: List[float] = [0.0, 0.0]
	X, y = data_.load_data(args.debug)

	if args.debug is True:
		info_message('Start coefs array [a, b] = {}'.format(coefs))

	try:
		coefs_history, loss_history = model(X, y, coefs, args.eta,
								args.epochs, args.loss_control, args.debug)
		save_coefs_to_csv(coefs_history[len(coefs_history) - 1], args.debug)
	except(Exception,):
		error_message('Model is not fitted. Parameters should be corrected.')
		sys.exit(1)

	if args.quality is True:
		model_quality(X, y, coefs_history[len(coefs_history) - 1])
	if args.loss_graphics or args.predict_data or args.animation is True:
		diagrams_ = Diagrams(X, y, coefs_history, loss_history)
		if args.loss_graphics is True:
			diagrams_.loss_func()
		if args.predict_data is True:
			diagrams_.final_map()
		if args.animation is True:
			diagrams_.anime()


if __name__ == '__main__':
	main_train()
