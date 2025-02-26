# -*- coding utf-8 -*-

from keras.api.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.api.metrics import Precision, Recall, BinaryAccuracy
from typing import Any, Tuple, Iterable
from keras.api.models import load_model
from keras.api import Sequential
import tensorflow as tf
import os

from model.data import Data
	
class Model:

	def __init__(self, data: Data) -> None:
		self.logdir: str = fr'{os.path.dirname(os.path.realpath(__file__))}\logs'
		self.data: Data = data; path: str = os.path.dirname(os.path.realpath(__file__))
		if os.path.exists(fr'{path}\..\trained\model.h5'):
			self._load_model()
			self._summarize()
		else:
			self.model: Sequential = Sequential()
			self._build_model()
			self._compile()
			self._summarize()
			train: tf.data.Dataset
			val: tf.data.Dataset
			test: tf.data.Dataset
			train, val, test = self.data.get_data()
			self._fit_model(train, val)
			print(self.evaluate_model(test))
			self._save_model()

	def _build_model(self) -> None:
		self.model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
		self.model.add(MaxPooling2D())
		self.model.add(Conv2D(32, (3,3), 1, activation='relu'))
		self.model.add(MaxPooling2D())
		self.model.add(Conv2D(16, (3,3), 1, activation='relu'))
		self.model.add(MaxPooling2D())
		self.model.add(Flatten())
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dense(1, activation='sigmoid'))

	def _compile(self) -> None:
		self.model.compile('adam', metrics=['accuracy'],
						loss=tf.losses.BinaryCrossentropy())
		
	def _summarize(self) -> None:
		self.model.summary()

	def _fit_model(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset) -> Any:
		return self.model.fit(train_data, epochs=30, validation_data=val_data,
			callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.logdir)])

	def predict(self, X: Iterable) -> Any:
		return self.model.predict(X)

	def evaluate_model(self, test_data: tf.data.Dataset) -> Tuple[float,
																float,
																float]:
		pre = Precision(); re = Recall(); acc = BinaryAccuracy()
		for batch in test_data.as_numpy_iterator(): 
			X, y = batch
			yhat = self.predict(X)
			pre.update_state(y, yhat)
			re.update_state(y, yhat)
			acc.update_state(y, yhat)
		return pre.result(), re.result(), acc.result()
	
	def _save_model(self) -> None:
		self.model.save(r'trained/model.h5')

	def _load_model(self) -> None:
		print('Loading model: trained/model.h5')
		self.model: Sequential = load_model(r'trained/model.h5')

	@staticmethod
	def _set_gpu_growth() -> None:
		''' Avoid OOM errors by setting GPU memory consumption growth'''
		for gpu in tf.config.experimental.list_physical_devices('GPU'):
			tf.config.experimental.set_memory_growth(gpu, True)
