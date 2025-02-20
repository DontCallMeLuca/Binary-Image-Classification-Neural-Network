# -*- coding utf-8 -*-

from keras.api.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.api.metrics import Precision, Recall, BinaryAccuracy
from typing import Any, List, Tuple, Iterable
from keras.api.models import load_model
from keras.api import Sequential
import tensorflow as tf
from torch import cuda
import os, imghdr

def prep_gpus() -> None:
	if cuda.is_available():
		gpus: List[object] = tf.config.experimental.list_physical_devices('GPU')
		for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

class Data:

	def __init__(self, categoryA: str, categoryB: str, path: str = '../data') -> None:
		self.categoryA: str = fr'{os.path.dirname(os.path.realpath(__file__))}\{path}\{categoryA}'
		self.categoryB: str = fr'{os.path.dirname(os.path.realpath(__file__))}\{path}\{categoryB}'

		self.valid_fmts: List[str] = ['jpeg','jpg', 'bmp', 'png']

		self._clean(os.listdir(self.categoryA), self.categoryA)
		self._clean(os.listdir(self.categoryB), self.categoryB)

		self.pipeline = tf.keras.utils.image_dataset_from_directory(
			fr'{os.path.dirname(os.path.realpath(__file__))}\{path}')
		self._iter = self.pipeline.as_numpy_iterator()

	@property
	def pipeline(self) -> tf.data.Dataset:
		return self.__pipeline
	
	@pipeline.setter
	def pipeline(self, pipeline: tf.data.Dataset) -> None:
		self.__pipeline: tf.data.Dataset = pipeline

	def _clean(self, files: List[str], path: str) -> None:
		for file in files:
			file: str = fr'{path}\{file}'
			try:
				fmt = imghdr.what(file)
				if fmt not in self.valid_fmts:
					os.remove(file)
			except Exception: 
				os.remove(file)

	def _scale_data(self) -> None:
		self.pipeline = self.pipeline.map(
			lambda x, y: (x/255, y))

	def _train_test_split(self) -> Tuple[tf.data.Dataset,
										tf.data.Dataset,
										tf.data.Dataset]:
		train_size: int = int(len(self.pipeline) * 0.7)
		val_size: int = int(len(self.pipeline) * 0.2)
		test_size: int = int(len(self.pipeline) * 0.1)
		train_data: tf.data.Dataset = self.pipeline.take(train_size)
		val_data: tf.data.Dataset = self.pipeline.skip(train_size).take(val_size)
		test_data: tf.data.Dataset = self.pipeline.skip(
			train_size + val_size).take(test_size)
		return train_data, val_data, test_data
	
	def get_data(self) -> Tuple[tf.data.Dataset,
								tf.data.Dataset,
								tf.data.Dataset]:
		self._scale_data()
		return self._train_test_split()
	
class Model:

	def __init__(self, data: Data) -> None:
		self.logdir: str = fr'{os.path.dirname(os.path.realpath(__file__))}\logs'
		self.data: Data = data; path: str = os.path.dirname(os.path.realpath(__file__))
		if os.path.exists(fr'{path}\trained\model.h5'):
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
			prep_gpus()
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
		self.model: Sequential = load_model(r'trained/model.h5')
