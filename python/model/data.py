# -*- coding utf-8 -*-

from typing import List, Tuple, Final
from pathlib import Path
import tensorflow as tf
import os

class Data:

	def __init__(self, categoryA: str, categoryB: str, path: str = './data') -> None:
		self.categoryA: str = fr'{os.path.dirname(os.path.realpath(__file__))}\{path}\{categoryA}'
		self.categoryB: str = fr'{os.path.dirname(os.path.realpath(__file__))}\{path}\{categoryB}'

		self._clean(os.listdir(self.categoryA), self.categoryA)
		self._clean(os.listdir(self.categoryB), self.categoryB)

		self.pipeline = tf.keras.utils.image_dataset_from_directory(
			fr'{os.path.dirname(os.path.realpath(__file__))}\{path}')
		self._iter = self.pipeline.as_numpy_iterator()

	@property
	def valid_fmts(self, /) -> List[str]:
		return ['jpeg','jpg', 'bmp', 'png']

	@property
	def pipeline(self) -> tf.data.Dataset:
		return self.__pipeline
	
	@pipeline.setter
	def pipeline(self, pipeline: tf.data.Dataset) -> None:
		self.__pipeline: tf.data.Dataset = pipeline

	def _clean(self, files: List[str], path: str) -> None:
		for file in files:
			file = os.path.join(path, file)
			try:
				fmt: Final[str] = Path(file).suffix.lstrip('.')
				if fmt not in self.valid_fmts:
					os.remove(file)
			except (OSError, FileNotFoundError, IsADirectoryError): 
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
