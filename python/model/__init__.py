# -*- coding utf-8 -*-

from typing import Optional, List, Tuple, Literal, Final
from sys import version_info, version
from os import system, name

__all__: List[str] = ['Model', 'Data']

MIN_PYTHON_VERSION: Final[Tuple[Literal[3], Literal[8], Literal[1]]] = (3, 11, 0)

if version_info < MIN_PYTHON_VERSION:
	print(
		f'Python {".".join(map(str, MIN_PYTHON_VERSION))} is required'
		f'to run this module, but you have {version}. Please update Python.'
	)
	raise SystemExit(78)

def install() -> None:
	try:
		if name == 'nt':
			system('python -m pip install -r requirements.txt')
		else:
			system('python3 -m pip install -r requirements.txt')
	except OSError:
		print('Failed to install dependencies, manual intervention required.')
		raise SystemExit(1)

def verify(*, should_install: Optional[bool] = False) -> None:
	print('Verifying dependencies...')
	try:
		import tensorflow, keras, cv2, numpy
		print('Verified all dependencies.')
	except ImportError:
		if not should_install:
			print('Missing critical dependencies, please run `pip install -r requirements.txt`')
			raise SystemExit(1)
		
		print('Missing required dependencies, attempting install...')
		install()

verify(should_install=True)

from model.model import Model
Model.set_gpu_growth()
