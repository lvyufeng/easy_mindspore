python setup.py bdist_wheel
pip uninstall mindtorch -y && pip install dist/*.whl