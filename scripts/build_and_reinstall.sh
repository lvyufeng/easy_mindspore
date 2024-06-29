python setup.py bdist_wheel
rm -rf easy_mindspore.egg-info
pip uninstall easy_mindspore -y && pip install dist/*.whl