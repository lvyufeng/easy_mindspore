from mindspore import context
def is_available():
    return context.get_context('device_target') == 'GPU'

    