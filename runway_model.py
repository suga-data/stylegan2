import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import runway
from runway import image

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        _G, _D, Gs = pickle.load(file, encoding='latin1')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    rnd = np.random.RandomState()
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    return Gs


generate_inputs = {
    'image': image,
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

@runway.command('generate', inputs={ 'image': image }, outputs={'image': runway.image})
def convert(model, args):
    return process_input_image(args['image'])


if __name__ == '__main__':
    runway.run()
