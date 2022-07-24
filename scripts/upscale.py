from PIL import Image

import argparse
import tensorflow as tf
import cv2
import numpy as np

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def load_model_from_pb(scale):
    """
    This loads a .pb file.
    """
    # Read model
    print("Loading model from edsr-upscaling/models/EDSR_x{}.pb".format(scale))
    pbPath = "edsr-upscaling/models/EDSR_x{}.pb".format(scale)

    # Get model
    model = load_pb(pbPath)
    return model

def setup():
    # TODO: load and return all models (2,3,4)
    model = load_model_from_pb(4)

    # TO-DO: utilize GPU runtime
    # device = tf.device("cuda") if tf.cuda.is_available() else tf.device("cpu")
    # model = model.to(device)

    return {"model": model}

def run_upscale(image_path, model):
    mean = [103.1545782, 111.561547, 114.35629928]

    # below line could be redundant
    fullimg = cv2.imread(image_path, 3)
    floatimg = fullimg.astype(np.float32) - mean
    LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

    LR_tensor = model.get_tensor_by_name("IteratorGetNext:0")
    HR_tensor = model.get_tensor_by_name("NHWC_output:0")

    result = []
    with tf.Session(graph=model) as sess:
        print("Loading pb...")
        output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
        Y = output[0]
        HR_image = (Y + mean).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)

        result.append(Image.fromarray(HR_image))
        
        #cv2.imshow('EDSR upscaled image', HR_image)

    sess.close()
    return result

def main(opt):
    model  = setup(opt.self)
    scale = opt.scale
    raw_img = opt.raw_img

    run_upscale(scale,model,raw_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=str,
        nargs="?",
        default="4",
        help="the multiplier to upscale resolution to"
    )

    parser.add_argument(
        "--raw_img",
        type=str,
        nargs="?",
        help="raw binary data to read from"
    )

    main(parser.parse_args())
