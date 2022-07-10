import argparse
import tensorflow as tf
import cv2
import numpy as np

# OBJECTIVE: break out into init and run, run should return raw bytes for delegant to write
def load_model_from_pb(self, scale):
    """
    This loads a .pb file.
    """
    # Read model
    print("Loading model from ./models/EDSR_x{}.pb".format(scale))
    pbPath = "./models/EDSR_x{}.pb".format(scale)

    # Get model
    model = self.load_pb(pbPath)
    return model

def setup(self):
    model = load_model_from_pb(self)

    # TO-DO: utilize GPU runtime
    # device = tf.device("cuda") if tf.cuda.is_available() else tf.device("cpu")
    # model = model.to(device)

    return {"model": model}

def run_upscale(self, model, raw_img):
    np_arr = np.fromstring(raw_img, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    # below line could be redundant
    # fullimg = cv2.imread(img_np, 3)
    floatimg = img_np.astype(np.float32) - self.mean
    LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

    LR_tensor = model.get_tensor_by_name("IteratorGetNext:0")
    HR_tensor = model.get_tensor_by_name("NHWC_output:0")

    with tf.Session(graph=model) as sess:
        print("Loading pb...")
        output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
        Y = output[0]
        HR_image = (Y + self.mean).clip(min=0, max=255)
        HR_image = (HR_image).astype(np.uint8)
        # convert to jpeg and save in variable
        img_bytes = cv2.imencode('.png', HR_image)[1].tobytes()
        
        #cv2.imshow('EDSR upscaled image', HR_image)

    sess.close()
    return img_bytes


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
