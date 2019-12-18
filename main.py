#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import io
from model import OpenNsfwModel, InputType
import flask
from PIL import Image
import numpy as np
import skimage
import skimage.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model_weights_path = 'data/open_nsfw-weights.npy'
model = OpenNsfwModel()

VGG_MEAN = [104, 117, 123]

img_width, img_height = 224, 224

app = flask.Flask(__name__)


# 将RGB按照BGR重新组装，然后对每一个RGB对应的值减去一定阈值
def prepare_image(image):
    H, W, _ = image.shape
    h, w = (img_width, img_height)

    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]

    image = image[:, :, :: -1]

    image = image.astype(np.float32, copy=False)
    image = image * 255.0
    image = image-np.array(VGG_MEAN, dtype=np.float32)

    image = np.expand_dims(image, axis=0)
    return image

# 使用TFLite文件检测
def getResultFromFilePathByTFLite(path):
    model_path = "./model/nsfw.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    # print(str(input_details))
    output_details = interpreter.get_output_details()
    # print(str(output_details))

    im = Image.open(path)
    # im = Image.open(r"./images/image1.png")
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize((256, 256), resample=Image.BILINEAR)
    fh_im = io.BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)

    image = (skimage.img_as_float(skimage.io.imread(fh_im, as_grey=False))
             .astype(np.float32))

    # 填装数据
    final = prepare_image(image)
    interpreter.set_tensor(input_details[0]['index'], final)

    # 调用模型
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 出来的结果去掉没用的维度
    result = np.squeeze(output_data)
    print('TFLite-->>result:{},path:{}'.format(result, path))
    print(
        "==========================================================================================================")
    print("")
    print("")

def getResultFromFilePathByPyModle(path):
    # print("numpy-version:" + np.__version__)
    # print("tensorflow-version:" + tf.__version__)
    im = Image.open(path)

    if im.mode != "RGB":
        im = im.convert('RGB')

    # print("图片reSize：256*256")
    imr = im.resize((256, 256), resample=Image.BILINEAR)

    fh_im = io.BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)

    image = (skimage.img_as_float(skimage.io.imread(fh_im, as_grey=False))
             .astype(np.float32))

    final = prepare_image(image)

    tf.reset_default_graph()
    with tf.Session() as sess:
        input_type = InputType[InputType.TENSOR.name.upper()]
        model.build(weights_path=model_weights_path, input_type=input_type)
        sess.run(tf.global_variables_initializer())

        predictions = sess.run(model.predictions, feed_dict={model.input: final})
        # print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
        print(
            "==========================================================================================================")
        print('Python-->>result:{},path:{}'.format(predictions[0], path))


def getResultListFromDir():
    list = os.listdir("/Users/zhaowenwen/Downloads/testImages")
    for i in range(0, len(list)):
        if (list[i] != ".DS_Store" and list[i] != ".localized"):
            getResultFromFilePathByPyModle(os.path.join("/Users/zhaowenwen/Downloads/testImages", list[i]))
            getResultFromFilePathByTFLite(os.path.join("/Users/zhaowenwen/Downloads/testImages", list[i]))


# 代码生成tflite文件
def createTfliteFile():
    in_path = "./model/frozen_nsfw.pb"
    out_path = "./model/nsfw.tflite"

    # 模型输入节点
    input_tensor_name = ["input"]
    input_tensor_shape = {"input":[1, 224,224,3]}
    # 模型输出节点
    classes_tensor_name = ["predictions"]

    converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                                input_tensor_name, classes_tensor_name,
                                                input_shapes = input_tensor_shape)
    # converter.post_training_quantize = True
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

#生成.pb .index  .meta .ckpt.data文件
# freeze_graph --input_graph=/Users/jason/nsfw/flask-open-nsfw/model/nsfw-graph.pb --input_checkpoint=/Users/jason/nsfw/flask-open-nsfw/model/nsfw_model.ckpt --input_binary=true --output_graph=/Users/jason/nsfw/flask-open-nsfw/model/frozen_nsfw.pb --output_node_names=predictions

#命令行生成tflite文件(tonserflow1.3前可用)
#toco --graph_def_file=/Users/jason/nsfw/flask-open-nsfw/model/frozen_nsfw.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=/Users/jason/nsfw/flask-open-nsfw/model/nsfw.tflite --inference_type=FLOAT --input_type=FLOAT --input_arrays=input --output_arrays=predictions input_shapes=1,224,224,3


#
# toco \
#   --graph_def_file=/Users/jason/nsfw/flask-open-nsfw/model/frozen_nsfw.pb \
#   --output_file=/Users/jason/nsfw/flask-open-nsfw/model/aaaa.lite \
#   --input_format=TENSORFLOW_GRAPHDEF \
#   --output_format=TFLITE \
#   --input_shape=1,224,224,3 \
#   --input_array=input \
#   --output_array=predictions \
#   --inference_type=FLOAT \
#   --input_data_type=FLOAT


if __name__ == "__main__":
    #检测加载Downloads下所有文件，逐个输出检测结果
        getResultListFromDir()
    #生成TFLite文件
        # createTfliteFile()
        # print('tensorflowVersion:',tf.__version__)
        # print('npVersion:',np.__version__)