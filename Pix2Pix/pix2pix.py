#coding=utf-8
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import tensorflow as tf
import numpy as np
import os
import glob
import random
import collections
import math
import time
# https://github.com/affinelayer/pix2pix-tensorflow
 
train_input_dir="D:/GAN/Pix2Pix/facades/train/"       # 训练集输入
train_output_dir="D:/GAN/Pix2Pix/facades/train_out/"  # 训练集输出
 
test_input_dir="D:/GAN/Pix2Pix/facades/val/"          # 测试集输入
test_output_dir="D:/GAN/Pix2Pix/facades/test_out/"    # 测试集的输出
checkpoint="D:/GAN/Pix2Pix/facades/train_out/"        # 保存结果的目录
 
seed=None
max_steps=None     # number of training steps (0 to disable)
max_epochs=200     # number of training epochs
 
progress_freq=50   # display progress every progress_freq steps
trace_freq=0       # trace execution every trace_freq steps
display_freq=50     # write current training images every display_freq steps
save_freq=500     # save model every save_freq steps, 0 to disable
 
separable_conv=False    # use separable convolutions in the generator
aspect_ratio=1.0        # aspect ratio of output images (width/height)
batch_size=1            # help="number of images in batch")
which_direction="BtoA"  # choices=["AtoB", "BtoA"])
ngf=64                  # help="number of generator filters in first conv layer")
ndf=64                  # help="number of discriminator filters in first conv layer")
scale_size=286          # help="scale images to this size before cropping to 256x256")
flip=True               # flip images horizontally
no_flip=True            # don't flip images horizontally
 
lr=0.0002        # initial learning rate for adam
beta1=0.5        # momentum term of adam
l1_weight=100.0  # weight on L1 term for generator gradient
gan_weight=1.0   # weight on GAN term for generator gradient
 
output_filetype="png"  # 输出图像的格式
 
EPS = 1e-12       # 极小数，防止梯度为损失为0
CROP_SIZE = 256   # 图片的裁剪大小
 
# 命名元组,用于存放加载的数据集合创建好的模型
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
 
# 图像预处理 [0, 1] => [-1, 1]
def preprocess(image):
    with tf.name_scope("preprocess"):        
        return image * 2 - 1
 
# 图像后处理[-1, 1] => [0, 1]
def deprocess(image):
    with tf.name_scope("deprocess"):        
        return (image + 1) / 2
 
 
# 判别器的卷积定义，batch_input为 [ batch , 256 , 256 , 6 ]
def discrim_conv(batch_input, out_channels, stride):
    # [ batch , 256 , 256 , 6 ] ===>[ batch , 258 , 258 , 6 ]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    '''
    [0,0]: 第一维batch大小不扩充
    [1,1]：第二维图像宽度左右各扩充一列，用0填充
    [1,1]：第三维图像高度上下各扩充一列，用0填充
    [0,0]：第四维图像通道不做扩充
    '''
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
 
 
# 生成器的卷积定义，卷积核为4*4，步长为2，输出图像为输入的一半
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
 
# 生成器的反卷积定义
def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
 
# 定义LReLu激活函数
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
 
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
 
# 批量归一化图像
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
 
# 检查图像的维度
def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)
 
    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")
 
    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image
 
 
# 去除文件的后缀，获取文件名
def get_name(path):
    # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。
    # os.path.splitext(),分离文件名与扩展名；默认返回(fname,fextension)元组
    name, _ = os.path.splitext(os.path.basename(path))
    return name
 
 
# 加载数据集，从文件读取-->解码-->归一化--->拆分为输入和目标-->像素转为[-1,1]-->转变形状
def load_examples(input_dir):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
 
    # 匹配第一个参数的路径中所有的符合条件的文件，并将其以list的形式返回。    
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))    
    
    # 图像解码器
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png
 
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
 
    # 如果文件名是数字，则用数字进行排序，否则用字母排序    
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
 
    sess = tf.Session()
 
    with tf.name_scope("load_images"):
        # 把我们需要的全部文件打包为一个tf内部的queue类型，之后tf开文件就从这个queue中取目录了，
        # 如果是训练模式时，shuffle为True
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
 
        # Read的输出将是一个文件名（key）和该文件的内容（value,每次读取一个文件，分多次读取）。
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
 
        # 对文件进行解码并且对图片作归一化处理
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32) # 归一化处理
 
        # 判断两个值知否相等，如果不等抛出异常
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        '''
          对于control_dependencies这个管理器，只有当里面的操作是一个op时，才会生效，也就是先执行传入的
        参数op，再执行里面的op。如果里面的操作不是定义的op，图中就不会形成一个节点，这样该管理器就失效了。
        tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中，这时control_dependencies就会生效.
        '''
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
 
        raw_input.set_shape([None, None, 3])
 
        # 图像值由[0,1]--->[-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])  # 256*256*3
        b_images = preprocess(raw_input[:,width//2:,:])  # 256*256*3
 
    # 这里的which_direction为：BtoA
    if which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")
 
    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)    
 
    # 图像预处理，翻转、改变形状
    with tf.name_scope("input_images"):
        input_images = transform(inputs)
    with tf.name_scope("target_images"):
        target_images = transform(targets)
 
    # 获得输入图像、目标图像的batch块
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))
 
    return Examples(
        paths=paths_batch,               # 输入的文件名块
        inputs=inputs_batch,             # 输入的图像块 
        targets=targets_batch,           # 目标图像块
        count=len(input_paths),          # 数据集的大小
        steps_per_epoch=steps_per_epoch, # batch的个数
    )
 
 
# 图像预处理，翻转、改变形状
def transform(image):
    r = image
    if flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
 
    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
 
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
    if scale_size > CROP_SIZE:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    elif scale_size < CROP_SIZE:
        raise Exception("scale size cannot be less than crop size")
    return r
 
 
#创建生成器，这是一个编码解码器的变种，输入输出均为：256*256*3, 像素值为[-1,1]
def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
 
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf) # ngf为第一个卷积层的卷积核核数量，默认为 64
        layers.append(output)
 
    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
 
    # 卷积的编码器
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # 对最后一层使用激活函数
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)
 
    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
 
    # 卷积的解码器
    num_encoder_layers = len(layers) # 8
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
 
            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)
 
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
 
            layers.append(output)
 
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)
 
    return layers[-1]
 
 
 
# 创建判别器，输入生成的图像和真实的图像：两个[batch,256,256,3],元素值值[-1,1]，输出:[batch,30,30,1],元素值为概率
def create_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []
 
    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)
 
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)
 
    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)
 
    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)
 
    return layers[-1]
 
# 创建Pix2Pix模型，inputs和targets形状为：[batch_size, height, width, channels]
def create_model(inputs, targets):
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)
 
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets) # 条件变量图像和真实图像
 
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs) # 条件变量图像和生成的图像
 
    # 判别器的损失，判别器希望V(G,D)尽可能大
    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
 
    # 生成器的损失，生成器希望V(G,D)尽可能小
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))  
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
 
    # 判别器训练
    with tf.name_scope("discriminator_train"):
        # 判别器需要优化的参数
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        # 优化器定义
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        # 计算损失函数对优化参数的梯度
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        # 更新该梯度所对应的参数的状态，返回一个op
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
 
    # 生成器训练
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            # 生成器需要优化的参数列表
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            # 定义优化器
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            # 计算需要优化的参数的梯度
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            # 更新该梯度所对应的参数的状态，返回一个op
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
 
    '''
      在采用随机梯度下降算法训练神经网络时，使用 tf.train.ExponentialMovingAverage 滑动平均操作的意义在于
    提高模型在测试数据上的健壮性（robustness）。tensorflow 下的 tf.train.ExponentialMovingAverage 需要
    提供一个衰减率（decay）。该衰减率用于控制模型更新的速度。该衰减率用于控制模型更新的速度，
    ExponentialMovingAverage 对每一个（待更新训练学习的）变量（variable）都会维护一个影子变量
    （shadow variable）。影子变量的初始值就是这个变量的初始值，
        shadow_variable=decay×shadow_variable+(1−decay)×variable
    '''
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
 
    # 
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
 
    return Model(
        predict_real=predict_real,  # 条件变量(输入图像)和真实图像之间的概率值，形状为；[batch,30,30,1]
        predict_fake=predict_fake,  # 条件变量(输入图像)和生成图像之间的概率值，形状为；[batch,30,30,1]
        discrim_loss=ema.average(discrim_loss),          # 判别器损失
        discrim_grads_and_vars=discrim_grads_and_vars,   # 判别器需要优化的参数和对应的梯度
        gen_loss_GAN=ema.average(gen_loss_GAN),          # 生成器的损失
        gen_loss_L1=ema.average(gen_loss_L1),            # 生成器的 L1损失
        gen_grads_and_vars=gen_grads_and_vars,           # 生成器需要优化的参数和对应的梯度
        outputs=outputs,                                 # 生成器生成的图片
        train=tf.group(update_losses, incr_global_step, gen_train),  # 打包需要run的操作op
    )
 
 
# 保存图像
def save_images(output_dir,fetches, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
 
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
 
 
# 将结果写入HTML网页
def append_index(output_dir,filesets, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
 
    for fileset in filesets:
        index.write("<tr>")
 
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
 
        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])
 
        index.write("</tr>")
    return index_path
 
 
# 转变图像的尺寸、并且将[0,1]--->[0,255]
def convert(image):
    if aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [CROP_SIZE, int(round(CROP_SIZE * aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
 
    # 将数据的类型转换为8位无符号整型  
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True) 
 
# 主函数
def train():
    # 设置随机数种子的值
    global seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
 
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    # 创建目录
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
 
 
    # 加载数据集，得到输入数据和目标数据并把范围变为 :[-1,1]
    examples = load_examples(train_input_dir)
    print("load successful ! examples count = %d" % examples.count)
 
    # 创建模型，inputs和targets是：[batch_size, height, width, channels]
    # 返回值：
    model = create_model(examples.inputs, examples.targets)
    print ("create model successful!")
 
 
    #图像处理[-1, 1] => [0, 1]
    inputs = deprocess(examples.inputs)  
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)
 
    # 把[0,1]的像素点转为RGB值：[0,255]
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
 
    # 对图像进行编码以便于保存
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            # tf.map_fn接受一个函数对象和集合，用函数对集合中每个元素分别处理
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
    
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
 
    # 只保存最新一个checkpoint
    saver = tf.train.Saver(max_to_keep=20)     
    
    init=tf.global_variables_initializer()
 
    with tf.Session() as sess:
        sess.run(init)
        print("parameter_count =", sess.run(parameter_count))
        if max_epochs is not None:
            max_steps = examples.steps_per_epoch * max_epochs   # 400X200=8000
        
        # 因为是从文件中读取数据，所以需要启动start_queue_runners()
        # 这个函数将会启动输入管道的线程，填充样本到队列中，以便出队操作可以从队列中拿到样本。            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)          
 
        # 运行训练集        
        print ("begin trainning......")
        print ("max_steps:",max_steps)           
        start = time.time()
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
            print ("step:",step) 
 
            # 定义一个需要run的所有操作的字典
            fetches = {
                "train": model.train              
            }
 
            # progress_freq为 50，每50次计算一次三个损失，显示进度
            if should(progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1
 
            # display_freq为 50，每50次保存一次输入、目标、输出的图像
            if should(display_freq):
                fetches["display"] = display_fetches
 
            # 运行各种操作，
            results = sess.run(fetches)            
 
            # display_freq为 50，每50次保存输入、目标、输出的图像
            if should(display_freq):
                print("saving display images")
                filesets = save_images(train_output_dir,results["display"], step=step)
                append_index(train_output_dir,filesets, step=True)  
 
            # progress_freq为 50，每50次打印一次三种损失的大小，显示进度
            if should(progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(step/examples.steps_per_epoch)
                train_step = (step - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * batch_size / (time.time() - start)
                remaining = (max_steps - step) * batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
 
            # save_freq为500，每500次保存一次模型
            if should(save_freq):
                print("saving model")
                saver.save(sess, os.path.join(train_output_dir, "model"), global_step=step)
 
# 测试
def test():
 # 设置随机数种子的值
    global seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
 
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    # 创建目录
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    if checkpoint is None:
        raise Exception("checkpoint required for test mode")
 
    # disable these features in test mode
    scale_size = CROP_SIZE
    flip = False
 
    # 加载数据集，得到输入数据和目标数据
    examples = load_examples(test_input_dir)
    print("load successful ! examples count = %d" % examples.count)
 
    # 创建模型，inputs和targets是：[batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)
    print ("create model successful!")
 
 
    #图像处理[-1, 1] => [0, 1]
    inputs = deprocess(examples.inputs)  
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)
 
    # 把[0,1]的像素点转为RGB值：[0,255]
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
     
 
    # 对图像进行编码以便于保存
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            # tf.map_fn接受一个函数对象和集合，用函数对集合中每个元素分别处理
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
 
    sess=tf.InteractiveSession()  
    saver = tf.train.Saver(max_to_keep=1)
 
    ckpt=tf.train.get_checkpoint_state(checkpoint)
    # saver.restore(sess, ckpt.model_checkpoint_path)
 
    start = time.time()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    for step in range(examples.count):
        results = sess.run(display_fetches)
        filesets = save_images(test_output_dir,results)
        for i, f in enumerate(filesets):
            print("evaluated image", f["name"])
        index_path = append_index(test_output_dir,filesets)
    print("wrote index at", index_path)
    print("rate", (time.time() - start) / max_steps)
 
 
 
if __name__ == '__main__':
    # test()
    train()