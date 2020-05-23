from PIL import Image

import time

import os

import numpy as np

from skimage import io

# 导入model中的函数
from model0 import *

from layers import *

to_train = True  # 设置为true进行训练
to_test = False  # 不进行test
to_restore = False  # 不存储
output_path = "./output"  # 设置输出文件路径
check_dir = "./output/checkpoints/"  # 输出模型参数的文件路径
data_dir = "./vangogh2photo"  # 数据的根目录

temp_check = 0

max_epoch = 1
max_images = 100
h1_size = 150
h2_size = 300
z_size = 100
sample_size = 10
save_training_images = True




# 读取数据到内存当中
def get_data(input_dir, floderA, floderB):
    '''
    函数功能：输入根路径，和不同数据的文件夹，读取数据
    :param input_dir:根目录的参数
    :param floderA: 数据集A所在的文件夹名
    :param floderB: 数据集B所在的文件夹名
    :return: 返回读取好的数据，train_set_A即A文件夹的数据, train_set_B即B文件夹的数据
    '''

    # 读取路径，并判断路径下有多少张影像
    imagesA = os.listdir(input_dir + floderA)
    imagesB = os.listdir(input_dir + floderB)
    imageA_len = len(imagesA)
    imageB_len = len(imagesB)

    # 定义用于存放读取影像的变量
    dataA = np.empty((imageA_len, image_width, image_height, image_channel), dtype="float32")
    dataB = np.empty((imageB_len, image_width, image_height, image_channel), dtype="float32")

    # 读取文件夹A中的数据
    for i in range(imageA_len):
        # 逐个影像读取
        img = Image.open(input_dir + floderA + "/" + imagesA[i])
        img = img.resize((image_width, image_height))
        arr = np.asarray(img, dtype="float32")
        # 对影像数据进行归一化[-1, 1]，并将结果保存到变量中
        dataA[i, :, :, :] = arr * 1.0 / 127.5 - 1.0

    # 读取文件夹B中的数据
    for i in range(imageB_len):
        # 逐个影像读取
        img = Image.open(input_dir + floderB + "/" + imagesB[i])
        img = img.resize((image_width, image_height))
        arr = np.asarray(img, dtype="float32")
        # 对影像数据进行归一化[-1, 1]，并将结果保存到变量中
        dataB[i, :, :, :] = arr * 1.0 / 127.5 - 1.0

    # 随机打乱图像的顺序，当然也可以选择不打乱
    np.random.shuffle(dataA)
    np.random.shuffle(dataB)

    # 执行tensor
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        # 如果输入图像不是 256 * 256， 最好执行reshape
        dataA = tf.reshape(dataA, [-1, image_width, image_height, image_channel])
        dataB = tf.reshape(dataB, [-1, image_width, image_height, image_channel])

        train_set_A = sess.run(dataA)
        train_set_B = sess.run(dataB)

    return train_set_A, train_set_B


# 定义训练过程
def train():
    # 读取数据
    data_A, data_B = get_data(data_dir, "/trainA", "/trainB")

    # CycleGAN的模型构建 ----------------------------------------------------------
    # 输入数据的占位符
    input_A = tf.placeholder(tf.float32, [batch_size, image_width, image_height, image_channel], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, image_width, image_height, image_channel], name="input_B")

    fake_pool_A = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name="fake_pool_B")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    num_fake_inputs = 0

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    # 建立生成器和判别器
    with tf.variable_scope("Model") as scope:
        fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
        fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
        rec_A = build_gen_discriminator(input_A, "d_A")
        rec_B = build_gen_discriminator(input_B, "d_B")

        scope.reuse_variables()

        fake_rec_A = build_gen_discriminator(fake_A, "d_A")
        fake_rec_B = build_gen_discriminator(fake_B, "d_B")
        cyc_A = build_generator_resnet_9blocks(fake_B, "g_B")
        cyc_B = build_generator_resnet_9blocks(fake_A, "g_A")

        scope.reuse_variables()

        fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
        fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")

    # 定义损失函数
    cyc_loss = tf.reduce_mean(tf.abs(input_A - cyc_A)) + tf.reduce_mean(tf.abs(input_B - cyc_B))

    disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A, 1))
    disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B, 1))

    g_loss_A = cyc_loss * 10 + disc_loss_B
    g_loss_B = cyc_loss * 10 + disc_loss_A

    d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(
        tf.squared_difference(rec_A, 1))) / 2.0
    d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(
        tf.squared_difference(rec_B, 1))) / 2.0

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)

    model_vars = tf.trainable_variables()

    d_A_vars = [var for var in model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in model_vars if 'g_B' in var.name]

    d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
    g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
    g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

    for var in model_vars: print(var.name)

    # Summary variables for tensorboard

    g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
    g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
    d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
    d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
    # 模型构建完毕-------------------------------------------------------------------

    # 生成结果的存储器
    fake_images_A = np.zeros((pool_size, 1, image_height, image_width, image_channel))
    fake_images_B = np.zeros((pool_size, 1, image_height, image_width, image_channel))

    # 全局变量初始化
    init = tf.global_variables_initializer()
    # 结果保存器
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # #断点训练
        # saver.restore(sess, './output/checkpoints/' + "cyclegan-" + str(104))
        sess.run(init)

        writer = tf.summary.FileWriter("./output/2")

        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        # 开始训练
        for epoch in range(sess.run(global_step), 200):
            print("In the epoch ", epoch)
            saver.save(sess, os.path.join(check_dir, "cyclegan"), global_step=epoch)

            # 按照训练的epoch调整学习率。更高级的写法可参考：
            # lr = lr if epoch < epoch_step else adjust_rate * ((epochs - epoch) / (epochs - epoch_step))
            if (epoch < 100):
                curr_lr = 0.0002
            else:
                curr_lr = 0.0002 - 0.0002 * (epoch - 100) / 100

            # 保存图像-----------------------------------------------------------------
            if (save_training_images):
                # 检查路径是否存在
                if not os.path.exists("./output/imgs"):
                    os.makedirs("./output/imgs")

                # 保存10张影像
                for i in range(0, 10):
                    fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                        [fake_A, fake_B, cyc_A, cyc_B],
                        feed_dict={input_A: np.reshape(data_A[i], [-1, 256, 256, 3]),
                                   input_B: np.reshape(data_B[i], [-1, 256, 256, 3])})
                    # fake表示输入A，通过B的特征而变成B
                    io.imsave("./output/imgs/fakeB_" + str(epoch) + "_" + str(i) + ".jpg",
                              ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                    io.imsave("./output/imgs/fakeA_" + str(epoch) + "_" + str(i) + ".jpg",
                              ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
                    # cyc表示输入A，通过B的特征变成B，再由A的特征变成A结果
                    io.imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i) + ".jpg",
                              ((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
                    io.imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i) + ".jpg",
                              ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))

            # 保存图像结束------------------------------------------------------------

            # 循环执行cycleGAN
            for ptr in range(0, max_images):
                print("In the iteration ", ptr)

                # Optimizing the G_A network
                _, fake_B_temp, summary_str = sess.run([g_A_trainer, fake_B, g_A_loss_summ],
                                                       feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                                  input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                                  lr: curr_lr})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)

                # Optimizing the D_B network
                _, summary_str = sess.run([d_B_trainer, d_B_loss_summ],
                                          feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                     input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                     lr: curr_lr,
                                                     fake_pool_B: fake_B_temp1})
                writer.add_summary(summary_str, epoch * max_images + ptr)

                # Optimizing the G_B network
                _, fake_A_temp, summary_str = sess.run([g_B_trainer, fake_A, g_B_loss_summ],
                                                       feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                                  input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                                  lr: curr_lr})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)

                # Optimizing the D_A network
                _, summary_str = sess.run([d_A_trainer, d_A_loss_summ],
                                          feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                     input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                     lr: curr_lr,
                                                     fake_pool_A: fake_A_temp1})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                num_fake_inputs += 1

            sess.run(tf.assign(global_step, epoch + 1))

        writer.add_graph(sess.graph)


if __name__ == '__main__':
    if to_train:
        train()
