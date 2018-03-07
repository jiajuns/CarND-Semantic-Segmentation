import os.path
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy.misc
from glob import glob

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input_tensor, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1 = tf.layers.conv2d(inputs=vgg_layer7_out, filters=num_classes,
                                kernel_size=1, strides=(1, 1), padding='same')

    de_conv1 = tf.layers.conv2d_transpose(inputs=conv_1x1, filters=num_classes,
                                          kernel_size=4, strides=(2, 2), padding='same')
    vgg_layer4_out = tf.layers.conv2d(inputs=vgg_layer4_out, filters=num_classes,
                                      kernel_size=1, strides=(1, 1), padding='same')
    de_conv1_added = tf.add(de_conv1, vgg_layer4_out)

    vgg_layer3_out = tf.layers.conv2d(inputs=vgg_layer3_out, filters=num_classes,
                                      kernel_size=1, strides=(1, 1), padding='same')
    de_conv2 = tf.layers.conv2d_transpose(inputs=de_conv1_added, filters=num_classes,
                                          kernel_size=4, strides=(2, 2), padding='same')
    de_conv2_added = tf.add(de_conv2, vgg_layer3_out)

    output = tf.layers.conv2d_transpose(inputs=de_conv2_added, filters=num_classes,
                                        kernel_size=16, strides=(8, 8), padding='same')

    # TODO: implement scaling layers

    return output



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=0.9,
                                          momentum=0.0,
                                          epsilon=1e-10)
    train_step = optimizer.minimize(loss)
    return logits, train_step, loss



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, logits,
             input_placeholder,label_placeholder, keep_prob, keep_prob_value, image_shape, num_classes):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print_every = 5
    iter_cnt = 1

    saver = tf.train.Saver()
    for e in range(epochs):
        losses = []
        best_accuracy = 0
        for train_data, train_label in get_batches_fn(batch_size):
            feed_dict = {
                input_placeholder: train_data,
                label_placeholder: train_label,
                keep_prob: keep_prob_value,
            }
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict)
            losses.append(loss)
            if (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g}".format(iter_cnt, loss))
            iter_cnt += 1
        accuracy = compute_accuracy(sess, input_placeholder, label_placeholder,
                                    num_classes, logits, keep_prob, image_shape)
        print('epoch {1} average accuracy {2:.3g}'.format(e, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = saver.save(sess, '/tmp/model.ckpt')
            print('Model saved in path {}'.format(save_path))

def compute_accuracy(sess, input_placeholder, label_placeholder, num_classes, logits, keep_prob, image_shape, data_folder=None):
    acc, _ = mean_iou(label_placeholder, tf.nn.softmax(logits), num_classes)
    list_accuracy = []

    # TODO: where to find validation dataset
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        accuracy = sess.run([acc], {keep_prob: 1.0, input_placeholder: [image]})
        list_accuracy.append(accuracy)
    return np.mean(list_accuracy)

def mean_iou(label_placeholder, prediction, num_classes):
    iou, iou_op = tf.metrics.mean_iou(label_placeholder, prediction, num_classes)
    return iou, iou_op

def train(learning_rate, epochs, batch_size, keep_prob_value, debug=False):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    if debug==True:
        tests.test_load_vgg(load_vgg, tf)
        tests.test_layers(layers)
        tests.test_optimize(optimize)
        tests.test_train_nn(train_nn)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network


        # TODO: Build NN using load_vgg, layers, and optimize function
        label_placeholder = tf.placeholder(tf.float32, [None, None, None, num_classes])
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, optimizer, loss = optimize(output, label_placeholder, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, loss, logits, input_image,
                 label_placeholder, keep_prob, keep_prob_value, image_shape, num_classes)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

def inference():
    # OPTIONAL: Apply the trained model to a video
    pass

if __name__ == '__main__':
    train(learning_rate=1e-3, epochs=6, batch_size=6, keep_prob_value=0.7, debug=False)
