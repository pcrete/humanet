# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tqdm import tqdm

def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=None, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_images=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)

            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image


def _create_image_encoder(preprocess_fn, factory_fn, image_shape, batch_size=32,
                         session=None, checkpoint_path=None,
                         loss_mode="cosine"):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)

    preprocessed_image_var = tf.map_fn(
        lambda x: preprocess_fn(x, is_training=False),
        tf.cast(image_var, tf.float32))

    l2_normalize = loss_mode == "cosine"
    feature_var, _ = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
    feature_dim = feature_var.get_shape().as_list()[-1]

    if session is None:
        session = tf.Session()
    if checkpoint_path is not None:
        slim.get_or_create_global_step()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)

    def encoder(data_x):
        out = np.zeros((len(data_x), feature_dim), np.float32)
        _run_in_batches(
            lambda x: session.run(feature_var, feed_dict=x),
            {image_var: data_x}, out, batch_size)
        return out

    return encoder


def create_image_encoder(model_filename, batch_size=32, loss_mode="cosine",
                         session=None):
    image_shape = 128, 64, 3
    factory_fn = _network_factory(num_classes=1501, is_training=False, weight_decay=1e-8)

    return _create_image_encoder(_preprocess, factory_fn, image_shape, batch_size, session,
        model_filename, loss_mode)


def create_box_encoder(model_filename, batch_size=32, loss_mode="cosine"):
    image_shape = 128, 64, 3
    image_encoder = create_image_encoder(model_filename, batch_size, loss_mode)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches)

    return encoder


def generate_detections(encoder, video_dir, det_dir, feat_dir):    
    videos = os.listdir(args.video_dir)
    videos.sort()
    for video_name in videos:

        if(video_name != args.test_video and args.test_video != '' ): 
            continue

        print("Processing %s" % video_name)

        detection_file = os.path.join(det_dir, video_name[:-3]+'csv')
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in tqdm(range(min_frame_idx, max_frame_idx + 1)):
            # print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]
            ret, bgr_image = cap.read()
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        feature_filename = os.path.join(feat_dir, "%s.npy" % video_name[:-4])
        np.save(feature_filename, np.asarray(detections_out), allow_pickle=False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.ckpt-68577",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--loss_mode", default="cosine", help="Network loss training mode"
    )
    parser.add_argument(
        "--test_video", 
        help="To run specific one", 
        default=''
    )
    parser.add_argument(
        "--video_dir", 
        help="Video directory.", 
        default="../dataset/videos"
    )
    parser.add_argument(
        "--det_dir", help="Path to detection directory",
        default='../dataset/detections'
    )
    parser.add_argument(
        "--feat_dir", 
        help="Features directory.",
        default="../dataset/features"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    f = create_box_encoder(args.model, batch_size=32, loss_mode=args.loss_mode)
    generate_detections(f, args.video_dir, args.det_dir, args.feat_dir)