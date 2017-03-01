'''
This script generates the metric scores for the captioning results.
The captioning results and the corresponding GT are required.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from PIL import Image
import numpy as np
import json
import time
import pickle

import tensorflow as tf

import configuration
# import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap


# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_boolean("image_id_from_tfrecords", False,
#                         "Whether to retrieve image id from TFRecords. If not, a text file containing image ids is required.")
#
#
# # TFRecords file that contains image/caption info (required if FLAG.image_id_from_tfrecords == True)
# tfrecords_file_names = ['data/mscoco/test-00000-of-00008',
# 	                'data/mscoco/test-00001-of-00008',
#           	        'data/mscoco/test-00002-of-00008',
# 	                'data/mscoco/test-00003-of-00008',
#           	        'data/mscoco/test-00004-of-00008',
# 	                'data/mscoco/test-00005-of-00008',
#           	        'data/mscoco/test-00006-of-00008',
# 	                'data/mscoco/test-00007-of-00008']
# # image id file (required if FLAG.image_id_from_tfrecords == False)
# image_id_file = "data/selftest_image_ids.txt"
# # Model checkpoint file or directory containing a model checkpoint file.
# checkpoint_path = "model/train"
# # Text file containing the vocabulary.
# vocab_file = "data/mscoco/word_counts.txt"
# JSON file containing caption annotations.
captions_file = "/cis/phd/cxz2081/data/mscoco/captioning/annotations/captions_val2014_addtype.json"
# File to save the output dict in json format.
result_file = "results/captions_selftest2014_bnrhn_results.json"
# Evaluation result: Img
evalImgsFile = "results/captions_selftest2014_bnrhn_evalImgs.json"
# Evaluation result
evalFile = "results/captions_selftest2014_bnrhn_eval.json"


tf.logging.set_verbosity(tf.logging.INFO)


# def parse_tfrecords(file_names):
#   ''' Parse TFRecords files into image ids.
#
#   Args:
#     file_names: a list of file names of TFRecords. E.g.,
#              ['mscoco/test-00000-of-00008', 'mscoco/test-00001-of-00008']
#
#   Returns:
#     image_ids: a list of intergers indicating the ids of test images.
#   '''
#
#   start_time = time.time()
#
#   image_ids = []
#   for file_name in file_names:
#     data = []
#     image_feature_name = "image/image_id"
#     caption_feature_name = "image/caption_ids"
# #     DEBUG_counter = 0  # DEBUG_mode
#     for s_example in tf.python_io.tf_record_iterator(file_name):
#         context, sequence = tf.parse_single_sequence_example(
#                             s_example,
#                             context_features={
#                                                image_feature_name: tf.VarLenFeature(dtype=tf.int64)
#                                              },
#                             sequence_features={
#                                                caption_feature_name: tf.FixedLenSequenceFeature([], dtype=tf.int64)
#                                               })
#         data.append(context[image_feature_name].values)
#
# #         # DEBUG_mode
# #         DEBUG_counter += 1
# #         if DEBUG_counter == 3:
# #             break
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         single_file_id_list = sess.run(data)
#
#     sess.close()
#     tf.reset_default_graph()
#
#     single_file_id_list = map(np.int64, single_file_id_list)
#     image_ids += single_file_id_list
#
#   elapsed_time = time.time() - start_time
#   print('Extracting image_ids completed. (Elapsed time = %.2f s)' % elapsed_time)
#   print('%d image ids are extracted' % len(np.unique(image_ids)))
#
#   return list(set(image_ids))
#
#
# def load_image_ids(image_id_file):
#   ''' Load image ids from file as a list.
#
#   Args:
#     image_id_file: a (pickled) txt file containing image ids.
#
#   Returns:
#     image_ids: a list of image ids.
#   '''
#
#   print("Loading Image ids...")
#
#   # fetch image_id
#   with open(image_id_file, "rb") as fp:   # Unpickling
#     image_ids = pickle.load(fp)
#
#   print("%d image ids are loaded." % len(image_ids))
#   return image_ids
#
#
# def generate_captions(image_ids, checkpoint_path, vocab_file, captions_file, result_file):
#   ''' Genarate captions for input images. Save the result dict in a json file.
#
#   Args:
#     image_ids: a list of image ids (int) in mscoco dataset.
#     check_point: model checkpoint file or directory containing a model checkpoint file. E.g., "model/train"
#     vocab_file: text file containing the vocabulary. E.g., "data/mscoco/word_counts.txt"
#     captions_file: JSON file containing caption annotations. E.g., "/cis/phd/cxz2081/data/mscoco/captioning/annotations/captions_val2014.json"
#     result_file: the file to save the output dict in json format. E.g., "results/mscoco_captioning_results.json"
#   '''
#
#   # Extract the filenames and ids.
#   with tf.gfile.FastGFile(captions_file, "r") as f:
#     caption_data = json.load(f)
#   filename_to_id = {"/cis/phd/cxz2081/data/mscoco/captioning/val2014/" + x["file_name"]: x["id"] for x in caption_data["images"]}
#   id_to_filename = {x["id"]: "/cis/phd/cxz2081/data/mscoco/captioning/val2014/" + x["file_name"] for x in caption_data["images"]}
#
#   # convert ids to filenames
#   file_name_list = [id_to_filename[x] for x in image_ids]
#
#   # Build the inference graph.
#   g = tf.Graph()
#   with g.as_default():
#     model = inference_wrapper.InferenceWrapper()
#     restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)
#   g.finalize()
#
#   # Create the vocabulary.
#   vocab = vocabulary.Vocabulary(vocab_file)
#
#   filenames = []
#   for file_pattern in file_name_list:
#     filenames.extend(tf.gfile.Glob(file_pattern))
#   tf.logging.info("Running caption generation on %d files matching.", len(filenames))
#
#   # generate captions
#   with tf.Session(graph=g) as sess:
#     # Load the model from checkpoint.
#     restore_fn(sess)
#
#     # Prepare the caption generator. Here we are implicitly using the default
#     # beam search parameters. See caption_generator.py for a description of the
#     # available beam search parameters.
#     generator = caption_generator.CaptionGenerator(model, vocab)
#
#     captions_results = []
#     for filename in filenames:
#         with tf.gfile.GFile(filename, "r") as f:
#             image = f.read()
#         captions = generator.beam_search(sess, image)
# #        print("Captions for image %s:" % os.path.basename(filename))
#         for i, caption in enumerate(captions):
#             # Ignore begin and end words.
#             sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
#             sentence = " ".join(sentence)
#             if i == 0:
#                 # save the caption with highest score.
#                 captions_results.append({'image_id': filename_to_id[filename], 'caption':sentence})
# #            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
#
#   # save captions in json format, as a list of dict
#   with open(result_file, 'w') as f:
#     json.dump(captions_results, f)


def calculate_metrics(annFile, resFile, evalImgsFile, evalFile):
  ''' Calculate CIDEr, BLEU, ROUGE, METEOR score.

  Args:
    annFile: Annotation file of mscoco captioning dataset. JSON format. Should have attribute "type in it."
    resFile: The generated captions for test images (to be evaluated). JSON format
    evalImgsFile: json file containing a list of metric scores and image id for each individual.
    evalFile: json file to save metric scores.

  '''
  # create coco object and cocoRes object
  coco = COCO(annFile)
  cocoRes = coco.loadRes(resFile)

  # create cocoEval object by taking coco and cocoRes
  cocoEval = COCOEvalCap(coco, cocoRes)

  # evaluate on a subset of images by setting
  # cocoEval.params['image_id'] = cocoRes.getImgIds()
  # please remove this line when evaluating the full validation set
  cocoEval.params['image_id'] = cocoRes.getImgIds()

  # evaluate results
  cocoEval.evaluate()

  # print output evaluation scores
  print("=================================")
  for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))

  # save evaluation results to ./results folder
  json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
  json.dump(cocoEval.eval,     open(evalFile, 'w'))


def main(unused_argv):

  # if FLAGS.image_id_from_tfrecords:
  #   assert tfrecords_file_names, "--tfrecords_file_names is required."
  #   # extract image ids from TFrecords
  #   image_ids = parse_tfrecords(tfrecords_file_names)
  # else:
  #   assert image_id_file, "--image_id_file is required."
  #   # load image ids from file
  #   image_ids = load_image_ids(image_id_file)
  #
  # # generate captions for test images
  # generate_captions(image_ids, checkpoint_path, vocab_file, captions_file, result_file)

  # calculate metrics
  calculate_metrics(captions_file, result_file, evalImgsFile, evalFile)


if __name__ == "__main__":
  tf.app.run()
