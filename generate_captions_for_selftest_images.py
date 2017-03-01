# This script generate captions for macoco test set 2014.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os, sys, pdb
from PIL import Image
import numpy as np
import json
import time
import pickle
import pdb

import tensorflow as tf

import configuration
from inference_utils import caption_generator
from inference_utils import vocabulary
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

# model selection
MODEL = "BNRHN"  # LSTM/RHN/BNRHN
if MODEL == 'RHN':
    import inference_wrapper_rhn
elif MODEL == 'BNRHN':
    import inference_wrapper_bnrhn
else:
    import inference_wrapper

# Note: Please confirm tht all the images in the dir is JPEG format before running this script.

# filename to id file
filename_to_id_file = "data/selftest_image_ids.txt"
# the directory of test images
image_path = '/cis/phd/cxz2081/data/mscoco/captioning/val2014/COCO_val2014_000000'
# Model checkpoint file or directory containing a model checkpoint file.
checkpoint_path = "model/train_bnrhn"
# Text file containing the vocabulary.
vocab_file = "data/mscoco/word_counts.txt"
# File to save the output dict in json format.
result_file = "results/captions_selftest2014_bnrhn_results.json"


tf.logging.set_verbosity(tf.logging.INFO)


def get_id(filename_to_id_file,image_path):
  ''' create filename-to-id map.

  Args:
    filename_to_id_file: a pickle file contains the list of ids.

  Returns:
    filename_to_id: a list of filename with path mapping to id.
  '''

  # load json file filename_to_idfile
  with open(filename_to_id_file, "rb") as fp:   # Unpickling
    filename_to_id = pickle.load(fp)

  new_dict = {}
  for i in filename_to_id:
      new_dict[image_path + str(i).zfill(6) + '.jpg'] = int(i)

  return new_dict

def generate_captions(filename_to_id, checkpoint_path, vocab_file, result_file):
  ''' Genarate captions for input images. Save the result dict in a json file.

  Args:
    filename_to_id_file: a json file contains the map from filename to id.
    id_to_filename_file: a json file contains the map from id to filename.
    check_point: model checkpoint file or directory containing a model checkpoint file. E.g., "model/train"
    vocab_file: text file containing the vocabulary. E.g., "data/mscoco/word_counts.txt"
    result_file: the file to save the output dict in json format. E.g., "results/mscoco_captioning_results.json"
  '''

  # convert ids to filenames
  file_name_list = filename_to_id.keys()
  print("Created filename list for %d images." % len(file_name_list))

  # Create the vocabulary.
  # vocab = vocabulary.Vocabulary(vocab_file)
  # with open('data/mscoco/vocab_MSCOCO.pkl','w') as f:
    # pickle.dump(vocab,f)
  # Load vocabulary.
  with open('data/mscoco/vocab_MSCOCO.pkl','r') as f:
    vocab = pickle.load(f)

  # Build the inference graph.
  print('Start graph construction.')
  g = tf.Graph()
  with g.as_default():
    # model = inference_wrapper.InferenceWrapper()
    if MODEL == 'RHN':
        model = inference_wrapper_rhn.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.RHNModelConfig(), checkpoint_path)
    elif MODEL == 'BNRHN':
        model = inference_wrapper_bnrhn.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.BNRHNModelConfig(), checkpoint_path)
    else:
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)
  g.finalize()
  print('Finish graph construction.')

  # filenames = []
  # for file_pattern in file_name_list:
    # filenames.extend(tf.gfile.Glob(file_pattern))
  filenames = file_name_list
  tf.logging.info("Running caption generation on %d files matching.", len(filenames))

  # generate captions
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    print('Start session initialization.')
    restore_fn(sess)
    print('Finish session initialization.')

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    captions_results = []
    counter = 1
    for filename in filenames:
        tStart = time.time()
        with tf.gfile.GFile(filename, "r") as f:
            image = f.read()
        #  pdb.set_trace()
        captions = generator.beam_search(sess, image)
#        print("Captions for image %s:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            if i == 0:
                # save the caption with highest score.
                captions_results.append({'image_id': filename_to_id[filename], 'caption':sentence})
#            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        tStop = time.time()
        print('Time cost for image %d/%d: %.2f' % (counter, len(filenames), tStop - tStart))
        counter += 1
  # save captions in json format, as a list of dict
  with open(result_file, 'w') as f:
    json.dump(captions_results, f)


def main(unused_argv):

  # Extract the filenames and ids.
  filename_to_id = get_id(filename_to_id_file, image_path)

  # generate captions for test images
  generate_captions(filename_to_id, checkpoint_path, vocab_file, result_file)


if __name__ == "__main__":
  tf.app.run()
