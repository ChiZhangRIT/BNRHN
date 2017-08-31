# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import pdb
import time

import tensorflow as tf

import configuration
import inference_wrapper_bnrhn
# import inference_wrapper_rhn
# import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "txt file including image names.")
tf.flags.DEFINE_string("output_file", "",
                       "file to save image captions.")

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper_bnrhn.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.BNRHNModelConfig(),
                                               FLAGS.checkpoint_path)
    # model = inference_wrapper_rhn.InferenceWrapper()
    # restore_fn = model.build_graph_from_config(configuration.RHNModelConfig(),
    #                                            FLAGS.checkpoint_path)
    # model = inference_wrapper.InferenceWrapper()
    # restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
    #                                            FLAGS.checkpoint_path)
  g.finalize()

  # # Create the vocabulary.
  # vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  # Load vocabulary.
  with open('data/mscoco/vocab_MSCOCO.pkl','r') as f:
      vocab = pickle.load(f)

  with open(FLAGS.input_files, 'r') as f:
      image_names = f.readlines()
  filenames = []
  for file_pattern in image_names:
      filenames.extend(tf.gfile.Glob(file_pattern.rstrip()))

  # filenames = []
  # # for file_pattern in image_names:
  # for file_pattern in FLAGS.input_files.split(","):
  #     filenames.extend(tf.gfile.Glob(file_pattern))
  #     pdb.set_trace()

  tf.logging.info("Running caption generation on %d files matching",
                  len(filenames))

  # generate captions
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    with open(FLAGS.output_file, 'w') as fout:

        start_time = time.time()
        for filename in filenames:
          with tf.gfile.GFile(filename, "r") as f:
              image = f.read()
          captions = generator.beam_search(sess, image)
          print("Generating captions for image %s:" % os.path.basename(filename))
          for i, caption in enumerate(captions):
              # Ignore begin and end words.
              sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
              sentence = " ".join(sentence)
            #   pdb.set_trace()
            #   print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
              fout.writelines(sentence + '\n')
        elapsed_time = time.time() - start_time
        print('elapsed_time: %f' % elapsed_time)

if __name__ == "__main__":
  tf.app.run()
