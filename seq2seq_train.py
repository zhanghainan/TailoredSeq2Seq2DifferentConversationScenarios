# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/pdf/1412.2007v2.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import nltk
import math
import os
import random
import sys
import time
import pickle

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model
import seq2seq_model
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 200, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size",5000, "Query vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./OpenSubData/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./OpenSubData/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_string("en_cover_dict_path", "./OpenSubData/query_cover_dict", "en_cover_dict_path")
tf.app.flags.DEFINE_string("ff_cover_dict_path", "./OpenSubData/answer_cover_dict", "ff_cover_dict_path")
tf.app.flags.DEFINE_float("fixed_rate", 0., "The scala of fixed set in batch train set")
tf.app.flags.DEFINE_string("fixed_set_path", "./OpenSubData/", "")
tf.app.flags.DEFINE_float("weibo_rate", 0., "The scala of weibo set in batch train set")
tf.app.flags.DEFINE_string("weibo_set_path", "./OpenSubData/", "")
tf.app.flags.DEFINE_float("qa_rate", 0.0, "The scala of qa set in batch train set")
tf.app.flags.DEFINE_string("qa_set_path", "./OpenSubData", "")
tf.app.flags.DEFINE_boolean("reinforce_learning", False, "Turn on/off reinforce_learning")
tf.app.flags.DEFINE_boolean("beam_size", 20, "beam size")
tf.app.flags.DEFINE_boolean("beam_search", False, "Turn on/off beam search")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS
if FLAGS.reinforce_learning:
  import data0_utils as du
  config = {}
  config['fill_word'] = du._PAD_
  config['embedding'] = du.embedding
  config['fold'] =1
  config['model_file']="model_mp"
  config['log_file']="dis.log"
  config['train_iters']= 50000
  config['model_tag']= "mxnet"
  config['batch_size'] = 64
  config['data1_maxlen'] = 46
  config['data2_maxlen'] = 74
  config['data1_psize'] =5
  config['data2_psize'] =5

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]

def read_data(source_path, target_path, min_size=None,max_size=None):
  data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    with gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < min_size):
          counter +=1
          source, target = source_file.readline(), target_file.readline()
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets): #[bucket_id, (source_size, target_size)]
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, dummy_set,forward_only,beam_search=False,beam_size=1):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, FLAGS.vocab_size, _buckets, dummy_set,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.fixed_rate, FLAGS.weibo_rate, FLAGS.qa_rate,
      forward_only=forward_only,
      beam_search = beam_search,
      beam_size=beam_size)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def train():
  """Train a en->fr translation model using WMT data."""
  #with tf.device("/gpu:0"):
  # Prepare WMT data.
  train_path = os.path.join(FLAGS.data_dir, "weibo")
  fixed_path = os.path.join(FLAGS.data_dir, "fixed")
  weibo_path = os.path.join(FLAGS.data_dir, "wb")
  qa_path = os.path.join(FLAGS.data_dir, "qa")

  voc_file_path = [train_path+".answer", fixed_path+".answer", weibo_path+".answer", qa_path+".answer",
                     train_path+".query", fixed_path+".query", weibo_path+".query", qa_path+".query"]

  vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.txt" % FLAGS.vocab_size)

  data_utils.create_vocabulary(vocab_path, voc_file_path, FLAGS.vocab_size)

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  print(len(vocab))
  print("Preparing Chitchat data in %s" % FLAGS.data_dir)
  train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
      FLAGS.data_dir, vocab, FLAGS.vocab_size)

  print("Preparing Fixed data in %s" % FLAGS.fixed_set_path)
  fixed_path = os.path.join(FLAGS.fixed_set_path, "fixed")
  fixed_query , fixed_answer = data_utils.prepare_defined_data(fixed_path, vocab, FLAGS.vocab_size)

  print("Preparing Weibo data in %s" % FLAGS.weibo_set_path)
  weibo_path = os.path.join(FLAGS.weibo_set_path, "wb")
  weibo_query, weibo_answer = data_utils.prepare_defined_data(weibo_path, vocab, FLAGS.vocab_size)

  print("Preparing QA data in %s" % FLAGS.qa_set_path)
  qa_path = os.path.join(FLAGS.qa_set_path, "qa")
  qa_query, qa_answer = data_utils.prepare_defined_data(qa_path, vocab, FLAGS.vocab_size)

  dummy_path = os.path.join(FLAGS.data_dir, "dummy")
  dummy_set = data_utils.get_dummy_set(dummy_path, vocab, FLAGS.vocab_size)
  print("Get Dummy Set : ", dummy_set)
  if FLAGS.reinforce_learning == True:
    from importlib import import_module
    mo = import_module(config['model_file'])
    disModel = mo.Model(config)
    disSess = tf.Session()
    disModel.init_step(disSess)

  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  #outputFile = open("RL_ouput.txt","w")
  if sys.argv[1] !="no":
    disModel.saver.restore(disSess, sys.argv[1])
  with tf.Session(config = tfconfig) as sess:
#with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, dummy_set, False)
        #sess.run(model.learning_rate_set_op)
        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        lofFile=open("./OpenSubData/log"+str(model.global_step.eval())+".txt","w")
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        en_dict_cover = {}
        fr_dict_cover = {}
        if model.global_step.eval() > FLAGS.steps_per_checkpoint:
            try:
                with open(FLAGS.en_cover_dict_path, "rb") as ef:
                    en_dict_cover = pickle.load(ef)
                    # for line in ef.readlines():
                    #     line = line.strip()
                    #     key, value = line.strip(",")
                    #     en_dict_cover[int(key)]=int(value)
            except Exception:
                print("no find query_cover_file")
            try:
                with open(FLAGS.ff_cover_dict_path, "rb") as ff:
                    fr_dict_cover = pickle.load(ff)
                    # for line in ff.readlines():
                    #     line = line.strip()
                    #     key, value = line.strip(",")
                    #     fr_dict_cover[int(key)]=int(value)
            except Exception:
                print("no find answer_cover_file")

        step_loss_summary = tf.Summary()
        #merge = tf.merge_all_summaries()
        #writer = tf.summary.FileWriter("./logs/", sess.graph)

        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          for ind in range(17):
            dev_set = read_data(dev_query, dev_answer,0,100000)
            train_set = read_data(train_query, train_answer, ind*100000,(ind+1)*100000)
            fixed_set = read_data(fixed_query, fixed_answer, FLAGS.max_train_data_size)
            weibo_set = read_data(weibo_query, weibo_answer, FLAGS.max_train_data_size)
            qa_set = read_data(qa_query, qa_answer, FLAGS.max_train_data_size)

            train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
            train_total_size = float(sum(train_bucket_sizes))
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
            for kk in range(500):
              random_number_01 = np.random.random_sample()
              bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])
              batch_num=1+int(len(train_set[bucket_id])/FLAGS.batch_size)
              # Get a batch and make a step.
              for mm in range(batch_num):
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                  train_set, bucket_id, 0, fixed_set, weibo_set, qa_set,mm*FLAGS.batch_size)
                #resp_b = model.ids2tokens(batch_source_decoder,rev_vocab)
                #resp_a = model.ids2tokens(batch_source_encoder,rev_vocab)
                #for ii in range(len(resp_a)):
                #  if ii%10==0:
                #    print ("")
                #  print (' '.join(w for w in resp_a[0]))
                #  print (' '.join(w for w in resp_b[ii]))

                if FLAGS.reinforce_learning:
                  _, step_loss, _ = model.step_rl(sess, _buckets, encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder, bucket_id,rev_vocab=rev_vocab, disSession=disSess,disModel=disModel)
                else:
                  _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, forward_only=False, force_dec_input=True)
                #print(step_loss)
                lossmean=0.
                for ii in step_loss:
                  lossmean=lossmean+ii
                lossmean=lossmean/len(step_loss)

                loss +=lossmean / FLAGS.steps_per_checkpoint
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                current_step += 1

                query_size, answer_size = _buckets[bucket_id]
                for batch_index in xrange(FLAGS.batch_size):
                  for query_index in xrange(query_size):
                    query_word = encoder_inputs[query_index][batch_index]
                    if en_dict_cover.has_key(query_word):
                      en_dict_cover[query_word] += 1
                    else:
                      en_dict_cover[query_word] = 0

                  for answer_index in xrange(answer_size):
                    answer_word = decoder_inputs[answer_index][batch_index]
                    if fr_dict_cover.has_key(answer_word):
                      fr_dict_cover[answer_word] += 1
                    else:
                      fr_dict_cover[answer_word] = 0

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                  outputFile = open("OpenSubData/RL_"+str(model.global_step.eval())+".txt","w")
                  bucket_value = step_loss_summary.value.add()
                  bucket_value.tag = "loss"
                  bucket_value.simple_value = float(loss)
                  #writer.add_summary(step_loss_summary, current_step)

                  print ("query_dict_cover_num: %s" %(str(en_dict_cover.__len__())))
                  print ("answer_dict_cover_num: %s" %(str(fr_dict_cover.__len__())))

                  ef = open(FLAGS.en_cover_dict_path, "wb")
                  pickle.dump(en_dict_cover, ef)
                  ff = open(FLAGS.ff_cover_dict_path, "wb")
                  pickle.dump(fr_dict_cover, ff)
                  num =0
                  pick=0.
                  mmm=1
                  eval_loss=0
                  dictt={}
                  dictt_b={}
                  eval_ppl=0.
                  eval_num=0
                  for idd in range(2):
                    bucket_id=idd+2
                    batch_num = 1+int(len(dev_set[bucket_id])/FLAGS.batch_size)
                    for mm in range(batch_num):
                      #encoder_inputs, decoder_inputs, target_weights, batch_source_encoder,batch_source_decoder = model.get_batch(train_set, bucket_id, 0, fixed_set, weibo_set, qa_set,mm*FLAGS.batch_size)
                      encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch_dev(dev_set, bucket_id,mm*FLAGS.batch_size, fixed_set, weibo_set, qa_set)
                      _, eval_loss_per, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True, force_dec_input=False,beam_search = False)
                      _, eval_loss_per, _ = model.step(sess, encoder_inputs,decoder_inputs, target_weights, bucket_id, forward_only=False, force_dec_input=True,beam_search =    False)
                      eval_sum=0.
                      for ee in eval_loss_per:
                        eval_sum+=ee
                      #print (eval_loss_per)
                      eval_ppl+=eval_sum/len(eval_loss_per)
                      eval_num+=1
                      resp_tokens = model.remove_type(output_logits, model.buckets[bucket_id], type=1)
                      #prob = model.calprob(sess,_buckets, encoder_inputs, decoder_inputs, target_weights,batch_source_encoder, batch_source_decoder, bucket_id,rev_vocab=rev_vocab)
                      resp_c = model.ids2tokens(resp_tokens,rev_vocab)
                      resp_b = model.ids2tokens(batch_source_decoder,rev_vocab)
                      resp_a = model.ids2tokens(batch_source_encoder,rev_vocab)
                      for ii in range(len(resp_a)):
                        aa=""
                        for ww in resp_a[ii]:
                          aa=aa+" "+ww
                        bb=""
                        for ww in resp_b[ii]:
                          bb=bb+" "+ww
                        cc=""
                        pre=""
                        for ww in resp_c[ii]:
                          cc=cc+" "+ww
                          if ww not in dictt:
                            dictt[ww]=0
                          if pre+ww not in dictt_b:
                            dictt_b[pre+ww]=0
                          dictt[ww]+= 1
                          dictt_b[pre+ww]+=1
                          pre=ww
                        #print("Q:",aa)
                        #print("A1:",bb)
                        #print("A2:",cc)
                        #print("\n")
                        outputFile.write("%s\n%s\n%s \n\n"%(aa,bb,cc))
                        outputFile.flush()
                        BLEUscore =   nltk.translate.bleu_score.sentence_bleu([resp_c[ii]],resp_b[ii])
                        eval_loss += BLEUscore
                        mmm +=1
                      #dummy = model.caldummy(sess,_buckets, encoder_inputs, decoder_inputs,    target_weights,batch_source_encoder, batch_source_decoder, bucket_id,rev_vocab=rev_vocab)
                      #print(dummy)
                      #eval_loss +=dummy
                  eval_loss = eval_loss/mmm
                  eval_ppl=eval_ppl/eval_num
                  # Print statistics for the previous epoch.
                  perplexity = math.exp(loss) if loss < 300 else float('inf')
                  print ("global step %d learning rate %.4f step-time %.2f loss "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time,  loss))
                  # Decrease learning rate if no improvement was seen over last 3 times.
                  if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                  previous_losses.append(loss)
                  # Save checkpoint and zero timer and loss.
                  checkpoint_path = os.path.join(FLAGS.train_dir, "weibo.model")
                  model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                  summ=[dictt[w] for w in dictt]
                  summ=1.0*sum(summ)
                  eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                  print("  eval: bucket %d distinct-1 %.5f  distinct-2  %.5f bleu  %.5f   ppl  %.5f" % (bucket_id, len(dictt)/summ,len(dictt_b)/summ,eval_loss,eval_ppl))
                  outputFile.write("  eval: loss  %.5f distinct-1 %.5f  distinct-2  %.5f bleu  %.5f  ppl  %.5f\n" %(loss, len(dictt)/summ,len(dictt_b)/summ,eval_loss,eval_ppl))
                  outputFile.flush()
                  outputFile.close()
                  step_time, loss = 0.0, 0.0
                  # Run evals on development set and print their perplexity.
                  # for bucket_id in xrange(len(_buckets)):
                  #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  #       dev_set, bucket_id)
                  #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                  #                                target_weights, bucket_id, True)
                  #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                  #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                  sys.stdout.flush()

def self_test():
  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  with tf.Session(config = tfconfig) as sess:
    beam_size = FLAGS.beam_size
    beam_search = FLAGS.beam_search
    if beam_search:
      FLAGS.batch_size=1
    #FLAGS.batch_size=1
    print("Self-test for neural translation model.")
    train_path = os.path.join(FLAGS.data_dir, "weibo")
    fixed_path = os.path.join(FLAGS.data_dir, "fixed")
    weibo_path = os.path.join(FLAGS.data_dir, "wb")
    qa_path = os.path.join(FLAGS.data_dir, "qa")

    voc_file_path = [train_path+".answer", fixed_path+".answer", weibo_path+".answer", qa_path+".answer",
                     train_path+".query", fixed_path+".query", weibo_path+".query", qa_path+".query"]

    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.txt" % FLAGS.vocab_size)

    data_utils.create_vocabulary(vocab_path, voc_file_path, FLAGS.vocab_size)

    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    print(len(vocab))
    print("Preparing Chitchat data in %s" % FLAGS.data_dir)
    train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
      FLAGS.data_dir, vocab, FLAGS.vocab_size)

    print("Preparing Fixed data in %s" % FLAGS.fixed_set_path)
    fixed_path = os.path.join(FLAGS.fixed_set_path, "fixed")
    fixed_query , fixed_answer = data_utils.prepare_defined_data(fixed_path, vocab, FLAGS.vocab_size)

    print("Preparing Weibo data in %s" % FLAGS.weibo_set_path)
    weibo_path = os.path.join(FLAGS.weibo_set_path, "wb")
    weibo_query, weibo_answer = data_utils.prepare_defined_data(weibo_path, vocab, FLAGS.vocab_size)

    print("Preparing QA data in %s" % FLAGS.qa_set_path)
    qa_path = os.path.join(FLAGS.qa_set_path, "qa")
    qa_query, qa_answer = data_utils.prepare_defined_data(qa_path, vocab, FLAGS.vocab_size)

    dummy_path = os.path.join(FLAGS.data_dir, "dummy")
    dummy_set = data_utils.get_dummy_set(dummy_path, vocab, FLAGS.vocab_size)
    print("Get Dummy Set : ", dummy_set)

    model = create_model(sess,dummy_set, True, beam_search=beam_search, beam_size=beam_size)

    dev_set = read_data(dev_query, dev_answer,0,100000)
    train_set = read_data(train_query, train_answer, 0*100000,(0+1)*100000)
    fixed_set = read_data(fixed_query, fixed_answer, FLAGS.max_train_data_size)
    weibo_set = read_data(weibo_query, weibo_answer, FLAGS.max_train_data_size)
    qa_set = read_data(qa_query, qa_answer, FLAGS.max_train_data_size)
    outputFile = open("OpenSubData/RL_test.txt","w")
    for idd in range(2):
      bucket_id=idd+2
      batch_num = 1+int(len(dev_set[bucket_id])/FLAGS.batch_size)
      for mm in range(batch_num):
        encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch_dev(dev_set, bucket_id,mm*FLAGS.batch_size, fixed_set, weibo_set, qa_set)
        if beam_search:
          path, symbol ,embedding , output_logits= model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, forward_only=True,beam_search= beam_search)
          k = output_logits[0]
          paths = []
          for kk in range(beam_size):
            paths.append([])
          curr = range(beam_size)
          num_steps = len(path)
          for i in range(num_steps-1, -1, -1):
            for kk in range(beam_size):
              paths[kk].append(symbol[i][curr[kk]])
              curr[kk] = path[i][curr[kk]]
          recos = set()
          resp_b = model.ids2tokens(batch_source_decoder,rev_vocab)
          resp_a = model.ids2tokens(batch_source_encoder,rev_vocab)
          aa=" ".join(ww for ww in resp_a[0])
          bb=" ".join(ww for ww in resp_b[0])
          outputFile.write("%s\n%s\n"%(aa,bb))

          for kk in range(beam_size):
            foutputs = [int(logit)  for logit in paths[kk][::-1]]
          # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in foutputs:
              foutputs = foutputs[:foutputs.index(data_utils.EOS_ID)]
            rec = " ".join([tf.compat.as_str(rev_vocab[output]) for output in foutputs])
            if rec not in recos:
              recos.add(rec)
              outputFile.write("%s\n"%(rec))
          outputFile.write("\n")
          outputFile.flush()
          sys.stdout.flush()
        else:
              _, eval_loss_per, output_logits = model.step(sess, encoder_inputs,
                      decoder_inputs,target_weights, bucket_id, forward_only=True, force_dec_input=False,beam_search = False)

              resp_tokens = model.remove_type(output_logits, model.buckets[bucket_id], type=1)
              resp_c = model.ids2tokens(resp_tokens,rev_vocab)
              resp_b = model.ids2tokens(batch_source_decoder,rev_vocab)
              resp_a = model.ids2tokens(batch_source_encoder,rev_vocab)
              for ii in range(len(resp_a)):
                aa=""
                for ww in resp_a[ii]:
                  aa=aa+" "+ww
                bb=""
                for ww in resp_b[ii]:
                  bb=bb+" "+ww
                cc=""
                for ww in resp_c[ii]:
                  cc=cc+" "+ww
                outputFile.write("%s\n%s\n%s \n\n"%(aa,bb,cc))
                outputFile.flush()
              sys.stdout.flush()

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    pass
    #decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
