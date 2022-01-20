from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import re
import sys
import QAtrain_weights
import QAreader
import json
import warnings
import os

warnings.filterwarnings(action='ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
FLAGS = QAtrain_weights.flags.FLAGS

# 数据类型是float16还是float32
def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

session=None

# 预测的主函数
def main(_):
    
  global session
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  config = QAtrain_weights.get_config()
  eval_config = QAtrain_weights.get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  # 输入document和question,预测答案
  while(True):
      answering_data = []
      terminate = False
      answering_documents = []
      answering_answers = []
      answering_questions = []
      if (terminate): break
      print("输入文档和问题(输入问题后空行回车):")
      currentDocument = []
      newarray = []
      pline = ""
      for newline in iter(sys.stdin.readline, ''):
          # print(newline)
          newline = re.sub("\?|\.|,|\n", "", newline.lower())
          newline = re.sub("[0-9]|[0-9][0-9]", " ", newline)
          if (newline):
              currentDocument += pline.split()
              pline = newline
          else:
              break
      if (len(currentDocument) == 0): continue
      # 空行的上一句是question,即pline保存的是question
      newarray = pline.split()
      answering_documents.append(currentDocument + ["PAD", "PAD", "PAD"])
      answering_answers.append("PAD")
      answering_questions.append(newarray[0:len(newarray)] + ["PAD"])
      all_words = []
      # 将answering_documents、answering_questions、answering_answers的词都加入到all_word中
      for document in answering_documents:
          all_words += document
      for question in answering_questions:
          all_words += question
      for answer in answering_answers:
          all_words.append(answer)
      # 加载词映射
      with open('./word_to_id.json', 'r') as f:
          try:
              QAreader.word_to_id = json.load(f)
          # if the file is empty the ValueError will be thrown
          except ValueError:
              QAreader.word_to_id = {}
      for i in range(len(answering_documents)):
          answering_documents[i] = QAreader._words_to_word_ids(answering_documents[i], QAreader.word_to_id)
      for i in range(len(answering_questions)):
          answering_questions[i] = QAreader._words_to_word_ids(answering_questions[i], QAreader.word_to_id)
      # 将answering_answers映射成id
      answering_answers = QAreader._words_to_word_ids(answering_answers, QAreader.word_to_id)
      answering_data.append(answering_documents)
      answering_data.append(answering_questions)
      answering_data.append(answering_answers)

      with tf.Graph().as_default():
          initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
          with tf.name_scope("Train"):
              answering_input = QAtrain_weights.PTBInput(config=eval_config, data=answering_data, name="AnsweringInput",answering=True)
              with tf.variable_scope("Model", reuse=False, initializer=None):
                  manswering = QAtrain_weights.PTBModel(is_training=False, config=eval_config, input_=answering_input,answering=True)

          sv = tf.train.Supervisor()
          # 加载保存的模型进行预测
          with sv.managed_session() as session:
              sv.saver.restore(session, './model_weight/model_weights_' + FLAGS.data_type)
              print("try to answer")
              # 获取question的answer
              answering_perplexity = QAtrain_weights.run_epoch(session, manswering, verbose=True)
              # print("answer:",answering_perplexity)

if __name__ == '__main__':
    tf.compat.v1.app.run()
