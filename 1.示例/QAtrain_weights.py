

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

模型正使用到的参数:
- init_scale - 权重的初始化比例
- learning_rate - 初始学习率的值
- max_grad_norm - 梯度的最大允许范数
- num_layers - LSTM的层数
- num_steps - LSTM 的展开步数
- hidden_size - LSTM的单元数量
- max_epoch - 使用初始学习率训练的epoch数
- max_max_epoch - 训练的总epoch数
- keep_prob - 在 dropout 层中保持权重的概率
- lr_decay - "max_epoch" 之后每个 epoch 的学习率衰减
- batch_size - 批数据的大小

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import rnn, rnn_cell
import time
import tensorflow as tf
import QAreader
import warnings
import os

# 消除警告
warnings.filterwarnings(action='ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

# -----------------------定义参数---------------------------
flags.DEFINE_string(
    "model", "small",
    "模型的类型,可以选择: small, medium, large.")

flags.DEFINE_string("data_path", None,
                    "train/test数据存储的路径")
flags.DEFINE_string("save_path", "./model_weight/model_weights",
                    "模型保存的位置")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool(
    "ifcontinue",True,
    "是开始训练还是继续训练")
flags.DEFINE_string(
    "data_type","1",
    "选择训练的数据类型")
FLAGS = flags.FLAGS

# 数据类型是float16还是float32
def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


# 输入数据处理
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data,vocab=1000, name=None,answering=False):
    self.batch_size = batch_size = config.batch_size
    if (answering):
        self.epoch_size=1
    else:
        self.epoch_size = (len(data[0]) // batch_size)
    self.documents, self.questions,self.vanswers,self.answers,_= QAreader.ptb_producer(data[0],data[1],data[2],batch_size,vocab,name=name,config=config)

# 定义RNN模型
class PTBModel(object):
  """The PTB model."""
  def __init__(self, is_training, config, input_,answering=False):

    '''
    def BiRNN(x):

      # Prepare data shape to match `bidirectional_rnn` function requirements
      # Current data input shape: (batch_size, n_steps, n_input)
      # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

      # Permuting batch_size and n_steps
      x = tf.transpose(x, [1, 0, 2])
      # Reshape to (n_steps*batch_size, n_input)
      x = tf.reshape(x, [-1, size])
      # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
      x = tf.split(x, num_steps, 0)

      # Define lstm cells with tensorflow
      # Forward direction cell
      lstm_fw_cell = rnn.BasicLSTMCell(size, forget_bias=1.0)
      # Backward direction cell
      lstm_bw_cell = rnn.BasicLSTMCell(size, forget_bias=1.0)

      # Get lstm cell output
      try:
          outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
      except Exception: # Old TensorFlow version only returns outputs not states
          outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)

      # Linear activation, using rnn inner loop last output
      return  outputs,output_state_fw,output_state_bw #tf.matmul(outputs[-1], weights['out']) + biases['out']
    '''
    self._input = input_
    documents_input =  input_.documents      # document和question的输入
    questions_input = input_.questions
    self.answers_input = input_.answers      # answer的输入
    batch_size = input_.batch_size  # 批处理数据的大小
    document_steps = documents_input.get_shape().as_list()[1]   # docement的步长
    question_steps = questions_input.get_shape().as_list()[1]   # question的步长
    self.document_steps=document_steps
    self.question_steps=question_steps
    # 隐藏层大小
    size = config.hidden_size
    # 词的大小
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.

    # 定义一个基本的 LSTM 循环网络单元
    def lstm_cell():
      return rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
    # RNN单元
    cell = rnn_cell.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    # 初始化各种状态
    self._initial_state = cell.zero_state(batch_size, data_type())
    # 嵌入层
    embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
    self.embedding =embedding.name
    print(embedding.name)
    # document和question嵌入
    documents = tf.nn.embedding_lookup(embedding, input_.documents)
    questions = tf.nn.embedding_lookup(embedding, input_.questions)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    print("1")
    doc_outputs=[]
    cell_output_fws=[]
    cell_output_bws=[]
    documents_reverse= tf.reverse(documents,[1])
    questions_reverse= tf.reverse(questions,[1])
    doc_weights=[]
    print(document_steps)

    # 对documents构造前向和后向传播
    with tf.variable_scope("documents"):
        state_fw = self._initial_state
        state_bw = self._initial_state
        for time_step in range(document_steps):
            #print(time_step)
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output_fw, state_fw)=cell(documents[:, time_step, :], state_fw,scope="doc_fw")
            (cell_output_bw, state_bw) = cell(documents_reverse[:, time_step, :],state_bw,scope="doc_bw")
            cell_output_fws.append(cell_output_fw)
            cell_output_bws.append(cell_output_bw)
        for time_step in range(document_steps):   
            doc_outputs.append(tf.concat([cell_output_fws[time_step],tf.reverse(cell_output_bws,[0])[time_step]],1))
        doc_outputs=tf.convert_to_tensor(doc_outputs)

    # print("2")
    # 对questions构造前向和后向传播
    with tf.variable_scope("questions"):  
        #compute question output
        state_fw = self._initial_state
        state_bw = self._initial_state
        for time_step in range(question_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output_fw, state_fw)= cell(questions[:, time_step, :], state_fw,scope="que_fw")
            (cell_output_bw, state_bw) = cell(questions_reverse[:, time_step, :],state_bw,scope="que_bw")
        que_output=tf.concat([cell_output_fw,cell_output_bw],1)
    matrix_w=tf.get_variable("W", [2*size, 2*size], dtype=data_type())
    
    for batch in range(batch_size):
        temp_vector = tf.matmul(tf.matmul(doc_outputs[:, batch, :], matrix_w),
                                tf.reshape(que_output[batch, :], [2 * size, 1]))
        doc_weights.append(tf.nn.softmax(temp_vector, 0))
    doc_weights=tf.convert_to_tensor(doc_weights)
    doc_weights= tf.transpose(tf.reshape( doc_weights,[batch_size,document_steps]))
    self.doc_weights=doc_weights  # document权重

    logits=[]
    for batch in range(batch_size): 
        tmp=tf.one_hot(input_.documents[batch], config.vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=data_type())
        tmp=tf.transpose(tmp)
        #tmp:vocab_size,time_step
        tmp2=tf.matmul(tmp,tf.reshape(doc_weights[:,batch],[document_steps,1]))
        #tmp2:vocab_size,1
        tmp2=tf.reshape(tmp2,[vocab_size])
        logits.append(tmp2)
    logits=tf.convert_to_tensor(logits)

    self.word_index=tf.argmax( tf.nn.softmax(logits)[0],0)
    self.logits_origin=logits[0]
    self.logits = tf.nn.softmax(logits[0])#tf.divide(tf.exp(logits[0]), tf.reduce_sum(tf.exp(logits[0])))
    self.vanswer=input_.vanswers[0]
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_.vanswers, logits=logits))
    self._cost=cost=loss
    self.first_loss=tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(input_.vanswers,data_type()), logits=logits)[0]
    self.actual=input_.answers[0]
    self.correct_prediction = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(logits,1),tf.cast(input_.answers,tf.int64)), data_type()) )

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)   # 梯度下降的优化器
    
    self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")  # 新学习率
    self._lr_update = tf.assign(self._lr, self._new_lr)  #  更新学习率

  #   学习率
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

# 训练small模型的参数
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size =500
  max_epoch = 4
  max_max_epoch = 15
  keep_prob = 1
  lr_decay = 0.6
  batch_size = 20
  vocab_size = 100

# 训练medium模型的参数
class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000

# 训练large模型的参数
class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

# 测试模型的参数
class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

# 运行每个epoch，并计算准确率，预测答案的时候也将会运行该函数
def run_epoch(session, model, eval_op=None, verbose=False,training=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  accuracy = 0

  # 参数
  fetches = {
      "cost": model.cost,
      "logits":model.logits,
     # "final_state": model.final_state,
      "index":model.word_index,
      "actual":model.actual,
      "vanswer":model.vanswer,
      "first_loss":model.first_loss,
      #"doc_avg":model.doc_avg,
      
      "logits_origin":model.logits_origin,
      "correct_prediction":model.correct_prediction,
      "embedding": model.embedding,
      
      "initial_state":model._initial_state,
      "doc_weights":model.doc_weights
  }
 
  if training:
    #fetches["train"]=model.train_step
    fetches["train"]=model._train_op
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    #print("inside")
    state = session.run(model.initial_state)
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    index = vals["index"]

    # 统计准确的数量
    accuracy=accuracy+vals["correct_prediction"]

    if(verbose):
        # print("Accuracy in this round: %s"  % vals["correct_prediction"])
        answer_word=QAreader._word_id_to_word(index)
        print("The Question answer is: ",answer_word)
        # print("actual word: %s" % QAreader._word_id_to_word (vals["actual"]))
    costs += cost
    iters += model.document_steps+model.question_steps
    if verbose and step % 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,cost, #cost,#np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
  return costs/ model.input.epoch_size, accuracy/model.input.epoch_size  #np.exp(costs / iters)


# 训练small、mediu或test模型，还是测试模型
def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

# 训练的主函数
def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  # 读取train_data,test_data数据集并生成词典
  train_data, test_data,vocab = QAreader.prepare_data(FLAGS.data_path,FLAGS.data_type)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
      # 使用CPU进行训练
      with tf.device('/cpu:0'):
        if FLAGS.ifcontinue==False:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        else:
            initializer = None
        # 训练
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config,vocab=vocab, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)
        # 测试
        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=test_data,vocab=vocab, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)
        print("model built")
        #saver = tf.train.Saver()
        sv = tf.train.Supervisor()#(logdir=FLAGS.save_path)
        # 重新加载模型，继续训练
        with sv.managed_session() as session:
          if FLAGS.ifcontinue:
             sv.saver.restore(session,  "./model_weight/model_weights_"+FLAGS.data_type)
             print("model restored!")
          for mm in tf.global_variables():
             print (mm.name)

          for i in range(config.max_max_epoch):
            if FLAGS.ifcontinue==False:
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
            else:
                m.assign_lr(session, 0.0005)
            # 输出每个epoch的学习率
            print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(m.lr)))
            train_perplexity,accuracy = run_epoch(session, m , verbose=True,training=True)   #, eval_op=m.train_op,
            # 输出每个train数据epoch的Loss和accuracy
            print("Epoch: %d Train Loss: %.5f" % (i + 1, train_perplexity))
            print("Epoch: %d Train Accuracy: %.5f " % (i + 1, accuracy))
            # 输出每个Valid数据epoch的Loss和accuracy
            valid_perplexity,accuracy = run_epoch(session, mvalid)
            print("Epoch: %d Valid Loss: %.5f" % (i + 1, valid_perplexity))
            print("Epoch: %d Valid Accuracy: %.5f " % (i + 1, accuracy))
          # 保存模型
          if FLAGS.save_path:
            print("Saving model to %s." %  "./model_weight/model_weights_"+FLAGS.data_type)
            print(sv.saver.save(session,  "./model_weight/model_weights_"+FLAGS.data_type))
            #saver.save(session, "./model.ckpt")

if __name__ == "__main__":
  tf.compat.v1.app.run()
