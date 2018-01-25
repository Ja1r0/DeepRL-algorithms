import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
import random

def huber_loss():
    pass
def collect_params():
    pass
def build_cnn(input,act_num,scope,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        out=input
        with tf.variable_scope('convnet'):
            out=layers.convolution2d(out,num_outputs=32,kernel_size=8,stride=4,activation_fn=tf.nn.relu)
            out=layers.convolution2d(out,num_outputs=64,kernel_size=4,stride=2,activation_fn=tf.nn.relu)
            out=layers.convolution2d(out,num_outputs=64,kernel_size=3,stride=1,activation_fn=tf.nn.relu)
        out=layers.flatten(out)
        with tf.variable_scope('action_value'):
            out=layers.fully_connected(out,num_outputs=512,activation_fn=tf.nn.relu)
            out=layers.fully_connected(out,num_outputs=act_num,activation_fn=None)
        return out
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos=device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type=='GPU']
def get_session():
    tf.reset_default_graph()
    tf_config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    session=tf.Session(config=tf_config)
    print('AVAILABLE GPUS:',get_available_gpus())
    return session

class Replay_buffer:
    def __init__(self,size,history_frames):
        '''
        :param size: the capacity of replay buffer
        :param history_frames: the number of recent frames to stack
        '''
        self.size=size
        self.history_frames=history_frames
        self.next_idx=0
        self.num_in_buffer=0
        self.obs=None
        self.act=None
        self.rew=None
        self.end=None

    def can_sample(self,batch_size):
        '''
        :param batch_size: the number of samples per batch
        :return: True if num_in_buffer > batch_size
        '''
        return batch_size+1<=self.num_in_buffer

    def sample(self,batch_size):
        '''
        :param batch_size: the number of transitions to sample
        :return:
        obs_batch
        =========
            {ndarray}
            (batch_size , img_h , img_w , img_c*history_frames)
            np.unint8
        act_batch
        =========
            {ndarray}
            (batch_size,)
            np.int32
        rew_batch
        =========
            {ndarray}
            (batch_size,)
            np.float32
        next_obs_batch
        ==============
            {ndarray}
            (batch_size , img_h , img_w , img_c*history_frames)
            np.uint8
        done_batch
        ==========
            {ndarray}
            (batch_size,)
            np.float32
        '''
        assert self.can_sample(batch_size)
        idxes=sample_n_unique(lambda random.randint(0,self.num_in_buffer-2),batch_size)
        return self._stack_sample(idxes)

    def store_frame(self,frame):
        '''
        push the current frame into the buffer in the next available index,
        delete the oldest frame if the buffer is filled.
        :param
        frame
        =====
            {ndarray}
            (img_h,img_w,img_c)
            np.uint8
        :return:
        idx
        ===
            the index at where the frame is stored.
            {int}
        '''
        if self.obs is None:
            self.obs=np.empty([self.size]+list(frame.shape),dtype=np.uint8)
            self.act=np.empty([self.size],dtype=np.int32)
            self.rew=np.empty([self.size],dtype=np.float32)
            self.end=np.empty([self.size],dtype=np.bool)
        self.obs[self.next_idx]=frame
        idx_of_frame=self.next_idx
        self.next_idx=(self.next_idx+1)%self.size
        self.num_in_buffer=min(self.size,self.num_in_buffer+1)
        return idx_of_frame

    def store_transition(self,idx,action,reward,done):
        '''
        :param idx: {int}
        :param action: {int}
        :param reward: {float}
        :param done: {bool}
        :return: store the transition sample (a,r,done) to buffer
        '''
        self.act[idx]=action
        self.rew[idx]=reward
        self.end[idx]=done

    def stack_recent_obs(self):
        '''
        :return: stack recent observations of length history_frames
        observations
        ============
            {ndarray}
            (img_h,img_w,img_c*history_frames)
            np.uint8
        '''
        assert self.num_in_buffer>0
        return self._stack_obs((self.next_idx-1)%self.size)

    def _stack_sample(self,idxes):
        obs_batch=np.concatenate([self._stack_obs(idx)[None] for idx in indes],0)
        act_batch=self.act[idxes]
        rew_batch=self.rew[idxes]
        next_obs_batch=np.concatenate([self._stack_obs(idx+1)[None] for idx in idxes],0)
        done_batch=np.array([1.0 if self.end[idx] else 0.0 for idx in idxes],dtype=np.float32)
        return obs_batch,act_batch,rew_batch,next_obs_batch,done_batch

    def _stack_obs(self,idx):
        end_idx=idx+1
        start_idx=end_idx-self.history_frames
        if start_idx<0 and self.num_in_buffer!=self.size:
            start_idx=0
        for idx in range(start_idx,end_idx-1):
            if self.end[idx%self.size]:
                start_idx=idx+1
        missing_context=self.history_frames-(end_idx-start_idx)
        if start_idx<0 or missing_context>0:
            frames=[np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx,end_idx):
                frames.append(self.obs[idx%self.size])
            return np.concatenate(frames,2)
        else:
            img_h,img_w=self.obs.shape[1],self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1,2,0,3).reshape(img_h,img_w,-1)

def initialize_interdependent_variables(session,vars_list,feed_dict):
    '''
    initialize a list of variables,when the initialization of these variables
    depends on the initialization of the other variables.
    '''
    vars_left=vars_list
    while len(vars_left)>0:
        new_vars_left=[]
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]),feed_dict)
            # raised when running an operation that reads a tf.Variable before it has been initialized
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left)>=len(vars_left):
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left=new_vars_left
def minimize_and_clip(optimizer,objective,var_list,clip_val=10):
    '''
    Minimize 'objective' using 'optimizer' w.r.t. variables in
    'var_list' while ensure the norm of the gradients for each
    variables is clipped to 'clip_val'.
    '''
    gradients=optimizer.compute_gradients(objective,var_list=var_list)
    for i,(grad,var) in enumerate(gradients):
        if grad is not None:
            gradients[i]=(tf.clip_by_norm(grad,clip_val),var)
    return optimizer.apply_gradients(gradients)