import tensorflow as tf
import itertools
import utils
import random
import sys
import numpy as np

def learn_by_dqn(env,
              q_net,
              optimizer_spec,
              session,
              exploration,
              replay_buffer_size,
              batch_size,
              gamma,
              learn_start,
              learn_freq,
              history_frames_num,
              target_update_freq,
              grad_norm_clipping,
              stop_criterion=None):
    ###################
    # build functions #
    ###################
    img_h,img_w,img_c=env.observation_space.shape
    input_shape=(img_h,img_w,history_frames_num*img_c)
    act_num=env.action_space.n
    # c : current
    # n : next
    ### set up placeholders ###
    obs_c_ph=tf.placeholder(tf.uint8,[None]+list(input_shape))
    act_c_ph=tf.placeholder(tf.int32,[None])
    rew_c_ph=tf.placeholder(tf.float32,[None])
    obs_n_ph=tf.placeholder(tf.uint8,[None]+list(input_shape))
    done_ph=tf.placeholder(tf.float32,[None]) # if next state is the end, 0. or 1.
    ### transform the observation pixels value to float between 0.~1. ###
    obs_c_float=tf.cast(obs_c_ph,tf.float32)/255.0
    obs_n_float=tf.cast(obs_n_ph,tf.float32)/255.0
    ### compute the TD error ###
    q_c_values=q_net(obs_c_float,act_num,scope='q_net',reuse=False)
    q_c_selected=tf.reduce_sum(q_c_values*tf.one_hot(act_c_ph,act_num),1)
    q_n_values=q_net(obs_n_float,act_num,scope='target_q_net')
    q_n_max=tf.reduce_max(q_n_values,1)
    q_c_selected_target=rew_c_ph+gamma*(1.0-done_ph)*q_n_max
    td_error=q_c_selected-tf.stop_gradient(q_c_selected_target)
    errors=utils.huber_loss(td_error)
    mean_error=tf.reduce_mean(errors)
    ### collection parameters of q_net and target_q_net ###
    q_net_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_net')
    target_q_net_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_q_net')
    ### optimization function ###
    learn_rate=tf.placeholder(tf.float32,(),name='learn_rate')
    optimizer=optimizer_spec.constructor(learning_rate=learn_rate,**optimizer_spec.kwargs)
    train_fn=utils.minimize_and_clip(optimizer,mean_error,var_list=q_net_params,clip_val=grad_norm_clipping)
    ### update target q network function ###
    update_target_fn=[]
    for param,target_param in zip(sorted(q_net_params,key=lambda p:p.name),
                            sorted(target_q_net_params,key=lambda p:p.name)):
        update_target_fn.append(target_param.assign(param))
    ### set up replay buffer ###
    replay_buffer=utils.Replay_buffer(replay_buffer_size,history_frames_num)
    #######################
    # interation with env #
    #######################
    model_initialized=False
    train_num=-1
    mean_episode_reward=-float('nan')
    best_mean_episode_reward=-float('inf')
    last_obs=env.reset()
    #env.render()
    LOG_EVERY_N_STEPS=10000
    init_op=tf.global_variables_initializer()
    session.run(init_op)
    ### the counter of time steps ###
    for t in itertools.count():
        if stop_criterion is not None and stop_criterion(env,t):
            break
        idx=replay_buffer.store_frame(last_obs)
        obs_c=replay_buffer.stack_recent_obs()
        explore_prob=random.random()
        ### epsilon greedy exploration policy ###
        if explore_prob<exploration.value(t):
            action=env.action_space.sample() # ??? #
        else:
            action_values=session.run(q_c_values,feed_dict={obs_c_ph:obs_c[None]})
            action=np.argmax(action_values)
        ### step to next state ###
        obs,reward,done,info=env.step(action)
        #env.render()
        replay_buffer.store_transition(idx,action,reward,done)
        if not done:
            last_obs=obs
        else:
            last_obs=env.reset()
        ### train the networks ###
        if (t>learn_start and
            t % learn_freq==0 and
            replay_buffer.can_sample(batch_size)):
            # 1.sample transitions #
            obs_batch,act_batch,rew_batch,next_obs_batch,done_batch\
                =replay_buffer.sample(batch_size)
            # 2.initialize the model #
            if not model_initialized:
                utils.initialize_interdependent_variables(session,tf.global_variables(),{
                    obs_c_ph:obs_batch,
                    obs_n_ph:next_obs_batch,
                })
                model_initialized=True
            # 3.train the model #
            session.run(train_fn,feed_dict={
                obs_c_ph:obs_batch,
                act_c_ph:act_batch,
                rew_c_ph:rew_batch,
                obs_n_ph:next_obs_batch,
                done_ph:done_batch,
                learn_rate:optimizer_spec.lr_schedule.value(t)
            })
            train_num+=1
            # 4.update target network #
            if train_num % target_update_freq==0:
                session.run(update_target_fn)
        #######
        # log #
        #######
        episode_rewards=utils.get_wrapper_by_name(env, "Monitor").get_episode_rewards() # ??? #
        if len(episode_rewards)>0:
            mean_episode_reward=np.mean(episode_rewards[-100:])
        if len(episode_rewards)>100:
            best_mean_episode_reward=max(best_mean_episode_reward,mean_episode_reward)
        if t % LOG_EVERY_N_STEPS==0 and model_initialized:
            print('##########log##########')
            print('Timestep %d'%(t,))
            print('mean reward (100 episodes) %f' % mean_episode_reward)
            print('best mean reward %f' % best_mean_episode_reward)
            print('episodes %d' % len(episode_rewards))
            print('exploration %f' % exploration.value(t))
            print('learning_rate %f' % optimizer_spec.lr_schedule.value(t))
            print('#######################')
            sys.stdout.flush() # show all information on terminal before next interation begin














