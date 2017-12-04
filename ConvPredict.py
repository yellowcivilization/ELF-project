from tensorflow import *
import numpy as np
from ConvNeuroNet.Preprocess import Preprocess
from ConvNeuroNet.Config import config
from frequent_patten.visualize import plothist
import matplotlib.pyplot as plt
import random


class SellPredict:
    def __init__(self):
        self.sess=Session()
        self.keep_prob=placeholder(dtype=float32)
    '''
    这个类用于做销量预测
    '''
    class Item:
        '''
        这个类用于存储数据，一个Item代表一个产品
        '''
        def __init__(self,id,image):
            '''
            id表示产品编号
            image是产品图片的三维np.array
            :param id:
            :param image:
            '''
            self.id=id
            self.image=image

    def get_FC(self,input_tensor,input_shape,num_unit,name):
        '''
        获得一个全连接层
        :return:
        '''
        # Variable_tmp=Variable(random_uniform((input_shape[1],num_unit),
        #                                      minval=-sqrt(6/(input_shape[1]+num_unit)),
        #                                      maxval=sqrt(6/(input_shape[1]+num_unit))),
        #                       dtype=float32)
        Variable_tmp=Variable(np.random.randn(input_shape[1],num_unit)*np.sqrt(2/input_shape[1]),dtype=float32)
        bias_tmp=Variable(zeros(num_unit),dtype=float32)
        features=matmul(input_tensor,Variable_tmp)+bias_tmp
        active=nn.relu(features)
        summary.histogram(name,active)
        return active

    def init_forward(self):
        '''
        定义初始前向过程
        :return:
        '''
        #输入的张良形状为[None,config.height,config.width,3]
        with name_scope('input'):
            self.input_op=placeholder(shape=[None,config.height,config.width,3],dtype=float32)
        #归一化
        formate=self.input_op/255-0.5
        summary.histogram('raw_input',formate)
        #定义第一层卷积用于读取颜色信息,不需要进行pooling,卷积结束后的张量形状为[None,config.height,config.width,config.color_units]
        #初始化卷积核权重和偏置
        # color_kernel=Variable(random_uniform(shape=[1,1,3,config.color_units],
        #                                      minval=-sqrt(6/(3+config.color_units)),
        #                                      maxval=sqrt(6/(3+config.color_units))),
        #                       dtype=float32)
        with name_scope('conv_net'):
            def leek_relu(x):
                return maximum(x*(-0.1),x)
            # color_kernel=Variable(random_normal(shape=[1,1,3,config.color_units],)/np.sqrt(3/2)
            #                       )
            # add_to_collection(GraphKeys.WEIGHTS, color_kernel)
            # color_bias=Variable(zeros(config.color_units)*0.1,dtype=float32)
            # color_conv=nn.relu(nn.conv2d(formate,color_kernel,[1,1,1,1],padding='SAME')+color_bias,name='color_conv')
            # summary.image('color_conv_activation_map',color_conv)
            #定义卷积层，需要进行pooling，各层参数见Execl
            #conv_1
            # conv_kernel_1=Variable(random_uniform(shape=[3,3,config.color_units,config.conv_units*2],
            #                                       minval=-sqrt(6/(config.color_units+config.conv_units*2)),
            #                                       maxval=sqrt(6/(config.color_units+config.conv_units*2))),
            #                        dtype=float32)
            conv_kernel_1=Variable(np.random.randn(3,3,3,3)*np.sqrt(2/(3*3*3)),dtype=float32)
            summary.histogram('conv_kernel',conv_kernel_1)
            add_to_collection(GraphKeys.WEIGHTS, conv_kernel_1)
            conv_bias_1=Variable(zeros(3),dtype=float32)
            conv_1=nn.relu(nn.conv2d(formate,conv_kernel_1,[1,1,1,1],padding='SAME')+conv_bias_1)
            summary.histogram('conv_1',conv_1)
            pool_1=nn.max_pool(conv_1,[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_1')
            #conv_2
            # conv_kernel_2=Variable(random_uniform(shape=[3,3,config.conv_units*2,config.conv_units*4],
            #                                       minval=-sqrt(6/(config.conv_units*2+config.conv_units*4)),
            #                                       maxval=sqrt(6/(config.conv_units*2+config.conv_units*4))),
            #                        dtype=float32)
            conv_kernel_2 = Variable(random_normal(shape=[3,3,3,config.conv_units*4])*np.sqrt(2/(3*3*3)))
            add_to_collection(GraphKeys.WEIGHTS, conv_kernel_2)
            conv_bias_2 = Variable(zeros(config.conv_units*4), dtype=float32)
            conv_2=nn.relu(nn.conv2d(pool_1,conv_kernel_2,[1,1,1,1],padding='SAME')+conv_bias_2)
            summary.histogram('conv_2',conv_2)
            pool_2=nn.max_pool(conv_2,[1,3,3,1],[1,3,3,1],'SAME')
            #conv_3
            # conv_kernel_3=Variable(random_uniform(shape=[3,3,config.conv_units*4,config.conv_units*8],
            #                                       minval=-sqrt(6 / (config.conv_units * 4 + config.conv_units * 8)),
            #                                       maxval=sqrt(6 / (config.conv_units * 4 + config.conv_units * 8))),
            #                        dtype=float32)
            conv_kernel_3 = Variable(random_normal(shape=[3,3,config.conv_units*4,config.conv_units*8])*np.sqrt(2/(3*3*config.conv_units*4)))
            add_to_collection(GraphKeys.WEIGHTS, conv_kernel_3)
            conv_bias_3 = Variable(zeros(config.conv_units*8), dtype=float32)
            conv_3 = nn.relu(nn.conv2d(pool_2,conv_kernel_3,[1,1,1,1],padding='SAME')+conv_bias_3)
            summary.histogram('conv_3',conv_3)
            pool_3=nn.max_pool(conv_3,[1,3,3,1],[1,3,3,1],'SAME')
            conv_kernel_4=Variable(random_normal(shape=[3,3,config.conv_units*8,config.conv_units*16])*np.sqrt(2/(3*3*config.conv_units*8)))
            add_to_collection(GraphKeys.WEIGHTS, conv_kernel_4)
            conv_bias_4 = Variable(zeros(config.conv_units * 16), dtype=float32)
            conv_4=nn.conv2d(pool_3,conv_kernel_4,[1,1,1,1],padding='SAME')+conv_bias_4
            conv_4=nn.relu(conv_4)
            summary.histogram('conv_4',conv_4)
            pool_4=nn.max_pool(conv_4,[1,3,3,1],[1,3,3,1],'SAME')

        #对卷积后的特征进行reshape，使其成为一个形为[batch,num_feature]的矩阵
        with name_scope('Full_Connect_network'):
            conv_result=reshape(pool_4,[config.batch_size,-1])
            FC_1=self.get_FC(conv_result,[config.batch_size,9216],config.FC_units,'FC_1')
            FC_1=nn.dropout(FC_1,keep_prob=self.keep_prob)
            FC_2=self.get_FC(FC_1,[config.batch_size,config.FC_units],config.FC_units,'FC_2')
            FC_2=nn.dropout(FC_2,keep_prob=self.keep_prob)
            FC_3 = self.get_FC(FC_2, [config.batch_size, config.FC_units], config.FC_units,'FC_3')
            summary.histogram('FC_3',FC_3)


        #进行输出
        with name_scope('output'):
            output_weight=Variable(random_normal([config.FC_units,1])*sqrt(2/config.FC_units))
            output_bias=Variable(0.1,dtype=float32)
            output=matmul(FC_3,output_weight)+output_bias
        init=global_variables_initializer()
        self.sess.run(init)
        self.pre_op=reshape(output,[-1])
        summary.histogram('finaly_output',self.pre_op)

        #设置检查点
        self.check_point=self.pre_op

    def get_all_data(self):
        '''
        调用预处理类获得全部数据
        :return:
        '''
        tmp_preprocess=Preprocess()
        tmp_all_data=tmp_preprocess.get_all_data()
        random.shuffle(tmp_all_data)
        length=len(tmp_all_data)
        self.train_data = tmp_all_data[:int(length*0.7)]
        self.test_data = tmp_all_data[int(length*0.7):]

    def get_batch(self,is_train=True):
        '''
        调用get_batch()之前需要先调用get_all_data()
        :return:
        '''
        if is_train:
            sample=random.sample(self.train_data,config.batch_size)
            X=list()
            Y=list()
            for i in sample:
                X.append(i.image)
                Y.append(i.sales)
            return np.stack(X),np.array(Y)
        else:
            sample = random.sample(self.test_data , config.batch_size)
            X = list()
            Y = list()
            for i in sample:
                X.append(i.image)
                Y.append(i.sales)
            return np.stack(X), np.array(Y)


    def train(self,my_times):
        '''
        训练过程
        :return:
        '''
        with name_scope('train'):
            target=placeholder(dtype=float32,shape=[config.batch_size])
            cost=square(target-self.pre_op)
            regularizer = contrib.layers.l2_regularizer(scale=5.0 / 2800)
            reg_term=contrib.layers.apply_regularization(regularizer)
            self.loss=reduce_mean(cost)+reg_term
            summary.scalar('reg_term',reg_term)
            summary.scalar('cost', self.loss)
            merged=summary.merge_all()
            optimizer=train.AdamOptimizer(0.0001)
            optimizer.compute_gradients()
            train_step=optimizer.minimize(self.loss)
            init=global_variables_initializer()
            self.sess.run(init)
            writer = summary.FileWriter(r'D:\njust\HWM\tensorboard_visualize\train')
            writer.add_graph(self.sess.graph)
            test_writer = summary.FileWriter(r'D:\njust\HWM\tensorboard_visualize\test')
            test_writer.add_graph(self.sess.graph)
        k=0
        while k<my_times:
            X,Y=self.get_batch()
            _,summary_merged,check_point,cost=self.sess.run((train_step, merged, self.check_point,self.loss), feed_dict={self.input_op:X, target:Y ,self.keep_prob:0.5})
            writer.add_summary(summary_merged,k)
            if k%20==0:
                X,Y=self.get_batch(is_train=False)
                test_merged,test_show = self.sess.run((merged, self.loss),
                                                                     feed_dict={self.input_op: X, target: Y,
                                                                                self.keep_prob: 1})
                print(np.reshape(check_point,[-1])[:5])
                test_writer.add_summary(test_merged,k)
                # if k%1==0:
                #     check_point_reshape=np.reshape(check_point,[-1])
                #     print('最大值为:%s|最小值为:%s|\n均值为:%s|方差为:%s|\n样本数量为:%s|未激活的神经元输出有:%s'%(
                #         np.max(check_point_reshape),
                #         np.min(check_point_reshape),
                #         np.mean(check_point_reshape),
                #         np.var(check_point_reshape),
                #         np.size(check_point_reshape),
                #         np.sum(check_point_reshape==0)
                #     ))
                    # return check_point[0]
                print('————————————————————————————————')
                print(Y[:5])
                print('**************************************************************')
                print('%s|||测试损失：%s'%(cost,test_show))
                print('----------------------分割线------------------------------------')
            k+=1
        saver=train.Saver()
        saver.save(self.sess,config.ckpt_path)

    def predict(self, X):
        '''
        输入X完成整个前向过程
        返回numpy数组
        :param X:
        :return:
        '''
        return self.sess.run(self.pre_op, {self.input_op: X})


if __name__ =='__main__':
    test=SellPredict()
    test.init_forward()
    test.get_all_data()
    one_image=test.train(150000)