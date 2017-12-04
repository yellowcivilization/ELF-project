import pyodbc
import os
import numpy as np
import pandas as pd
from ConvNeuroNet.Config import config
import tensorflow as tf
import matplotlib.pyplot as plt
class Preprocess:
    class Item:
        '''
        这个类用于存储数据，一个Item代表一个产品
        '''
        def __init__(self,id,image,sales,features=None):
            '''
            id表示产品编号
            image是产品图片的三维np.array
            :param id:
            :param image:
            '''
            self.id=id
            self.image=image
            self.features=features
            self.sales=sales

    def get_all_image_dir(self):
        '''
        获取所有的图片路径
        :return:
        '''
        image_path_list=list()
        item_list=list()
        for root,dirs,files in os.walk(config.image_dir):
            for file in files:
                if file.find('(')!=-1:
                    continue
                item_list.append(file.lower().split('.jpg')[0])
                image_path_list.append(root+'\\'+file)
        return  image_path_list,item_list
    def test(self):
        a=self.get_all_image_dir()
        for i in a:
            print(i)

    def all_image_to_array(self,n=-1):
        '''
        n限制读入的图片的数量，默认值为-1，-1读入全部图片
        将所有的图片读取成为一个[num_image,height,width,channel]的np.array
        并返回产品id
        :return:
        '''
        result=list()
        file_dir_opt=tf.placeholder(dtype=tf.string)
        image_reader=tf.image.decode_jpeg(tf.read_file(file_dir_opt),channels=3, ratio=1)
        image_format=tf.image.resize_images(image_reader,(200,200),method=1)
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        image_path_list,item_list=self.get_all_image_dir()
        counter=0
        for one_image_dir in image_path_list:
            if n!=-1:
                if counter>=n:
                    break
            result.append(
                sess.run(image_format,feed_dict={file_dir_opt:one_image_dir})
            )
            counter+=1
        return item_list,np.stack(result)

    def extract_feature_of_one_product(self,item_id):
        '''
        输入产品id获得产品特征，包括：
        产品合理价
        产品大类分类号
        产品上新年、月、日
        :param item_id:
        :return:
        '''
        try:
            con = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.1.232;DATABASE=ELF SACK;UID=just;PWD=just123')
            cur = con.cursor()
            sql_result=cur.execute('select count(*) from [ELF SACK].[dbo].[历史子订单明细] where dtsprodcode =\'%s\''%(item_id))
            sales=sql_result.fetchall()[0][0]
            sql_result=cur.execute('select 大类,合理价,上新时间,季节 from [ELF SACK].[dbo].[商品上新信息] where 商品代码=\'%s\''%(item_id))
            features=sql_result.fetchall()[0]
        except:
            return None,None
        time=features[2]
        time=[(time.year-2011)/10.0,time.month/12.0,time.day/30.0]
        features[2]=time
        cur.close()
        con.close()
        return features,sales

    def extract_all_product_feature(self):
        '''
        提取所有的产品特征，储再文件中
        :return:
        '''
        item_list, images = self.all_image_to_array()
        result=list()
        for i in range(len(images)):
            fea,sales=self.extract_feature_of_one_product(item_list[i])
            if fea==None:
                continue
            tmp_item=self.Item(id=item_list[i],image=images[i],sales=sales,features=fea)
            result.append(tmp_item)
        return result
    def extract_all_feature_from_exist_file(self):
        '''
        从已经存在的csv文档中读取产品数据
        :return:
        '''
        id_list, images = self.all_image_to_array()
        dateparse = lambda dates: pd.datetime.strptime(dates.split(' ')[0], '%Y-%m-%d')
        fea_df = pd.read_csv(config.fea_dict,index_col=True,date_parser=dateparse)
        #获取时间数据
        fea_df['year']=fea_df['上新时间'].apply(lambda x:x.year)
        fea_df['month']=fea_df['上新时间'].apply(lambda x:x.month)
        fea_df['day'] = fea_df['上新时间'].apply(lambda x: x.day)
        #产品大类token获取
        item_token_dict=dict()
        k=0
        for item in pd.unique(fea_df['大类']):
            item_token_dict[item]=k
            k+=1
        del k
        fea_df['大类']=fea_df['大类'].apply(lambda x:item_token_dict[x])

        #归一化
        fea_df=(fea_df-fea_df.min())/(fea_df.max()-fea_df.min())
        for id in id_list:
            try:
                fea=fea_df.ix[id]
            except:
                print('%s的数据没找到'%(id))
                continue


    def get_all_data(self,n=-1):
        '''
        根据情况不同选择生成测试数据或者训练数据
        :param n:
        :return:
        '''
        item_list,image=self.all_image_to_array(n)
        con = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.1.232;DATABASE=ELF SACK;UID=just;PWD=just123')
        cur = con.cursor()
        selles_dict=dict()
        sql_result=cur.execute('select dtsprodcode,count(*) from [ELF SACK].[dbo].[历史子订单明细] group by dtsprodcode')
        for record in sql_result:
             selles_dict[record[0]]=record[1]
        data=list()
        if n != -1:
            for index in range(n):
                if item_list[index] not in selles_dict:
                    continue
                tmp=self.Item(item_list[index],image[index],selles_dict[item_list[index]])
                data.append(tmp)
        else:
            for index in range(len(item_list)):
                if item_list[index] not in selles_dict:
                    continue
                tmp = self.Item(item_list[index], image[index], selles_dict[item_list[index]])
                data.append(tmp)
        return data

    def show_a_image(self,X):
        '''
        用于检测结果，输入一个形如[height,width,channel]的np.array,
        将其作为图片展示出来
        :param X:
        :return:
        '''
        plt.imshow(X)
        plt.show()

if __name__ =='__main__':
    test=Preprocess()
    feat=test.extract_all_product_feature()
    k=0
    for i in feat:
        k+=1
        if k%100==0:
            print(i.id)
            print('销量为：%s||时间：%s||季节：%s||大类：%s||价格：%s'%(i.sales,i.features[2],i.features[3],i.features[0],i.features[1]))
            print("——————————————————分割线———————————————————————————————")
            test.show_a_image(i.image)
            input('输入任意键继续')