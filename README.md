Aster组说明文档
===========================

#### 环境依赖
- Anaconda    version 1.7.2        
- WIN7或WIN10 64位
- pycharm编辑器

#### 部署步骤
1. 安装Anconda环境
2. **TextCnn.yml**可直接安装所需依赖库 
3. 添加系统环境变量: `conda env create -f Aster.yaml`

#### 目录结构描述
>├──Readme.md      //帮助  
>├──stopworls.txt  //停用词  
>├──TextCnn.yml    //anaconda虚拟环境  
>├──textcnn_model.h5    //模型     
>├──w2v.model      //词向量模型    
>├──w2v.model.trainables.syn1neg.npy   //词向量模型依赖        
>├──w2v.model.wv.vectors.npy       ////词向量模型依赖    
>├──main.py         //控制台入口程序        

>├──textCnnQT  //主文件夹
>>   ├── dir_pic.py     
>>   ├── dir_pic.ui     
>>   ├── index.py   
>>   ├── index.ui   
>>   ├── main.py    //图形化界面入口程序   

>├──utils   //依赖文件夹
>>   ├── const.py     
>>   ├── data.ui     
>>   ├── gui_util.py   
>>   ├── myjieba.ui   
>>   ├── textCnn.py     
>>   ├── word2vec.py

>├──tmp     //储存临时数据（必须要有）

## 实例
**接下来使用Anconda Console来打开程序**
### 创建虚拟环境
```
conda env create -f Aster.yaml
```
```
pip install -r requirements.txt
```
### 运行程序
使用**console控制台**运行
```
python main.py -h # 查看帮助
```
使用**可视化**运行,需要**进入textCnnQT文件夹**
```
python main.py
```
