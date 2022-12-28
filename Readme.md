# CCKS2022 面向数字商务的知识图谱评测任务一：商品常识知识显著性推理

-----

比赛链接:

https://tianchi.aliyun.com/competition/entrance/531955/introduction?spm=5176.12281957.0.0.4c883eafgHtjty

本文模型KG-BERT,初赛复赛成绩均第一，先开源一个base模型，完整代码后续可能上传

文件的结构如下

```
│  config.py                 
│  data_utils.py             
│  main.py                   
│  model.py                  
│  readme.md                 
│  run.py                    
│  utils.py                  
├─data
│      dev_answer.jsonl
│      dev_triple.jsonl
│      new.jsonl
│      train_triple.jsonl
│
├─log
│      train.log
│
└─output
    └─save_dict


```

