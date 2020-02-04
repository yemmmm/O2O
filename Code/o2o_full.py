# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:43:16 2019

@author: DELL
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pdb
from sklearn.metrics import roc_auc_score
from Features import history_field_feature as hf
from Features import label_field_feature as lf
from Features import week_feature as wf
import time

path_train = r".\数据集\ccf_offline_stage1_train.csv"
path_test = r".\数据集\ccf_offline_stage1_test_revised.csv"

def preproce(offline):
    #数据预处理
    offline["is_manjian"] = offline["Discount_rate"].map(lambda x : 1 if ":" in str(x) else 0)
    #满减全部转换为折扣率
    offline["discount_rate"] = offline["Discount_rate"].map(lambda x : float(x) if ":" not in str(x)
    else (float(str(x).split(":")[0]) - float(str(x).split(":")[1])) / float(str(x).split(":")[0]))
    #满减最低消费
    offline["min_cost_of_manjian"] = offline["Discount_rate"].map(lambda x : -1 if ":" not in str(x) else int(str(x).split(":")[0]))
    #时间类型转换
    offline["date_received"] = pd.to_datetime(offline["Date_received"],format='%Y%m%d')
    #距离控制填充为-1
    offline["Distance"].fillna(-1,inplace=True)
    #判断距离是否为空距离
    offline["null_distance"] = offline["Distance"].map(lambda x : 1 if x == -1 else 0)
    #为offline添加一项received_month为领券月份
    #offline["received_month"] = offline["date_received"].apply(lambda x : x.month)
    if "Date" in offline.columns.tolist():
        #消费时间转换
        offline["date"] = pd.to_datetime(offline["Date"],format="%Y%m%d")
        #为offline数据添加一个date_month为消费月份
        #offline["date_month"] = offline["date"].apply(lambda x : x.month)
    return offline


def get_label(dataset):
    """打标
    领取优惠券后十五天内使用为 1，否者为 0
    Args:
        dataset：DataFrame类型的数据集off_train，包含属性'User_id',"Merchant_id","Coupon_id",
        "Discount_rate","Distance","date_received","Date"
    return:
        打标后的DataFrame数据
    """
    #源数据
    data = dataset.copy()
    #打标
    data["label"] = list(map(lambda x,y : 1 if (x-y).total_seconds()/(60*60*24)<=15 else 0,data["date"],data["date_received"]))
    return data



def get_dataset(history_field,label_field):
    """构造数据集
    Args:us
    Return:
    """
    #特征工程
    history_feat = hf.get_history_field_feature(label_field,history_field) #历史区间特征
    #middle_feat = mf.get_middle_field_feature(label_field,middle_field) #中间区间特征
    label_feat = lf.get_label_field_feature(label_field) #标签区间特征
    week_feat = wf.get_week_feature(label_field)
    simple_feat = get_simple_feature(label_field)
    #构造数据集
    share_characters = list(set(history_feat.columns.tolist()) & set(week_feat.columns.tolist()) & set(label_feat.columns.tolist()))
    #dataset = pd.concat([history_feat,middle_feat.drop(share_characters,axis=1)],axis=1)
    dataset = pd.concat([history_feat,label_feat.drop(share_characters,axis=1)],axis=1)
    dataset = pd.concat([dataset, simple_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat([dataset, week_feat.drop(share_characters, axis=1)], axis=1)
    #删除无用属性并将label放于最后一列
    if "Date" in dataset.columns.tolist():#表示验证集和训练集
        dataset.drop(["Merchant_id","Discount_rate","Date","date_received","date"],axis=1,inplace=True)
        label = dataset["label"].tolist()
        dataset.drop(["label"],axis=1,inplace=True)
        dataset["label"] = label
    else:#表示测试集
        dataset.drop(["Merchant_id","Discount_rate","date_received",],axis=1,inplace=True)
    #修正数据类型
    dataset["User_id"] = dataset["User_id"].map(int)
    dataset["Coupon_id"] = dataset["Coupon_id"].map(int)
    dataset["Date_received"] = dataset["Date_received"].map(int)
    dataset["Distance"] = dataset["Distance"].map(int)
    if "label" in dataset.columns.tolist():
        dataset["label"] = dataset["label"].map(int)
    return dataset

def off_evaluate(validate, off_result):
    """
    线下验证:
    1.评测指标为AUC，但不直接计算AUC，是对每个Coupon_id单独计算核销预测的AUC值，再对所有的优惠券的AUC值求平均作为最终的评价标准
    2.注意计算AUC时标签的真实值必须为2值，所以应先过滤掉全被核销的Coupon_id(该Coupon_id标签的真实值均为1)和全没被核销的Coupon_id
    :param validate: 验证集，DataFrame类型数据集
    :param off_result: 验证集的预测结果，DataFrame类型数据集
    :return: 线下验证的AUC，float类型
    """
    evaluate_data = pd.concat([validate[['Coupon_id', 'label']], off_result[['prob']]], axis=1)
    aucs = 0
    lens = 0
    for name, group in evaluate_data.groupby('Coupon_id'):
        # 如果为2值，set(list(group['label']))为{0， 1}, len(set(list(group['label']))) == 2
        # 否则set(list(group['label']))为{1}或者{0}，len(set(list(group['label']))) == 1
        if len(set(list(group['label']))) == 1:
            continue
        aucs += roc_auc_score(group['label'], group['prob'])
        lens += 1
    auc = aucs / lens
    return auc

def get_simple_feature(label_field):
    data = label_field.copy()
    data["Coupon_id"] = data["Coupon_id"].map(int)
    data["date_received"] = data["Date_received"].map(int)
    data["cnt"] = 1
    #返回特征数据集
    feature = data.copy()
    
    #用户领券数
    keys = ["User_id"]  #主键
    prefixs = "simple_"+"_".join(keys)+"_"
    pivot = pd.pivot_table(data, index = keys, values="cnt", aggfunc = len)
    pivot = pd.DataFrame(pivot).rename(columns = {"cnt":prefixs + "receive_cnt"}).reset_index()
    feature = pd.merge(feature,pivot,on=keys,how="left")
    
    #用户领取特定优惠券数
    keys = ["User_id","Coupon_id"]
    prefixs = "simple_"+"_".join(keys)+"_"
    pivot = pd.pivot_table(data, index = keys, values="cnt", aggfunc = len)
    pivot = pd.DataFrame(pivot).rename(columns = {"cnt":prefixs + "receive_cnt"}).reset_index()
    feature = pd.merge(feature,pivot,on=keys,how="left")
    
    #用户当天领券数
    keys = ["User_id","date_received"]
    prefixs = "simple_"+"_".join(keys)+"_"
    pivot = pd.pivot_table(data, index = keys, values="cnt", aggfunc = len)
    pivot = pd.DataFrame(pivot).rename(columns = {"cnt":prefixs + "receive_cnt"}).reset_index()
    feature = pd.merge(feature,pivot,on=keys,how="left")

    #用户当天领取特定优惠券数
    keys = ["User_id","Coupon_id","date_received"]
    prefixs = "simple_"+"_".join(keys)+"_"
    pivot = pd.pivot_table(data, index = keys, values="cnt", aggfunc = len)
    pivot = pd.DataFrame(pivot).rename(columns = {"cnt":prefixs + "receive_cnt"}).reset_index()
    feature = pd.merge(feature,pivot,on=keys,how="left")

    #用户是否在同一天领取了特定优惠券
    keys = ["User_id","Coupon_id","date_received"]
    prefixs = "simple_"+"_".join(keys)+"_"
    pivot = pd.pivot_table(data, index = keys, values="cnt", aggfunc = lambda x : 1 if len(x)>1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns = {"cnt":prefixs + "receive_is_in_one_day"}).reset_index()
    feature = pd.merge(feature,pivot,on=keys,how="left")
    
    feature.drop(['cnt'],axis=1,inplace = True)
    
    return feature

def model_xgb(train,test):
    """xgboost模型
    调用xgboost模型进行训练预测
    
    """
    params={"booster":"gbtree",
            "objective":"binary:logistic",
            "eval_metric":"auc",
            "silent":1,
            #此处更改学习率
            "eta":0.05,
            "max_depth":5,
            "min_child_weight":1,
            "gamma":0,
            "lambda":1,
            "colsample_bylevel":0.7,
            "colsample_bytree":0.7,
            "subsample":0.9,
            "scale_pos_weight":1
            }
    dtrain = xgb.DMatrix(train.drop(["User_id","Coupon_id","Date_received","label"],axis=1),label=train["label"])
    dtest = xgb.DMatrix(test.drop(["User_id","Coupon_id","Date_received"],axis=1))
    #训练
    pdb.set_trace()
    watchlist={(dtrain,'train')}
    #此处更改迭代次数
    model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
    #预测
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict,columns = ["prob"])
    result = pd.concat([test[["User_id","Coupon_id","Date_received"]],predict],axis=1)
    
    feat_importance = pd.DataFrame(columns=["feature_name","importance"])
    feat_importance["feature_name"] = model.get_score().keys()
    feat_importance["importance"] = model.get_score().values()
    feat_importance.sort_values(["importance"],ascending=False,inplace=True)
    return result,feat_importance

def model_lgb(train,test):
    """
    lgb_train：DataFrame类型，
    
    """
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    lgb_train = lgb.Dataset(train.drop(["User_id","Coupon_id","Date_received","label"],axis=1), train["label"])
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, train, num_boost_round=20, valid_sets=test, early_stopping_rounds=5)
    # 模型预测
    predict = gbm.predict(test, num_iteration=gbm.best_iteration)

    
    pass


if __name__ == "__main__":
    #读取文件
    off_train = pd.read_csv(path_train).sample(frac=0.01).reset_index(drop=True)
    off_test = pd.read_csv(path_test).sample(frac=0.05).reset_index(drop=True)
    
    #预处理
    
    off_train = preproce(off_train)
    off_test = preproce(off_test)
    
    #打标
    off_train = get_label(off_train)
    
    #划分区间
    #训练集历史中间标签区间
    train_history_field = off_train[off_train["date_received"].isin(pd.date_range("2016/3/2",periods=75))]
    #train_middle_field = off_train[off_train["date"].isin(pd.date_range("2016/5/1",periods=15))]
    train_label_field = off_train[off_train["date_received"].isin(pd.date_range("2016/5/16",periods=31))]
    #验证集
    validate_history_field = off_train[off_train["date_received"].isin(pd.date_range("2016/1/16",periods=75))]
    #validate_middle_field = off_train[off_train["date"].isin(pd.date_range("2016/3/16",periods=15))]
    validate_label_field = off_train[off_train["date_received"].isin(pd.date_range("2016/3/31",periods=31))]
    #测试集
    test_history_field = off_train[off_train["date_received"].isin(pd.date_range("2016/4/17",periods=75))]
    #test_middle_field = off_train[off_train["date"].isin(pd.date_range("2016/6/16",periods=15))]
    test_label_field = off_test.copy()
    #构造数据集验证集测试集    

    #构造数据集
    train = get_dataset(train_history_field,train_label_field)
    validate = get_dataset(validate_history_field,validate_label_field)
    test = get_dataset(test_history_field,test_label_field)
    
# =============================================================================
#     #线下测试
#     off_result,off_feat_importance = model_xgb(train,validate.drop(["label"],axis=1))
#     auc = off_evaluate(validate,off_result)
#     print("线下验证auc={}".format(auc ))
# =============================================================================
    
    #线上测试
    big_train = pd.concat([train,validate],axis=0)
    result,feat_importance = model_xgb(big_train,test)
    
    #最终结果
    result.to_csv(r'.\结果提交\base'+time.strftime('%m_%d_%H_%M',time.localtime(time.time()))+r".csv",index=False,header=None)
    


