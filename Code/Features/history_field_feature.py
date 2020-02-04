# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:15:33 2019

@author: DELL
"""
import pandas as pd
import numpy as np

def get_history_field_feature(label_field,history_field):
    """
    历史区间特征：
        历史区间用户特征
        历史区间优惠券特征
        历史区间商家特征
        历史前用户商家特征
        历史区间用户优惠券特征
        历史区间商家优惠券特征
    """
    #提取特征
    u_feat = get_history_field_user_feature(label_field,history_field)
    c_feat = get_history_field_coupon_feature(label_field,history_field)
    m_feat = get_history_field_merchant_feature(label_field,history_field)
    um_feat = get_history_field_user_merchant_feature(label_field,history_field)
    uc_feat = get_history_field_user_coupon_feature(label_field,history_field)
    mc_feat = get_history_field_merchant_coupon_feature(label_field,history_field)

    #连接数据集
    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat,u_feat,on=["User_id"],how="left")
    history_feat = pd.merge(history_feat,c_feat,on=["Coupon_id"],how="left")
    history_feat = pd.merge(history_feat,m_feat,on=["Merchant_id"],how="left")
    history_feat = pd.merge(history_feat,um_feat,on=["User_id","Merchant_id"],how="left")
    history_feat = pd.merge(history_feat,uc_feat,on=["User_id","Coupon_id"],how="left")
    history_feat = pd.merge(history_feat,mc_feat,on=["Coupon_id","Merchant_id"],how="left")
    
    return history_feat


def get_history_field_user_feature(label_field,history_field):
    """
    历史区间用户特征:
        用户领券数
        用户核销数
        用户未核销数
        用户核销率
        用户核销的最小距离
        用户核销的最大距离
        用户核销的平均距离
        用户领取不同优惠券的数量
        用户核销不同优惠券的数量
        用户领券核销最小时间间隔
        用户领券核销平均时间间隔
        用户核销的最小折扣率
        用户核销的最大折扣率
        用户核销的平均折扣率
    
    """
    data = history_field.copy()
    #浮点类型转换为int类型
    data["Coupon_id"] = data["Coupon_id"].map(int)
    data["Date_received"] = data["Date_received"].map(int)
    data["cnt"] = 1
    
    #用户特征
    keys = ["User_id"]
    prefixs="history_field_"+u"_".join(keys)+"_"
    
    u_feat = label_field[keys].drop_duplicates(keep="first")
    
    #用户领券数
    #pivot_table透视表，索引为keys，取值为cnt，调用len函数求得长度
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    #将pivot转化成DataFrame类型并更改列名
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    #将pivot添加到u_feat上，关键字为User
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    #填充空值为0，downcast=“infer”不改变数据类型
    u_feat.fillna(0, downcast='infer', inplace=True)
    
    #用户核销数
    pivot = pd.pivot_table(data[data["Date"].map(lambda x : str(x)!='nan')],index = keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
# =============================================================================
#     #用户未核销数
#     pivot = pd.pivot_table(data[data["Date"].map(lambda x : str(x)=='nan')],index = keys,values="cnt",aggfunc=len)
#     pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_not_consume_cnt"}).reset_index()
#     u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
#     u_feat.fillna(0,downcast="infer",inplace=True)
# =============================================================================
    
    #用户核销率
    u_feat[prefixs+"received_and_consum_rate"] = list(map(lambda x,y: x/y if y!=0 else 0,
                                                      u_feat[prefixs+"received_and_consume_cnt"],
                                                      u_feat[prefixs+"received_cnt"]))
    
    #用户核销的最小距离
# =============================================================================
#     pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.min([np.nan if i==-1 else i for i in x]))
#     pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consum_min_distance"}).reset_index()
#     u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
#     u_feat.fillna(-1,downcast="infer",inplace=True)
# =============================================================================
    
    #用户核销的最大距离
# =============================================================================
#     pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.max([np.nan if i==-1 else i for i in x]))
#     pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consum_max_distance"}).reset_index()
#     u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
#     u_feat.fillna(-1,downcast="infer",inplace=True)
# =============================================================================
    
# =============================================================================
#     #用户核销的平均距离
#     pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.mean([np.nan if i==-1 else i for i in x]))
#     pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consum_mean_distance"}).reset_index()
#     u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
#     u_feat.fillna(-1,downcast="infer",inplace=True)
# =============================================================================
    
    #用户领取不同优惠券的数量
    pivot = pd.pivot_table(data,index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"Coupon_id":prefixs+"receive_different_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户核销不同优惠券的数量
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"Coupon_id":prefixs+"receive_and_consume_different_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    


    tmp = data[data["label"]==1]
    tmp["gap"] = (tmp["date"]-tmp["date_received"]).map(lambda x:x.total_seconds()/(60*60*24))

    #用户领券核销最小时间间隔    
    pivot = pd.pivot_table(tmp,index=keys,values="gap",aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={"gap":prefixs+"receive_and_consume_min_time"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(-1,downcast="infer",inplace=True)
    
    #用户领券核销平均时间间隔
    pivot = pd.pivot_table(tmp,index=keys,values="gap",aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={"gap":prefixs+"receive_and_consume_mean_time"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(-1,downcast="infer",inplace=True)
    
    #用户核销的最小折扣率
# =============================================================================
#     pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.min)
#     pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"consume_min_discount_rate"}).reset_index()
#     u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
#     u_feat.fillna(-1,downcast="infer",inplace=True)
# =============================================================================
    
    #用户核销的最大折扣率
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"consume_max_discount_rate"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(-1,downcast="infer",inplace=True)
    
    #用户核销的平均折扣率
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"consume_mean_discount_rate"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,on=keys,how="left")
    u_feat.fillna(-1,downcast="infer",inplace=True)

    #统计特征
    u_feat.fillna(0, downcast = 'infer', inplace = True)

    return u_feat

def get_history_field_coupon_feature(label_field,history_field):
    """
    历史区间优惠券特征：
        优惠券被领取次数
        优惠券被核销数
        优惠券被核销率
        优惠券未被核销数
        优惠券未被核销率
        优惠券被核销的平均时间间隔
    """
    data = history_field.copy()
    #浮点类型转换为int类型
    data["Coupon_id"] = data["Coupon_id"].map(int)
    data["Date_received"] = data["Date_received"].map(int)
    data["cnt"] = 1

    keys = ["Coupon_id"]
    prefixs="history_field_"+u"_".join(keys)+"_"
    
    c_feat = label_field[keys].drop_duplicates(keep="first")

    #优惠券领取数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,on=keys,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)
    
    #优惠券被核销数
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,on=keys,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)

    #优惠券被核销率
    c_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                          c_feat[prefixs+"received_and_consume_cnt"],
                                                          c_feat[prefixs+"received_cnt"]))
    
    #优惠券未被核销数
    pivot = pd.pivot_table(data[data["label"]!=1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_not_consume_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,on=keys,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)
    
    #优惠券未被核销率
    c_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                          c_feat[prefixs+"received_and_not_consume_cnt"],
                                                          c_feat[prefixs+"received_cnt"]))
    
    #优惠券被核销平均时间间隔
    # 筛选出label为1即领券15天内核销的样本
    tmp = data[data['label'] == 1]
    # 核销与领券的时间间隔，以天为单位
    tmp['gap'] = (tmp['date'] - tmp['date_received']).map(lambda x: x.total_seconds() / (60 * 60 * 24))

    pivot = pd.pivot_table(tmp,index=keys,values="gap",aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={"gap":prefixs+"received_and_consume_mean_time"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,on=keys,how="left")
    c_feat.fillna(-1,downcast="infer",inplace=True)

    c_feat.fillna(0, downcast = 'infer', inplace = True)
    return c_feat

def get_history_field_merchant_feature(label_field,history_field):
    """
    历史区间商家特征：
        商家的优惠券被领取数
        商家的优惠券被不同客户领取数
        商家的优惠券被核销数
        商家的优惠券被核销率
        商家有多少种优惠券
        商家的优惠券被弃用数
        商家的优惠券被弃用率
        商家的客户数
        商家的优惠券被核销的平均距离
        商家优惠券被核销的距离中位数
        商家优惠券被核销的平均折扣率
        商家优惠券被核销的折扣率最大值
        商家优惠券被核销的折扣率最小值
        商家优惠券被核销的平均时间间隔
        ......
    """
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 商家特征
    keys = ['Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    m_feat = label_field[keys].drop_duplicates(keep='first')

    #商家的优惠券被领取数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家优惠券被不同用户领取数
    pivot = pd.pivot_table(data,index=keys,values="User_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"User_id":prefixs+"received_user_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家的优惠券被核销数
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家的优惠券被核销率
    m_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                          m_feat[prefixs+"received_and_consume_cnt"],
                                                          m_feat[prefixs+"received_cnt"]))

    #商家的优惠券被弃用数
    m_feat[prefixs+"received_and_not_comsume_cnt"] = list(map(lambda x,y:(y-x) if y!=0 else 0,
                                                          m_feat[prefixs+"received_and_consume_cnt"],
                                                          m_feat[prefixs+"received_cnt"]))
    
    #商家的优惠券被弃用率
    m_feat[prefixs+"received_and_not_comsume_rate"] = list(map(lambda x,y:(y-x) if y!=0 else 0,
                                                          m_feat[prefixs+"received_and_consume_cnt"],
                                                          m_feat[prefixs+"received_cnt"]))
    
    #商家提供多少种优惠券
    pivot = pd.pivot_table(data,index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"Coupon_id":prefixs+"differ_coupon_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家15天内被核销的优惠券有多少种
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"Coupon_id":prefixs+"received_amd_consume_differ_coupon_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家被核销的优惠券占总的优惠券的比例
    m_feat["received_and_consume_differ_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                      m_feat[prefixs+"differ_coupon_cnt"],
                                                      m_feat[prefixs+"received_amd_consume_differ_coupon_cnt"]))
    
    #商家的客户数
    pivot = pd.pivot_table(data,index=keys,values="User_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"User_id":prefixs+"differ_User_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家的优惠券被核销的最小距离
    #预处理时将距离空值赋值为了-1
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.min([np.nan if i==-1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consume_Distance_min"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(-1,downcast="infer",inplace=True)
    
    #商家的优惠券被核销的最大距离
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.max([np.nan if i==-1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consume_Distance_max"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(-1,downcast="infer",inplace=True)
    
    #商家的优惠券被核销的平均距离
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.mean([np.nan if i==-1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consume_Distance_mean"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(-1,downcast="infer",inplace=True)

    #商家的优惠券被核销的距离中位数（没有直接求取众数的函数
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="Distance",aggfunc=lambda x:np.median([np.nan if i==-1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={"Distance":prefixs+"received_and_consume_Distance_median"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(-1,downcast="infer",inplace=True)

    #商家优惠券被核销的平均折扣率
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"received_and_consume_discount_rate_mean"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家优惠券被核销的最小折扣率
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"received_and_consume_discount_rate_min"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家优惠券被核销的最大折扣率
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="discount_rate",aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={"discount_rate":prefixs+"received_and_consume_discount_rate_max"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    tmp = data[data["label"]==1]
    tmp["gap"] = (tmp["date"]-tmp["date_received"]).map(lambda x:x.total_seconds()/(60*60*24))
    #商家优惠券被领取到被核销的平均时间间隔
    pivot = pd.pivot_table(tmp,index=keys,values="gap",aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={"gap":prefixs+"received_to_consume_gap_mean"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家优惠券被领取到被核销的最小时间间隔
    pivot = pd.pivot_table(tmp,index=keys,values="gap",aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={"gap":prefixs+"received_to_consume_gap_min"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    m_feat.fillna(0, downcast = 'infer', inplace = True)
    return m_feat

def get_history_field_user_merchant_feature(label_field,history_field):
    """
    历史区间用户商家特征：
        该用户在该商家的领券数
        该用户在该商家的领券核销数
        该用户在该商家的领券核销率
    """
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1
    
    # 用户商家特征
    keys = ["User_id","Merchant_id"]
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    um_feat = label_field[keys].drop_duplicates(keep='first')
    
    #该用户在该商家的领券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    um_feat = pd.merge(um_feat,pivot,how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户在该商家的领券核销数
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    um_feat = pd.merge(um_feat,pivot,how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户在该商家的领券核销率
    um_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                      um_feat[prefixs+"received_and_consume_cnt"],
                                                      um_feat[prefixs+"received_cnt"]))
    
    um_feat.fillna(0, downcast = 'infer', inplace = True)
    return um_feat
    
def get_history_field_user_coupon_feature(label_field,history_field):
    """
    历史区间用户优惠券特征：
        该用户领取该优惠券的数量
        该用户核销该优惠券的数量
        该用户核销该优惠券的核销率
    """
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1
    
    # 用户商家特征
    keys = ["User_id","Coupon_id"]
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    uc_feat = label_field[keys].drop_duplicates(keep='first')
    
    #该用户领取该优惠券的数量
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    uc_feat = pd.merge(uc_feat,pivot,how="left")
    uc_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户核销该优惠券的数量
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    uc_feat = pd.merge(uc_feat,pivot,how="left")
    uc_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户核销该优惠券的核销率
    uc_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                           uc_feat[prefixs+"received_and_consume_cnt"],
                                                           uc_feat[prefixs+"received_cnt"]))

    uc_feat.fillna(0, downcast = 'infer', inplace = True)
    return uc_feat

def get_history_field_merchant_coupon_feature(label_field,history_field):
    """
    历史区间商家优惠券特征：
        该商家的该优惠券领券数
        该商家的该优惠券核销数
        该商家的该优惠券核销率
    """
    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1
    
    # 用户商家特征
    keys = ["Merchant_id","Coupon_id"]
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    mc_feat = label_field[keys].drop_duplicates(keep='first')
    
    #该商家的该优惠券领券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    mc_feat = pd.merge(mc_feat,pivot,how="left")
    mc_feat.fillna(0,downcast="infer",inplace=True)
    
    #该商家的该优惠券核销数
    pivot = pd.pivot_table(data[data["label"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_and_consume_cnt"}).reset_index()
    mc_feat = pd.merge(mc_feat,pivot,how="left")
    mc_feat.fillna(0,downcast="infer",inplace=True)
    
    #该商家的该优惠券核销率
    mc_feat[prefixs+"received_and_consume_rate"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                           mc_feat[prefixs+"received_and_consume_cnt"],
                                                           mc_feat[prefixs+"received_cnt"]))
    
    return mc_feat
    

    











