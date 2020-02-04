import pandas as pd
import numpy as np
import pdb



def get_label_field_feature(label_field):
    """
    标签区间特征：
        标签区间用户特征
        标签区间商家特征
        标签区间优惠券特征
        标签区间折扣率特征
        标签区间用户-商家特征
        标签区间用户-优惠券特征
        标签区间用户-折扣率特征
    """
    u_feat = get_label_field_user_feature(label_field)
    m_feat = get_label_field_merchant_feature(label_field)
    c_feat = get_label_field_coupon_feature(label_field)
    dr_feat = get_label_discount_rate_feature(label_field)
    um_feat = get_label_user_merchant_feature(label_field)
    uc_feat = get_label_user_coupon_feature(label_field)
    udr_feat = get_label_user_discount_rate_feature(label_field)
    
    share_characters = list(set(u_feat.columns.tolist()) & set(m_feat.columns.tolist()))

    label_feat = pd.concat([u_feat, m_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, c_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, dr_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, um_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, uc_feat.drop(share_characters, axis=1)], axis=1)
    label_feat = pd.concat([label_feat, udr_feat.drop(share_characters, axis=1)], axis=1)
    
    return label_feat

def get_label_field_user_feature(label_field):
    """
    标签区间用户特征：
        用户领券数
        用户是否第一次领券
        用户是否最后一次领券
        用户领取非满减优惠券的数目
        用户领取满减券的数目
        用户领取的非满减券所占比例
        用户领取的满减券所占比例
    """
    data = label_field.copy()
    data['Date_received'] = data['Date_received'].map(int)
    data["cnt"] = 1
    
    keys = ["User_id"]
    prefixs = "label_field_"+"_".join(keys)+"_"
    u_feat = data.copy()
    
    #用户领券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
    tmp = data[keys+["Date_received"]].sort_values(["Date_received"],ascending=True)
    #用户是否是第一次领券
    #将领券时间按升序排序，删除用户ID重复项，保留第一项，添加新列为1，merge；连接时tmp不存在的话就为NaN，最后空值赋值为0
    first = tmp.drop_duplicates(keys,keep="first")
    first[prefixs+"is_first_received"] = 1
    u_feat = pd.merge(u_feat,first,on=keys+["Date_received"],how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户是否是最后一次领券
    last = tmp.drop_duplicates(keys,keep="last")
    last[prefixs+"is_last_received"] = 1
    u_feat = pd.merge(u_feat,last,on=keys+["Date_received"],how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户领取非满减优惠券的数量
    pivot = pd.pivot_table(data[data["is_manjian"]!=1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_not_manjian_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户领取满减优惠券的数量
    pivot = pd.pivot_table(data[data["is_manjian"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_manjian_cnt"}).reset_index()
    u_feat = pd.merge(u_feat,pivot,how="left")
    u_feat.fillna(0,downcast="infer",inplace=True)
    
# =============================================================================
#     #用户领取非满减券所占比例
#     u_feat[prefixs+"received_is_not_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
#                                                   u_feat[prefixs+"received_is_not_manjian_cnt"],
#                                                   u_feat[prefixs+"received_cnt"]))
#     
#     #用户领取满减券所占比例
#     u_feat[prefixs+"received_is_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
#                                                   u_feat[prefixs+"received_is_manjian_cnt"],
#                                                   u_feat[prefixs+"received_cnt"]))
# =============================================================================
    
    return u_feat

def get_label_field_merchant_feature(label_field):
    """
    标签区间商家特征：
        商家被领取优惠券数
        商家被领取的满减优惠券数
        商家被领取的非满减优惠券数
        商家被领取的满减优惠券占比
        商家被领取的非满减优惠券占比
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["Merchant_id"]
    prefixs = "label_field_"+"_".join(keys)+"_"    
    m_feat = data.copy()
    
    #商家被领取优惠券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家被领取的满减优惠券数
    pivot = pd.pivot_table(data[data["is_manjian"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_manjian_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)

    #商家被领取的非满减优惠券数
    pivot = pd.pivot_table(data[data["is_manjian"]!=1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_not_manjian_cnt"}).reset_index()
    m_feat = pd.merge(m_feat,pivot,how="left")
    m_feat.fillna(0,downcast="infer",inplace=True)
    
    #商家被领取的满减优惠券占比
    m_feat[prefixs+"received_is_not_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                  m_feat[prefixs+"received_is_not_manjian_cnt"],
                                                  m_feat[prefixs+"received_cnt"]))
    
    #用户领取满减券所占比例
    m_feat[prefixs+"received_is_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
                                                  m_feat[prefixs+"received_is_manjian_cnt"],
                                                  m_feat[prefixs+"received_cnt"])) 
    
    return m_feat

def get_label_field_coupon_feature(label_field):
    """
    标签区间优惠券特征：
        该优惠券被领取次数
        该优惠券被多少用户领取
        该优惠券被多少商家发放
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["Coupon_id"]
    prefixs = "label_field_"+"_".join(keys)+"_"    
    c_feat = data.copy()
    
    #该优惠券被领取次数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)
    
    #该优惠券被多少用户领取
    pivot = pd.pivot_table(data,index=keys,values="User_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)
    
    #该优惠券被多少商家发放
    pivot = pd.pivot_table(data,index=keys,values="Merchant_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    c_feat = pd.merge(c_feat,pivot,how="left")
    c_feat.fillna(0,downcast="infer",inplace=True)
    
    return c_feat

def get_label_discount_rate_feature(label_field):
    """
    标签区间折扣率特征：
        该折扣率优惠券被领取数
        该折扣率下有多少种优惠券
        该折扣率的优惠券被多少商家发放
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["discount_rate"]
    prefixs = "label_field"+"_".join(keys)+"_"    
    dr_feat = data.copy()
    
    #该折扣率优惠券被领取数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    dr_feat = pd.merge(dr_feat,pivot,how="left")
    dr_feat.fillna(0,downcast="infer",inplace=True)
    
    #该折扣率下有多少种优惠券
    pivot = pd.pivot_table(data,index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    dr_feat = pd.merge(dr_feat,pivot,how="left")
    dr_feat.fillna(0,downcast="infer",inplace=True)
    
    #该折扣率的优惠券被多少商家发放
    pivot = pd.pivot_table(data,index=keys,values="Merchant_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    dr_feat = pd.merge(dr_feat,pivot,how="left")
    dr_feat.fillna(0,downcast="infer",inplace=True)
    
    return dr_feat
    
def get_label_user_merchant_feature(label_field):
    """
    标签区间用户-商家特征：
        该用户在该商家的领券数
        该用户是否在该商家第一次领券
        该用户是否在该商家最后一次领券
        该用户在该商家领取的满减优惠券数量
        该用户在该商家领取的非满减优惠券数量
        该用户在该商家领取的满减优惠券占比
        该用户在该商家领取的非满减优惠券占比
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["User_id","Merchant_id"]
    prefixs = "label_field_"+"_".join(keys)+"_"    
    um_feat = data.copy()
    
    #该用户在该商家的领券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    um_feat = pd.merge(um_feat,pivot,how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    tmp = data[keys+["Date_received"]].sort_values(["Date_received"],ascending=True)
    #该用户是否在该商家第一次领券
    first = tmp.drop_duplicates(keys,keep="first")
    first[prefixs+"is_first_received"]=1
    um_feat = pd.merge(um_feat,first,on=keys+["Date_received"],how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户是否在该商家最后一次领券
    last = tmp.drop_duplicates(keys,keep="last")
    last[prefixs+"is_last_received"]=1
    um_feat = pd.merge(um_feat,last,on=keys+["Date_received"],how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户在该商家领取非满减优惠券的数量
    pivot = pd.pivot_table(data[data["is_manjian"]!=1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_not_manjian_cnt"}).reset_index()
    um_feat = pd.merge(um_feat,pivot,how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
    #用户在该商家领取满减优惠券的数量
    pivot = pd.pivot_table(data[data["is_manjian"]==1],index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_is_manjian_cnt"}).reset_index()
    um_feat = pd.merge(um_feat,pivot,how="left")
    um_feat.fillna(0,downcast="infer",inplace=True)
    
# =============================================================================
#     #该用户在该商家领取非满减券所占比例
#     um_feat[prefixs+"received_is_not_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
#                                                   um_feat[prefixs+"received_is_not_manjian_cnt"],
#                                                   um_feat[prefixs+"received_cnt"]))
#     
#     #该用户在该商家领取满减券所占比例
#     um_feat[prefixs+"received_is_manjian_rete"] = list(map(lambda x,y:x/y if y!=0 else 0,
#                                                   um_feat[prefixs+"received_is_manjian_cnt"],
#                                                   um_feat[prefixs+"received_cnt"]))
# =============================================================================
    
    return um_feat

def get_label_user_coupon_feature(label_field):
    """
    标签区间用户优惠券特征：
        该用户领取该优惠券的数量
        该用户是否是第一次领取该优惠券
        该用户是否最后一次领取该优惠券
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["User_id","Coupon_id"]
    prefixs = "label_field_"+"_".join(keys)+"_"    
    uc_feat = data.copy()
    
    #该用户领取该优惠券的数量
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    uc_feat = pd.merge(uc_feat,pivot,how="left")
    uc_feat.fillna(0,downcast="infer",inplace=True)
    
    tmp = data[keys+["Date_received"]].sort_values(["Date_received"],ascending=True)
    #该用户是否是第一次领取该优惠券
    first=tmp.drop_duplicates(keys,keep="first")
    first[prefixs+"is_first_received"] = 1
    uc_feat = pd.merge(uc_feat,first,on=keys+["Date_received"],how="left")
    uc_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户是否最后一次领取该优惠券
    last=tmp.drop_duplicates(keys,keep="last")
    last[prefixs+"is_last_received"] = 1
    uc_feat = pd.merge(uc_feat,last,on=keys+["Date_received"],how="left")
    uc_feat.fillna(0,downcast="infer",inplace=True)
    
    return uc_feat

def get_label_user_discount_rate_feature(label_field):
    """
    标签区间用户折扣率特征：
        该用户领取该折扣率的优惠券数
        该用户领取该折扣率的优惠券种类数
    """
    data = label_field.copy()
    data["cnt"]=1
    keys = ["User_id","discount_rate"]
    prefixs = "label_field_"+"_".join(keys)+"_"    
    udr_feat = data.copy()
    
    #该用户领取该折扣率的优惠券数
    pivot = pd.pivot_table(data,index=keys,values="cnt",aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    udr_feat = pd.merge(udr_feat,pivot,how="left")
    udr_feat.fillna(0,downcast="infer",inplace=True)
    
    #该用户领取该折扣率的优惠券种类数
    pivot = pd.pivot_table(data,index=keys,values="Coupon_id",aggfunc=lambda x:len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={"cnt":prefixs+"received_cnt"}).reset_index()
    udr_feat = pd.merge(udr_feat,pivot,how="left")
    udr_feat.fillna(0,downcast="infer",inplace=True)
    
    return udr_feat
