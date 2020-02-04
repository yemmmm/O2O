import pandas as pd



def get_week_feature(label_field):
    """
    根据date_received得到一些日期特征:
        1.根据date_received列得到领券日期是周几，新增一列week表示该信息
        2.根据week列得到领券日期是否为休息日，新增一列is_weekend表示该信息
        3.将week进行one-hot离散为week_0, week_1, week_2, week_3, week_4, week_5, week_6
    """
    data = label_field.copy()
    feat = data.copy()
    #weekday返回0-6
    feat['week'] = feat['date_received'].map(lambda x: x.weekday())
    # 判断领券日是否为休息日
    feat['is_weekday'] = feat['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)
    # one-hot离散星期几,prefix为离散之后的列前缀
    feat = pd.concat([feat, pd.get_dummies(feat['week'], prefix='week')], axis=1)
    # 重置index
    feat.index = range(len(feat))
    # 返回
    return feat