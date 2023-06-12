from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import random
import pdb
def CalMinkowski(df,W,k,p1,count):
    n = df.shape[0]  # 数据集中样本的数量
    distances = np.zeros((n, n))  # 初始化距离矩阵
    data=np.array(df)
    weights=W
    for i in range(n):
        for j in range(n):          
            distances[i, j] = np.power(np.sum(weights * np.abs(data[i] - data[j])**p1), 1/p1)
    sorted_indices = np.argsort(distances, axis=1)
    sorted_indices = sorted_indices[:, ::1]
    topk_indices= sorted_indices[:, :k]
#     if feature_names[2][1]=='NUMERIC':
#         data=np.array(df)
#         weights=W
#         for i in range(n):
#             for j in range(n):          
#                 distances[i, j] = np.power(np.sum(weights * np.abs(data[i] - data[j])**p1), 1/p1)
#         sorted_indices = np.argsort(distances, axis=1)
#         sorted_indices = sorted_indices[:, ::1]
#         topk_indices= sorted_indices[:, :k]
#     else:
#         sorted_indices = np.argsort(-W)
#         top_k_indices = sorted_indices[:count]
#         data=np.array(df.iloc[:, top_k_indices])
#         for i in range(n):
#             for j in range(n):          
#                 distances[i, j] = np.power(np.sum(np.abs(data[i] - data[j])**p1), 1/p1)
#         sorted_indices = np.argsort(distances, axis=1)
#         sorted_indices = sorted_indices[:, ::1]
#         topk_indices= sorted_indices[:, :k]
    return topk_indices
def assign_weights(arr):
    arr = [abs(x) for x in arr]
    weights = np.exp(arr)
#     arr = [abs(x) for x in arr]
#     for i in range(len(arr)):
#         if arr[i] < 0:
#             arr[i] = 0
#     weights=arr
    return weights
def weighted_minkowski_distance(v1, v2, w):
    # 计算每个维度上的差值
    diff = np.abs(v1 - v2)   
    # 加权每个维度上的差值
    weighted_diff = w * diff   
    # 计算加权闵可夫斯基距离
    distance = np.power(np.sum(np.power(weighted_diff, 2)), 1/2) 
    return distance
def SPECIAL(df1,df2,W,sp,method,p):
    np.random.seed(10)
    n_neighbor=5
    p=p
    card=1
    count=20
    cos_sim=label_similarity(np.array(df2))
    relavent_indices = np.argsort(-cos_sim, axis=1)[:, 1]
    assert len(relavent_indices)== df2.shape[1]
    ML_SMOTE_new_X=df1.copy(deep=True)
    ML_SMOTE_target=df2.copy(deep=True)
    MinLabeldic,MajLabeldic=Labeltype(np.array(df1),np.array(df2))
    ImbalanceRatioMatrix,MeanIR,countmatrix=Imbalance(np.array(df1),np.array(df2))
    MinLabelindex=list(MinLabeldic.keys())
    minIR=[ImbalanceRatioMatrix[i] for i in MinLabelindex]
    tem=['True']*len(MinLabelindex)
    dic=dict(zip(MinLabelindex,tem))  
    for tail_label in MinLabelindex:
#         W_tail_label=LLSF(np.array(df1), np.array(df2.iloc[:,tail_label]), optmParameter)
        W_tail_label=W[:,tail_label]
        featureWeight=assign_weights(W_tail_label)
        if(dic[tail_label]=='False'):
            continue        
        new_IRLbl=max(countmatrix)/countmatrix[tail_label]
        if(new_IRLbl<=MeanIR):
            dic[tail_label]='False'
            continue  
        sub_index=list(df2[df2[MinLabeldic[tail_label]]==1].index)
        sim=cos_sim[tail_label,:].tolist()
        sorted_indices = [i for i, x in sorted(enumerate(sim), key=lambda x: x[1], reverse=True)]
        all_relevant=set()
        for i in range(card):
            tmpindex=sorted_indices[i]
            relavant_index=list(df2[df2[label_names[tmpindex][0]]==1].index)
            all_relevant=all_relevant.union(set(relavant_index))
        assert set(sub_index) <= all_relevant
        # 得到局部标签组数据集
        dfX= df1[df1.index.isin(all_relevant)].reset_index(drop = True)
        dfy= df2[df2.index.isin(all_relevant)].reset_index(drop = True)   
        if dfX.shape[0]==1:
            continue
        dif=df1.shape[0]-dfX.shape[0]
        if sp==0:
            new_X = np.zeros((len(sub_index), dfX.shape[1]))
            target = np.zeros((len(sub_index), dfy.shape[1]))
        else:
            numsample=int((dif-dfX.shape[0])*sp)
            new_X = np.zeros((numsample, dfX.shape[1]))
            target = np.zeros((numsample, dfy.shape[1]))
        if(dfX.shape[0]>6):
            indices=CalMinkowski(dfX,featureWeight,6,p,count)
        else:
            indices=CalMinkowski(dfX,featureWeight,dfX.shape[0],p,count)
        count=0
        for i in range(new_X.shape[0]): 
            tmpindex=i%len(sub_index)
            tmp=sub_index[tmpindex]
            seed=list(all_relevant).index(tmp)
#     随机选择参考样本
            if len(indices[seed,1:])==0:
                continue
            reference = np.random.choice(indices[seed,1:])
            all_point = indices[seed,:]
#             y的所有值
            nn_df = dfy[dfy.index.isin(all_point)]
            ser = nn_df.sum(axis = 0, skipna = True)
#             x的所有值和
            nn_dfX = dfX[dfX.index.isin(all_point)]
            serX = nn_dfX.sum(axis = 0, skipna = True)
            for j in range(dfX.shape[1]): 
                reference = np.random.choice(indices[seed,1:])
                ratio=np.random.random()
                if feature_names[j][1]=='NUMERIC':  
                    new_X[count,j] = dfX.iloc[seed,j] + ratio * (dfX.iloc[reference,j] - dfX.iloc[seed,j])
                elif feature_names[j][1]==['YES', 'NO'] or feature_names[j][1]==['0', '1']:
                    if dfX.iloc[reference,j]==dfX.iloc[seed,j]:
                        new_X[count,j]=dfX.iloc[reference,j]
                    else:
                        if serX[j]>=(n_neighbor+1)/2:
                            new_X[count,j] =1
                        else:
                            new_X[count,j] =0
                else:
                    new_X[count,j] =dfX.iloc[seed,j]
            if (method=="Ranking"):
                target[count] = np.array([1 if val>=((n_neighbor+1)/2) else 0 for val in ser])
            elif (method=="Union"):
                target[count] = np.array([1 if val>0 else 0 for val in ser])
            elif (method=="Rel"):
                dif=[]
                for j in range(dfy.shape[1]): 
                    if dfy.iloc[seed,j]==dfy.iloc[reference,j]:
                        target[count,j]==dfy.iloc[seed,j]
                    else:
                        dif.append(j)
                for j in dif:
                    W_j=W[:,j]
                    feature_jw=assign_weights(W_j)
                    distance1=weighted_minkowski_distance(np.array(dfX.iloc[seed,:]), new_X[count,:],feature_jw)
                    distance2=weighted_minkowski_distance(np.array(dfX.iloc[reference,:]), new_X[count,:],feature_jw)
                    if distance1<=distance2:
                        target[count,j] = dfy.iloc[seed,j]
                    else:
                        target[count,j] = dfy.iloc[reference,j]
            else:
#                 result = np.dot(new_X[count,:], W)
#                 cd1=np.dot(np.array(dfX.iloc[seed,:]), W)
#                 cd2=np.dot(np.array(dfX.iloc[reference,:]), W)
#                 for j in range(dfy.shape[1]):
#                     distance_to_value1 = abs(result[j]- cd1[j])
#                     distance_to_value2 = abs(result[j]- cd2[j])
#                     if distance_to_value1 < distance_to_value2:
#                         target[count,j] = dfy.iloc[seed,j]
#                     else:
#                         target[count,j] = dfy.iloc[reference,j]
                distance1=weighted_minkowski_distance(np.array(dfX.iloc[seed,:]), new_X[count,:],featureWeight)
                distance2=weighted_minkowski_distance(np.array(dfX.iloc[reference,:]), new_X[count,:],featureWeight)
                for j in range(dfy.shape[1]):
                    if distance1<=distance2:
                        target[count,j] = dfy.iloc[seed,j]
                    else:
                        target[count,j] = dfy.iloc[reference,j]
            for j in MinLabelindex:
                if target[count,j]==1:
                    countmatrix[j]+=1
            count+=1
        new_X = pd.DataFrame(new_X,columns=[x[0] for x in feature_names])
        target = pd.DataFrame(target,columns=[y[0] for y in label_names])
        ML_SMOTE_new_X = pd.concat([ML_SMOTE_new_X, new_X], axis=0).reset_index(drop=True)
        ML_SMOTE_target = pd.concat([ML_SMOTE_target, target], axis=0).reset_index(drop=True)    
    return ML_SMOTE_new_X,ML_SMOTE_target