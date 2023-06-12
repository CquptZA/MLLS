from skmultilearn.model_selection import IterativeStratification
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.dataset import load_from_arff
from skmultilearn.ensemble import RakelD
import warnings
import pandas as pd
import pdb
import numpy as np
warnings.filterwarnings("ignore")
def Imbalance(X,y):
    countmatrix=[]
    for i in range(y.shape[1]):
        count0=0
        count1=0
        for j in range(y.shape[0]):
            if y[j,i]==1:
                count1+=1
            else:
                count0+=1
        countmatrix.append(count1)
    maxcount=max(countmatrix)
    ImbalanceRatioMatrix=[maxcount/i for i in countmatrix]
    MaxIR=max(ImbalanceRatioMatrix)
    MeanIR=sum(ImbalanceRatioMatrix)/len(ImbalanceRatioMatrix)
    return ImbalanceRatioMatrix,MeanIR,countmatrix
def CalcuNN(df1,n_neighbor):
    nbs=NearestNeighbors(n_neighbors=n_neighbor,metric='euclidean',algorithm='kd_tree').fit(df1)
    euclidean,indices= nbs.kneighbors(df1)
    return euclidean,indices
def Labeltype(X,y):
    ImbalanceRatioMatrix,MeanIR,_=Imbalance(X,y)
    DifferenceImbalanceRatioMatrix=[i-MeanIR for i in ImbalanceRatioMatrix]
    MinLabelIndex=[]
    MajLabelIndex=[]
    count=0
    for i in (DifferenceImbalanceRatioMatrix):
        if i>0:
            MinLabelIndex.append(count)
        else:
            MajLabelIndex.append(count)
        count+=1
    MinLabelName=[]
    MajLabelName=[]
    for i in MinLabelIndex:
        MinLabelName.append(label_names[i][0])
    for i in MajLabelIndex:
        MajLabelName.append(label_names[i][0])
    MinLabeldic=dict(zip(MinLabelIndex,MinLabelName))
    MajLabeldic=dict(zip(MajLabelIndex,MajLabelName))
    return MinLabeldic,MajLabeldic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
class COCOA:
    np.random.seed(10)
    def __init__(self):
#         单标签训练的模型和阈值
        self.classifiers = []
        self.thresholds = []
#         存储三元多分类器和阈值
        self.mclassifiers=[]
        self.mthresholds=[]
#         考虑三个标签
        self.couples=3
        self.classname=np.array([0,1,2])
#  寻找最优阈值       
    def find_optimal_threshold(self, X, Y):
        self.classifiers = []
        self.thresholds = []
        num_labels = Y.shape[1]
        for i in range(num_labels):
            y = Y[:, i]
            clf = LogisticRegression()
            clf.fit(X, y)
            self.classifiers.append(clf)
# 寻找最优阈值         
            y_pred = clf.predict_proba(X)[:, 1]
            f1_scores = []
            thresholds = np.arange(0, 1.01, 0.01)
            for threshold in thresholds:
                y_pred_binary = (y_pred >= threshold).astype(int)
                f1 = f1_score(y, y_pred_binary, average='binary')
                f1_scores.append(f1)
            best_threshold_idx = np.argmax(f1_scores)
            self.thresholds.append(thresholds[best_threshold_idx])
# 训练多类分类器
    def fit(self, X, Y):
        self.find_optimal_threshold(X, Y)
        self.mclassifiers = []
#         self.mthresholds = []
        couples=self.couples
        num_labels = Y.shape[1]
        for i in range(num_labels):
            new_y=self.multilabel_to_multiclass(Y,i)
            clf = LogisticRegression()
            clf.fit(X, new_y)
            self.mclassifiers.append(clf)
#             y_pred = clf.predict_proba(X)
#             pro_matrix=clf.predict_proba(X)
#             y_prob = prob_matrix[:, 2]
# index：当前标签，couples 选择参考的标签数(要小于label-1) y多标签数据集的标签空间
    def multilabel_to_multiclass(self, y, index):
        np.random.seed(10)
        couples=self.couples
        classname=self.classname
        new_y = np.zeros((y.shape[0],))
        # Choose couples labels randomly (excluding the index label)
        other_labels = np.delete(np.arange(y.shape[1]), index)
        couple_labels = np.random.choice(other_labels, couples, replace=False)
        # For index label, set class 2 for samples where it is 1
        index_label = y[:, index]
        
        new_y[index_label == 1] = classname[2]
        # For samples where index is 0 and at least one couple label is 1, set class 1
        couple_labels_present = (y[:, couple_labels] == 1).any(axis=1)
        new_y[(index_label == 0) & couple_labels_present] = classname[1]
        # For all other samples, set class 0
        new_y[(index_label == 0) & (~couple_labels_present)] = classname[0]
        return new_y.astype(int)
# 真实值标签预测
    def predict(self, X):
        num_samples = X.shape[0]
        classname=self.classname
# self.classifiers表示标签的数量=Y.shape[1]
        Y_pred = np.zeros((num_samples, len(self.mclassifiers)))        
        for i, clf in enumerate(self.mclassifiers):
            y_pred = clf.predict_proba(X)[:, -1]
            Y_pred[:, i] = (y_pred >= self.thresholds[i]).astype(int)
        return Y_pred
# 预测概率矩阵输出   
    def predict_proba(self, X):
        num_samples = X.shape[0]
        classname=self.classname
        Y_pred = np.zeros((num_samples, len(self.mclassifiers)))
        for i, clf in enumerate(self.mclassifiers):
            y_pred = clf.predict_proba(X)[:, -1]
            Y_pred[:, i] = y_pred
        return Y_pred
def FeatureSelect(p):
    if p==1:
        return X.toarray(),feature_names
    else:
        if feature_names[1][1]=='NUMERIC':
            featurecount=int(X.shape[1]*p)
            column_variances = np.var(X.toarray(), axis=0)
            sorted_indices = column_variances.argsort()[::-1]
            Selectfeatureindex = sorted_indices[:featurecount]
            Allfeatureindex=[i for i in range(X.shape[1])]
            featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
            new_x=np.delete(X.toarray(),featureindex,axis=1)
            new_featurename=[feature_names[i] for i in Selectfeatureindex]          
        else:
            featurecount=int(X.shape[1]*p)
            Selectfeatureindex=[x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]),key=lambda x: x[1],reverse=True))][:featurecount]
            Allfeatureindex=[i for i in range(X.shape[1])]
            featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
            new_x=np.delete(X.toarray(),featureindex,axis=1)
            new_featurename=[feature_names[i] for i in Selectfeatureindex] 
        return new_x,new_featurename
def LabelSelect():
    b=[]
    new_labelname=[i for i in label_names]
    for i in range(y.shape[1]):
        if y[:,i].sum()<=5:
            b.append(i)
            new_labelname.remove(label_names[i])
    new_y=np.delete(y.toarray(),b,axis=1)
    return new_y,new_labelname

# optmParameter = {
#     'alpha': 2**(-8),  # 2.^[-10:10] % label correlation
#     'beta': 2**(2),  # 2.^[-10:10] % label specific feature 
#     'gamma': 0.1,  # {0.1, 1, 10} % initialization for W
# #     'lamda': 2**(-8),  # instance correlation
# #     'lamda2': 2**(-4),  # common features
#     'maxIter': 50,  # 最大迭代次数
#     'miniLossMargin': 0.0001,  # 两次迭代的最小损失间距 0.0001
#     'bQuiet': 1
# }
# X, y, feature_names, label_names = load_dataset('rcv1subset2','undivided')
# path_to_arff_file = "yahoo-Arts1.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=25,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# path_to_arff_file = "yahoo-Business1.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=28,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# path_to_arff_file = "cal500.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=174,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# path_to_arff_file = "flags.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=7,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# path_to_arff_file = "LLOG-F.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=75,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# path_to_arff_file = "SLASHDOT-F.arff"
# X, y, feature_names, label_names = load_from_arff(
#     path_to_arff_file,
#     label_count=22,
#     label_location="end",
#     load_sparse=False,
#     return_attribute_definitions=True
# )
# X,feature_names=FeatureSelect(0.05)
# y,label_names=LabelSelect()
# c=[]
# for i in range(y.shape[0]):
#     d=y[i,:]
#     if d.sum()==0:
#         c.append(i)
# X=np.delete(X, c, axis=0)
# y=np.delete(y, c, axis=0)
def training(index,method,it):
#     Randomlist=[7,10,19,30,23]
    Randomlist=[41]
    Macro=[]
    Micro=[]
    Hamming_loss=[]
    rankingloss=[]
    macropr=[]
    macroaucroc=[]
    Avgpr=[]
    for i in Randomlist:       
        k_fold = IterativeStratification(n_splits=5,order=1,random_state=i)
        for train,test in k_fold.split(X,y):
            if method==3:
                classifier =ClassifierChain(
                    classifier = DecisionTreeClassifier(random_state=20),
                    require_dense = [False, True]
                )
            elif method==2:
                classifier =BinaryRelevance(
                    classifier = DecisionTreeClassifier(random_state=20),
                    require_dense = [False, True]
                )
            elif method==4:
                classifier = RakelD(
                    base_classifier=DecisionTreeClassifier(random_state=20),
                    base_classifier_require_dense=[False, True],
                    labelset_size=3
                )
            elif method==1:
                classifier =MLkNN(k=10)
            elif method==5:
                classifier=COCOA()
            if index==1:
                X1,y1=X[train],y[train]
#                 print(X1.shape[0])
#                 print("----")
            else:
                dfx=pd.DataFrame(X[train],columns=[x[0] for x in feature_names])
                dfy=pd.DataFrame(y[train],columns=[x[0] for x in label_names])
                X1,y1=X[train],y[train]
                W=CLML(X1,y1,optmParameter1)
                new_X,new_y=SPECIAL(dfx,dfy,W,it,"R",1)      
#                 print(new_X.shape[0]-X1.shape[0])
#                 print("----")
                X1,y1=np.array(new_X),np.array(new_y)
            X2,y2=X[test],y[test]
            classifier.fit(X1,y1)
            ypred = classifier.predict(X2)
            yprob=classifier.predict_proba(X2)
#             yprob=yprob.toarray()
            Macro.append(metrics.f1_score(y2, ypred,average='macro'))
            Micro.append(metrics.f1_score(y2, ypred,average='micro'))
            rankingloss.append(metrics.label_ranking_loss(y2,yprob))                     
            macropr.append(metrics.average_precision_score(y2,yprob,average='macro'))
            macroaucroc.append(metrics.roc_auc_score(y2,yprob,average='macro'))  
            Avgpr.append(metrics.average_precision_score(y2,yprob,average='samples'))
            Hamming_loss.append(metrics.hamming_loss(y2, ypred))  
    Avgpr=[a for a in Avgpr if a==a]   
    MacroF=sum(Macro)/len(Macro)
#     print(max(Macro))
    MicroF=sum(Micro)/len(Micro)
    MacroAUCROC=sum(macroaucroc)/len(macroaucroc)
#     print(max(macroaucroc))
    MacroAUCPR=sum(macropr)/len(macropr)
    RankLoss=sum(rankingloss)/len(rankingloss)
    hamming=sum(Hamming_loss)/len(Hamming_loss)
    Avgprecison=sum(Avgpr)/len(Avgpr)
    res2=(sum((i-MacroF)**2 for i in Macro)/len(Macro))**0.5
    res3=(sum((i-RankLoss)**2 for i in rankingloss)/len(rankingloss))**0.5
    res4=(sum((i-MacroAUCROC)**2 for i in macroaucroc)/len(macroaucroc))**0.5
    res6=(sum((i-MacroAUCPR)**2 for i in macropr)/len(macropr))**0.5
    res7=(sum((i-Avgprecison)**2 for i in Avgpr)/len(Avgpr))**0.5
    MacroF=round(MacroF,4)
    MicroF=round(MicroF,4)
    MacroAUCROC=round(MacroAUCROC,4)
    MacroAUCPR=round(MacroAUCPR,4)
    RankLoss=round(RankLoss,4)
    Avgprecison=round(Avgprecison,4)
    hamming=round(hamming,4)
    res2=round(res2,4)
    res4=round(res4,4)
    res6=round(res6,4)
    res3=round(res3,4)
    res7=round(res7,4)
    print(it,MacroF,MicroF,MacroAUCROC,MacroAUCPR,RankLoss,Avgprecison,hamming)
#     return MacroF,MacroAUCROC,MacroAUCPR,RankLoss,Avgprecison,res2,res4,res6,res3,res7
    return MacroF,MicroF,MacroAUCROC,MacroAUCPR,RankLoss,Avgprecison,hamming