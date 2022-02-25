import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.utils import shuffle
import joblib
import jieba

def stopwordslist():  
    stopwords = [line.strip() for line in open('E:/bylw/分类/HGD_StopWords.txt', encoding='utf-8').readlines()]
    return stopwords
stopwords=stopwordslist()

# 中文分词并去除停用词
def seg_depart(sentence):
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()  
    outstr = ''
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


train_data=pd.read_excel('E:/bylw/分类/new/trainset.xlsx')
train_data=shuffle(train_data)
train_data['文本']=train_data.文本.apply(lambda x:seg_depart(str(x)))



trainx=train_data['文本']
trainy=train_data['分类']
TF_Vec=TfidfVectorizer(stop_words=frozenset(stopwords))
trainx_tfvec=TF_Vec.fit_transform(trainx)

DTC=tree.DecisionTreeClassifier()
DTC.fit(trainx_tfvec,trainy)
score_dtc=DTC.score(trainx_tfvec,trainy)
print("未调参的决策树训练集评分：",score_dtc)

RFC=RandomForestClassifier(n_estimators=200,n_jobs=-1)
RFC.fit(trainx_tfvec,trainy)
score_rfc=RFC.score(trainx_tfvec,trainy)
print("未调参的随机森林训练集评分：",score_rfc)

from sklearn.model_selection import cross_val_score
cvscores_rfc = cross_val_score(RFC, trainx_tfvec,trainy, cv=5)
cvscores_dtc = cross_val_score(DTC, trainx_tfvec,trainy, cv=5)
print("未调参的决策树训练集交叉评分：",cvscores_dtc)
print("未调参的随机森林训练集评交叉分：",cvscores_rfc)

test_data=pd.read_excel('E:/bylw/分类/new/testset.xlsx')
test_data=shuffle(test_data)
test_data['文本']=test_data.文本.apply(lambda x:seg_depart(str(x)))
testx=test_data['文本']
testy=test_data['分类']
testx_tfvec=TF_Vec.transform(testx)

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
testx_tfvec_pre=DTC.predict(testx_tfvec)

print("未调参的决策树测试集准确率：",accuracy_score(testy , testx_tfvec_pre)*100)
print("未调参的决策树测试集召回率：",recall_score(testy , testx_tfvec_pre,average='weighted')*100)
print("未调参的决策树测试集精确率：",precision_score(testy , testx_tfvec_pre,average='weighted')*100)
print("未调参的决策树测试集F1得分：",f1_score(testy , testx_tfvec_pre,average='weighted')*100)
testx_tfvec_pre=RFC.predict(testx_tfvec)
print("未调参的随机森林测试集准确率：",accuracy_score(testy , testx_tfvec_pre)*100)
print("未调参的随机森林测试集召回率：",recall_score(testy , testx_tfvec_pre,average='weighted')*100)
print("未调参的随机森林测试集精确率：",precision_score(testy , testx_tfvec_pre,average='weighted')*100)
print("未调参的随机森林测试集F1得分：",f1_score(testy , testx_tfvec_pre,average='weighted')*100)



from sklearn.model_selection import GridSearchCV
import time


parameters = {"min_samples_split":[*range(2,500,10)]
              ,'min_samples_leaf':[*range(1,500,10)]
              ,'max_features':[*range(2,500,10)]
              ,'max_depth':[*range(10,1000,30)]
              }

start_time=time.time()
rfc=RandomForestClassifier(n_estimators=500,n_jobs=-1)
GS = GridSearchCV(rfc,parameters,cv=10)
GS.fit(trainx_tfvec,trainy)
end_time=time.time()
print("调参用时:",end_time-start_time)
print("最佳参数:",GS.best_params_)
testx_tfvec_pre=GS.predict(testx_tfvec)
print("经过调参的随机森林测试集准确率：",accuracy_score(testy , testx_tfvec_pre)*100)
print("经过调参的随机森林测试集召回率：",recall_score(testy , testx_tfvec_pre,average='weighted')*100)
print("经过调参的随机森林测试集精确率：",precision_score(testy , testx_tfvec_pre,average='weighted')*100)
print("经过调参的随机森林测试集F1得分：",f1_score(testy , testx_tfvec_pre,average='weighted')*100)





