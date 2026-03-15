import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")
os.chdir("D:\\python编码\\初级机器学习\\实战项目")
df=pd.read_csv('card_transdata.csv')
#显示数据结构
print(df.shape)
#查看所有列名称
print(df.columns)
#查看前五行
print(df.head())

#划分变量
df_sample=df.sample(frac=0.2, random_state=1)
X=df_sample.drop(columns='fraud')
y=df_sample['fraud']

#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=1,stratify=y)

#使用随机森林
from sklearn.ensemble import RandomForestClassifier
base_model=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=30,n_jobs=-1,verbose=1,criterion='gini',max_features='sqrt',oob_score=True,class_weight='balanced')
base_model.fit(X_train,y_train)

#评估分数
from sklearn.metrics import accuracy_score
score=base_model.score(X_test, y_test)
print(f'基准模型的准确率为:{score:4f}')


#查看树的数量和准确率之间的关系
# number=[50,100,150]
# score1=[]
# for i in number:
#     from sklearn.ensemble import RandomForestClassifier
#     temp_model = RandomForestClassifier(n_estimators=i, max_depth=5, min_samples_split=30, n_jobs=-1, verbose=1,
#                                         criterion='gini', max_features='sqrt', oob_score=True, class_weight='balanced')
#     temp_model.fit(X_train, y_train)
#     score2 = base_model.score(X_test, y_test)
#     score1.append(score2)
#     print(f'基准模型的准确率为:{score1:4f}')
#
# plt.figure(figsize=(10,5))
# plt.plot(number,score1,color='blue',label='树的数量和准确率之间的关系')
# plt.xlabel('树的数量')
# plt.ylabel('准确率')
# plt.xticks(number)
# plt.legend()
# plt.grid(True)
# plt.show()

#寻找最佳模型
param_grid = {
    'n_estimators': [50, 100, 150],      # 树数量：测试3个值
    'max_depth': [5, 10, None],    # 树深度：包含None(不限制)
    'min_samples_split': [2, 5, 7]}     # 分裂最小样本数：控制树生长
total_combinations = (len(param_grid['n_estimators']) *
                     len(param_grid['max_depth']) *
                     len(param_grid['min_samples_split']))
print(f"\n🔢 总参数组合数: {total_combinations} 种")

#创建CV网格搜索
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier(random_state=1)
grid_search=GridSearchCV(estimator=rf,
                         param_grid=param_grid,
                         n_jobs=-1,
                         cv=5,
                         scoring='roc_auc', #标准要用auc
                         refit=True)
print("\n⚡ 开始网格搜索...")
grid_search.fit(X_train,y_train)

#最佳参数组合
print(f'最佳参数组合:{grid_search.best_params_}')
#最佳模型
best_model=grid_search.best_estimator_
#使用最佳模型用于评估测试集的准确率
test_score=best_model.score(X_test,y_test)
#预测准确率
y_pred=best_model.predict(X_test)
final_score=accuracy_score(y_test,y_pred)
print(final_score)
#添加auc分数（只关注 “模型能不能识别出欺诈交易”，不受样本比例影响）
from sklearn.metrics import roc_auc_score
y_qizha=best_model.predict_proba(X_test)[:,1]#获取二维数组，预测欺诈的概率
auc_score=roc_auc_score(y_test,y_qizha)
print(f'AUC分数为:{auc_score:.6f}')

#生成分析报告
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,target_names=['不违约(0)','违约(1)'],digits=4))

#混淆矩阵可视化
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=['正常交易', '欺诈交易'],
            yticklabels=['正常交易', '欺诈交易'],
            annot_kws={"size": 14})
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()

#ROC曲线可视化
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_qizha)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlabel('假阳性率（误判正常为欺诈）')
plt.ylabel('真阳性率（正确识别欺诈）')
plt.title('信用卡欺诈检测ROC曲线')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

#特征重要性分析
importances=best_model.feature_importances_
feature_names=X.columns
importances_pd=pd.DataFrame({'特征':feature_names,'重要性':importances}).sort_values(by='重要性',ascending=False)
#可视化
plt.figure(figsize=(12, 6))
bars = plt.barh(importances_pd['特征'][::-1],
                importances_pd['重要性'][::-1],# 倒序让重要性最高的在顶部
                color='steelblue', alpha=0.8)
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
plt.show()

#打印前3个最重要的特征
for i,bar in importances_pd.head(3).iterrows():
    print(f"{i+1:2d}. {bar['特征']:25s} {bar['重要性']:.4f}")








