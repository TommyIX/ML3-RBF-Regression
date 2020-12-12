function [layers_rec,weight_rec,theta_rec]=one_of_ten(x_tra,y_tra,x_tes,y_tes)
%%

dimhidlay_in=[5,5];%神经网络结构每个隐层的神经元的个数，如[5,4]表示有两个隐层，第一个隐层5个，第二个隐层4个
maxrrr_in=100;%全数据循环次数
epsilon_in=0.1;%对单个数据，训练到均方差要小于的值
%%
[layers_rec,weight_rec,theta_rec,maxtraacc]=traon(x_tra,y_tra,dimhidlay_in,maxrrr_in,epsilon_in);%对每个错误数据训练到指定均方差
[layers_rec,weight_rec,theta_rec,maxtraacc]=traonsp(x_tra,y_tra,dimhidlay_in,maxrrr_in,epsilon_in,layers_rec,weight_rec,theta_rec);%第二轮，对每个错误数据训练到正确
fprintf('训练集准确率:%f\n',maxtraacc);%计算训练集准确率
fprintf('测试集准确率:%f\n',calacc(x_tes,y_tes,layers_rec,weight_rec,theta_rec));%计算测试集准确率
