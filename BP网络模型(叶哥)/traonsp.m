function [layers_rec,weight_rec,theta_rec,maxtraacc]=traonsp(x_tra,y_tra,dimhidlay_in,maxrrr_in,epsilon_in,layers,weight,theta)
%%
%初始化
x=x_tra(:,1);y=y_tra(:,1);
dimx=length(x);dimy=length(y);
dimhidlay=dimhidlay_in;%每层个数
theta0=0.1;weight0=rand;eta=0.1;
numhidlay=length(dimhidlay);
layers=cell(1,2+numhidlay);layers(1)={zeros(dimx,1)};layers(end)={zeros(dimy,1)};for i=2:numhidlay+1;layers(i)={zeros(dimhidlay(i-1),1)};end
%theta=cell(1,2+numhidlay);for i=1:length(theta);theta{i}=theta0*ones(length(layers{i}),1);end
%weight=cell(1,1+numhidlay);for i=1:length(weight);weight{i}=weight0*ones(length(layers{i+1}),length(layers{i}));end
vderei=cell(1,2+numhidlay);for i=1:length(vderei);vderei{i}=zeros(length(layers{i}),1);end
num_tridat=size(x_tra);num_tridat=num_tridat(2);
%%
max_rrr=maxrrr_in;
epsilon=epsilon_in;
acc_tra_rec=ones(1,num_tridat*max_rrr);
acc_tes_rec=ones(1,num_tridat*max_rrr);
weight_rec=weight;theta_rec=theta;layers_rec=layers;tim_rec=0;
maxtraacc=0;%maxtesacc=0;
for rrr=1:max_rrr
    for i=1:num_tridat
        x=x_tra(:,i);y=y_tra(:,i);
        if ~iscorrect(x,y,layers,weight,theta)
            [weight,theta]=tra2cor(x,y,layers,weight,theta,vderei,eta);
            %[weight,theta]=tra2eps(x,y,epsilon,layers,weight,theta,vderei,eta);
            %[weight,theta]=traonce(x,y,layers,weight,theta,vderei,eta);
        end
        acc_tra_rec(i+num_tridat*(rrr-1))=calacc(x_tra,y_tra,layers,weight,theta);
        %acc_tes_rec(i+num_tridat*(rrr-1))=calacc(x_tes,y_tes,layers,weight,theta);
        if (acc_tra_rec(i+num_tridat*(rrr-1))>maxtraacc)
            weight_rec=weight;theta_rec=theta;layers_rec=layers;tim_rec=i+num_tridat*(rrr-1);
            maxtraacc=acc_tra_rec(i+num_tridat*(rrr-1));
        end
    end
    if maxtraacc>0.999;break;end%结束条件
end
%%
 %plot(1:num_tridat*max_rrr,acc_tra_rec);
%%
%fprintf('max_tra_acc:%10.8f\n',calacc(x_tra,y_tra,layers_rec,weight_rec,theta_rec));
end
