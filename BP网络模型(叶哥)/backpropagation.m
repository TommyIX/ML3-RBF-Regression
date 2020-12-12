function [weight,theta]=backpropagation(x,y,layers,weight,theta,vderei,eta)
%%
%先前向传播
layers=forward(x,layers,weight,theta);
%%
%求vderei{end}
vderei{end}=layers{end}.*(1-layers{end}).*(y-layers{end});%1-怎么转化为矩阵
%%
%求vderei
sumwv=0;
for i=length(vderei)-1:-1:2  %从后往前
    for j=1:length(layers{i})
        for k=1:length(layers{i+1})
            sumwv=sumwv+weight{i}(k,j)*vderei{i+1}(k);
        end
        vderei{i}(j)=layers{i}(j)*(1-layers{i}(j))*sumwv;
        sumwv=0;
    end
end
%%
%调整weihgt
for i=1:length(weight)
    for j=1:length(layers{i})
        for k=1:length(layers{i+1})
            weight{i}(k,j)=weight{i}(k,j)+eta*vderei{i+1}(k)*layers{i}(j);
        end
    end
end
%%
%调整theta
for i=1:length(theta)
    for j=1:length(layers{i})
        theta{i}(j)=theta{i}(j)-eta*vderei{i}(j);
    end
end
end