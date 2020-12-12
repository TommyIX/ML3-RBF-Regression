function [x_tra,y_tra]=rexy(path)
data=load(path);
x_all=data.feature;x_all=x_all';
label=data.label;y_all=zeros(3,720);
for i=1:720
    if(label(i)==0);y_all(1,i)=1;end
    if(label(i)==1);y_all(2,i)=1;end
    if(label(i)==2);y_all(3,i)=1;end
end
x_tra=x_all;
y_tra=y_all;
end