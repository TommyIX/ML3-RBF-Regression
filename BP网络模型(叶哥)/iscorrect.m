function bool =iscorrect(x,y,layers,weight,theta)
m1=0;m2=0;p1=0;p2=0;
layers=forward(x,layers,weight,theta);
y_pred=layers{end};
[m1,p1]=max(y_pred);
[m2,p2]=max(y);
if p1==p2
    bool=1;
else
    bool=0;
end
end