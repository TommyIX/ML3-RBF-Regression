function accrate=calacc(x,y,layers,weight,theta)
    [h,l]=size(x);
    accnum=0;
    for i=1:l
        if iscorrect(x(:,i),y(:,i),layers,weight,theta)
            accnum=accnum+1;
        end
    end
    accrate=accnum/l;
end