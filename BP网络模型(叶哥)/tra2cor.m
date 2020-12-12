function [weight,theta]=tra2cor(x,y,layers,weight,theta,vderei,eta)
while ~iscorrect(x,y,layers,weight,theta)
    [weight,theta]=backpropagation(x,y,layers,weight,theta,vderei,eta);
end
