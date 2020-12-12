function [weight,theta]=tra2eps(x,y,epsilon,layers,weight,theta,vderei,eta)
while calloss(x,y,layers,weight,theta)>epsilon
    [weight,theta]=backpropagation(x,y,layers,weight,theta,vderei,eta);
end
