function loss=calloss(x,y,layers,weight,theta)
    layers=forward(x,layers,weight,theta);
    loss=0;
    for i=1:length(y)
        loss=loss+(layers{end}(i)-y(i))^2;
    end
    loss=loss/2;
end