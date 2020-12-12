function layers = forward(x,layers,weight,theta)
layers{1}=x;
for i=2:length(layers)
    layers{i}=sigmoid(weight{i-1}*layers{i-1}-theta{i});
end
end