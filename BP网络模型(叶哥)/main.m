clear
clc

%%
load("train.mat");
all_data_feature=feature;
all_data_label=label;
all_number=size(all_data_feature);
train_feature=zeros(uint16(all_number(1)*9/10),all_number(2));
train_label=zeros(3,uint16(all_number(1)*9/10));
test_feature=zeros(uint16(all_number(1)*1/10),all_number(2));
test_label=zeros(3,uint16(all_number(1)*1/10));

%分层抽样分成10份
zero_label=find(all_data_label==0);
one_label=find(all_data_label==1);
two_label=find(all_data_label==2);

zero_size=size(zero_label);
one_size=size(one_label);
two_size=size(two_label);

tens_of_zero=floor(zero_size(2)/10);
tens_of_one=floor(one_size(2)/10);
tens_of_two=floor(two_size(2)/10);

ten_data_feature=zeros(uint16(all_number(1)*1/10),all_number(2),10);
ten_data_label=zeros(3,uint16(all_number(1)*1/10),10);

y_all=zeros(3,720);
for i=1:720
    if(label(i)==0);y_all(1,i)=1;end
    if(label(i)==1);y_all(2,i)=1;end
    if(label(i)==2);y_all(3,i)=1;end
end

zero_random=randperm(zero_size(2),zero_size(2));
one_random=randperm(one_size(2),one_size(2));
two_random=randperm(two_size(2),two_size(2));


index_zero=0;
index_one=0;
index_two=0;

for i=1:10
    if i==10
        left_zero=zero_size(2)-index_zero;
        left_one=one_size(2)-index_one;
        left_two=two_size(2)-index_two;
        ten_data_feature(1:left_zero,:,i)=all_data_feature(zero_label(zero_random(index_zero+1:end)),:);
        ten_data_feature(left_zero+1:left_zero+left_one,:,i)=all_data_feature(one_label(one_random(index_one+1:end)),:);
        ten_data_feature(left_zero+left_one+1:end,:,i)=all_data_feature(two_label(two_random(index_two+1:end)),:);
        ten_data_label(:,1:left_zero,i)=y_all(:,zero_label(zero_random(index_zero+1:end)));
        ten_data_label(:,left_zero+1:left_zero+left_one,i)=y_all(:,one_label(one_random(index_one+1:end)));
        ten_data_label(:,left_zero+left_one+1:end,i)=y_all(:,two_label(two_random(index_two+1:end)));
    else
        ten_data_feature(1:tens_of_zero,:,i)=all_data_feature(zero_label(zero_random(index_zero+1:index_zero+tens_of_zero)),:);
        ten_data_feature(tens_of_zero+1:tens_of_zero+tens_of_one,:,i)=all_data_feature(one_label(one_random(index_one+1:index_one+tens_of_one)),:);
        ten_data_feature(tens_of_zero+tens_of_one+1:tens_of_zero+tens_of_one+tens_of_two,:,i)=all_data_feature(two_label(two_random(index_two+1:index_two+tens_of_two)),:);
        ten_data_label(:,1:tens_of_zero,i)=y_all(:,zero_label(zero_random(index_zero+1:index_zero+tens_of_zero)));
        ten_data_label(:,tens_of_zero+1:tens_of_zero+tens_of_one,i)=y_all(:,one_label(one_random(index_one+1:index_one+tens_of_one)));
        ten_data_label(:,tens_of_zero+tens_of_one+1:tens_of_zero+tens_of_one+tens_of_two,i)=y_all(:,two_label(two_random(index_two+1:index_two+tens_of_two)));
        
        index_zero=index_zero+tens_of_zero;
        index_one=index_one+tens_of_one;
        index_two=index_two+tens_of_two;
        left=all_number(1)*1/10-(tens_of_zero+tens_of_one+tens_of_two);
        ten_data_feature(tens_of_zero+tens_of_one+tens_of_two+1:end,:,i)=all_data_feature(two_label(two_random(index_two+1:index_two+left)),:);
        ten_data_label(:,tens_of_zero+tens_of_one+tens_of_two+1:end,i)=y_all(:,two_label(two_random(index_two+1:index_two+left)));
        index_two=index_two+left;
    end


end
for i=1:10
    order=randperm(72,72);
ten_data_feature(:,:,i)=ten_data_feature(order,:,i);
ten_data_label(:,:,i)=ten_data_label(:,order,i);
end

for i=1:10
    count=1;
    for j=1:10
        if j==i
            test_feature=ten_data_feature(:,:,j);
            test_label=ten_data_label(:,:,j);
        else
            train_feature(all_number(1)*1/10*(count-1)+1:all_number(1)*1/10*(count),:)=ten_data_feature(:,:,j);
            train_label(:,all_number(1)*1/10*(count-1)+1:all_number(1)*1/10*(count))=ten_data_label(:,:,j);
            count=count+1;
        end
    end
    fprintf('第%d轮\n',i);
    one_of_ten(train_feature',train_label,test_feature',test_label);


end








