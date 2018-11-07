function accuracy = cnn_test(net,imdb,batch)

data=imdb.images.data(:,:,:,batch);
labels=imdb.images.labels(1,batch);

batchSize=256;
num_examples=size(data,4);
epochs=ceil(num_examples/batchSize);

net=vl_simplenn_tidy(net);
res=[];
accuracy=0;

for epoch=1:epochs
	% prepare the test data
	batchStart=(epoch-1)*batchSize+1;
	batchEnd=min(epoch*batchSize,num_examples);
	im=data(:,:,:,batchStart:batchEnd);
	label=labels(batchStart:batchEnd);

	dzdy=[];
	net.layers{end}.class=label;
	res=vl_simplenn(net,im,dzdy,res,...
					'mode','test',...
					'cudnn',true,...
					'conserveMemory',true);
    predictions=squeeze(res(end).x);
    [~,predictions] =max(predictions);
    acc=sum(predictions==label);
	accuracy=accuracy+acc;
end

accuracy=accuracy/num_examples;
fprintf('the test accuracy is %.2f.\n',accuracy);

end





