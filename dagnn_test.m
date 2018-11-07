function [accuracy,net] = dagnn_test(net,imdb,batch)

data=imdb.images.data(:,:,:,batch);
labels=imdb.images.labels(1,batch);

batchSize=100;
num_examples=size(data,4);
epochs=ceil(num_examples/batchSize);

% net=dagnn.DagNN.loadobj(net);
accuracy=0;

for epoch=1:epochs

	batchStart=(epoch-1)*batchSize+1;
	batchEnd=min(epoch*batchSize,num_examples);

	im=data(:,:,:,batchStart:batchEnd);
	label=labels(1,batchStart:batchEnd);
	inputs={'input',im};
	net.eval(inputs);

	outputs=net.vars(net.getVarIndex('prob')).value;
	[~,predictions]=max(squeeze(outputs));
	acc=sum(predictions==label);
	accuracy=accuracy+acc;
end

accuracy=accuracy/num_examples;

fprintf('The test accuracy is %.4f.\n',accuracy);

end