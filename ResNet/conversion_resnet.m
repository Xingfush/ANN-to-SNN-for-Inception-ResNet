% 设置数据集、模型路径
modelPath='C:\Program Files\MATLAB\R2016b\matconvnet-1.0-beta25\matconvnet-1.0-beta25\data\cifar-resnet\net-sorted.mat';
imdbPath= 'imdb_whiten.mat';

% 加载数据集
% imdb=load(imdbPath);
trainSet=find(imdb.images.set==1);
testSet=find(imdb.images.set==3);
num_examples=10000;

% 加载 CNN 训练模型，这里的模型必须是 deployed 之后的
% load(modelPath);
% net=dagnn.DagNN.loadobj(net);


% -------------------------------------------------------------------------
%                                                         建立层级关系连接图
% -------------------------------------------------------------------------

link=zeros(numel(net.layers),2);

link(1,2)=2;
link(1,1)=1;
for l=2:numel(net.layers)

	if isa(net.layers(l).block,'dagnn.Conv')
		ll=l;
		while true
			i=net.layers(ll).inputIndexes-1;
			if  isa(net.layers(i).block,'dagnn.ReLU')							  
				link(l,1)=i;
				break;
			end
			ll=i;
		end
		link(l,2)=l+1; % 现在以 relu 层作为 scale point


	elseif isa(net.layers(l).block,'dagnn.Sum')
		for i=net.layers(l).inputIndexes-1
			if isa(net.layers(i).block,'dagnn.ReLU')
				link(i,2)=l+1;
				link(i,1)=i;
			elseif isa(net.layers(i).block,'dagnn.Conv') && net.layers(i).block.stride(1)==1
				link(i,2)=l+1;
			elseif isa(net.layers(i).block,'dagnn.Conv') && net.layers(i).block.stride(1)==2
				j=net.layers(i).inputIndexes-1;
				link(j,2)=l+1;
				link(j,1)=j;
				% 考虑仍使用 conv scale，这里不使用 relu scale
				link(i,2)=l+1;
				link(i,1)=j;
			end
		end
	end
end

link(l-1,2)=l-1;


% -------------------------------------------------------------------------
%                                 前向传输，记录激活值，记录 link 相关层激活值
% -------------------------------------------------------------------------
for l=nonzeros(unique(link))'
	net.vars(net.layers(l).outputIndexes).precious=true;
end

% 查看所有卷积层的 max activation
% for l=1:numel(net.layers)
%     if isa(net.layers(l).block,'dagnn.Conv')
%         net.vars(net.layers(l).outputIndexes).precious=true;
%     end
% end

% 开始前向传输，前向传输前不要在 DagNN model 结构中添加任何自定义的元素
batchSize=100;
epochs=ceil(num_examples/batchSize);

% output 做激活值的中转，避免占用过多内存
output=cell(1,numel(net.vars));

net.mode='test';
for epoch=1:10

	batchStart=(epoch-1)*batchSize+1;
	batchEnd=min(epoch*batchSize,num_examples);
	batch=testSet(batchStart:batchEnd);

	im=imdb.images.data(:,:,:,batch);
	label=imdb.images.labels(1,batch);
    inputs={'input',im};
	net.eval(inputs);

    % l 是层遍历，output 是变量遍历，也可以直接变量遍历
	for l=nonzeros(unique(link))'
		output{net.layers(l).outputIndexes}=cat(4,output{net.layers(l).outputIndexes},...
												net.vars(net.layers(l).outputIndexes).value);
    end
    
    % 查看所有卷积层的 max_activation
%     for l=1:numel(net.layers)
%         if isa(net.layers(l).block,'dagnn.Conv')
%             output{net.layers(l).outputIndexes}=cat(4,output{net.layers(l).outputIndexes},...
%                                                     net.vars(net.layers(l).outputIndexes).value);
%         end
%     end

    fprintf('%d epoch forward finised.\n',epoch);
end

% 按变量遍历返回赋值 net.vars
for i=1:numel(net.vars)
	net.vars(i).value=output{i};
end
clear output;

snn=parse_resconv(net,link,99.99); 
opts.dt 			= 0.001;
opts.duration		= 2.500;
opts.report_every	= 0.010;
opts.threshold		=   1.0;
opts.batch 			= 50001:51000;
% 
[performance,stats]=resconv2snn(snn,imdb,opts);
