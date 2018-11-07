% 设置数据集、模型路径
modelPath='C:\Program Files\MATLAB\R2016b\matconvnet-1.0-beta25\matconvnet-1.0-beta25\data\cifar-inception-v4-4\net-deployed.mat';
imdbPath= 'imdb_whiten.mat';

% 加载数据集，一定要要事先检查数据集对不对
% imdb=load(imdbPath);
trainSet=find(imdb.images.set==1);
testSet=find(imdb.images.set==3);
num_examples=60000;

% 加载 CNN 训练模型，这里的模型必须是 deployed 之后的
% net=load(modelPath);
% net=dagnn.DagNN.loadobj(net);


% -------------------------------------------------------------------------
%                                                         建立层级关系连接图
% -------------------------------------------------------------------------

link=zeros(numel(net.layers),2); % 存储 layers 序号，这部分不改变网络结构

link(1,2)=1; % 第一层直接赋值，避免 l = 0.

for l=2:numel(net.layers) % 从 2 开始 
	
	% 按输入进行回溯找到上一个 conv 层或者 concat 层，concat 优先
	if isa(net.layers(l).block,'dagnn.Conv')
		ll=l;
		while true
			i=net.layers(ll).inputIndexes-1;
			if isa(net.layers(i).block,'dagnn.Concat') || isa(net.layers(i).block,'dagnn.Conv')
				link(l,1)=i;
				break;
			end
			ll=i;
		end
		link(l,2)=l; % 输出不与 concat 相连的 conv 输出连接为自身

	% 基于假设 max pooling 仅存在于 reduction block 的单独支路上
	% stem 中也存在 max pooling，不过临近输入层不会是 concat
	elseif isa(net.layers(l).block,'dagnn.Pooling') && strcmp(net.layers(l).block.method,'max')
		i=net.layers(l).inputIndexes-1;
		if isa(net.layers(i).block,'dagnn.Concat')
			link(l,1)=i;
        elseif isa(net.layers(i-1).block,'dagnn.Conv')
            link(l,1)=i-1;
		end
		
	elseif isa(net.layers(l).block,'dagnn.Concat')
		for i=net.layers(l).inputIndexes-1
			% 假设 concat 与 conv 的连接之间只有 ReLU, 没有 Pooling，两层足以回溯，且绝不超过三层
			% 这里可以直接使用层数递减回溯，只回溯两层确保不越过 concat.
			for j=i:-1:i-1 
				if isa(net.layers(j).block,'dagnn.Conv') || isa(net.layers(j).block,'dagnn.Pooling')
					link(j,2)=l;
                end
			end
		end
	end
end


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
for epoch=61:70

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

snn=parse_dagnn(net,link,99.99); % 记得改回99.99

opts.dt 			= 0.001;
opts.duration		= 2.500;
opts.report_every	= 0.010; % 此处有改动
opts.threshold		=   1.0;
opts.batch 			=56001:57000;

[performance,stats]=dagnn2snn(snn,imdb,opts);
