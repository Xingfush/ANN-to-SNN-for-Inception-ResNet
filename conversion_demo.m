% 设置数据集、模型路径
modelPath='C:\Program Files\MATLAB\R2016b\matconvnet-1.0-beta25\matconvnet-1.0-beta25\data\cifar-lenet\net-epoch-45.mat';
imdbPath= 'C:\Program Files\MATLAB\R2016b\matconvnet-1.0-beta25\matconvnet-1.0-beta25\data\cifar-lenet\imdb.mat';

% 加载数据集
imdb=load(imdbPath);
trainSet=find(imdb.images.set==1);
testSet=find(imdb.images.set==3);
num_examples=numel(trainSet);

% 加载 CNN 训练模型，具体前面加不加 net 看具体模型
load(modelPath); 
net=vl_simplenn_tidy(net);

% 记录卷积层前向传输的激活值
for l=1:numel(net.layers)
	if strcmp(net.layers{l}.type,'conv')
		net.layers{l}.precious=1;
	end
end

% 训练集前向传输获得每层的输出值 a
batchSize=200;
epochs=ceil(num_examples/batchSize);

output=cell(1,numel(net.layers));
res=[];

for epoch=1:10 % 为了节省内存，提高运算速度，减小 epochs

	batchStart=(epoch-1)*batchSize+1;
	batchEnd=min(epoch*batchSize,num_examples);
	batch=trainSet(batchStart:batchEnd);

	im=imdb.images.data(:,:,:,batch);
    labels=imdb.images.labels(1,batch);
    net.layers{end}.class=labels;
	res=vl_simplenn(net,im,[],res,'mode','test','cudnn',true,'conservememory',true);
    for l=1:numel(net.layers)
       output{l}=cat(4,output{l},res(l+1).x);
    end
    fprintf('%d epoch forward finised.\n',epoch);
end

for l=1:numel(net.layers)
	net.layers{l}.a=output{l};
end
clear output; 
clear res;

% 网络 weights normalization: [99.9,99.999]
[net,factor_log]=normalize_data(net,99.99);

for l=1:numel(net.layers)
	net.layers{l}.a=[];
end

opts.threshold	 	=	1.0;
opts.duration		= 2.000;
opts.dt 			= 0.001;
opts.report_every	= 0.010;
opts.batch 			= 1:1000;

% 设置 SNN 数据集
cnn2snn(net,imdb,opts);

