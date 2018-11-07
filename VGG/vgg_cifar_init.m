function net =vgg_cifar_init(varargin)

opts.scale=1;
opts.initBias=0;
opts.weightDecay=1;
opts.weightInitMethod='gaussian';
opts.networkType='simplenn';
opts.cudnnWorkspaceLimit=1024*1024*1024;
opts.classNames={};
opts.classDescriptions={};
opts.averageImage=zeros(3,1);
opts.colorDeviation=zeros(3);

opts.model='vgg16';  
opts.batchNormalization=false;

opts=vl_argparse(opts,varargin);

net.meta.normalization.imageSize=[32,32,3];
bs=256;

switch opts.model
	case 'vgg16'
		net=vgg16(net,opts);
	case 'vgg_low'
		net=vgg_low(net,opts);
    case 'vgg_v'
        net=vgg_v(net,opts);
	otherwise
		error('Unknown model ''%s''', opts.model);
end
net.layers{end+1}=struct('type','softmaxloss','name','loss');

net.meta.inputSize=[net.meta.normalization.imageSize, 32];
net.meta.normalization.cropSize=net.meta.normalization.imageSize(1)/256;

net.meta.normalization.averageImage=opts.averageImage;
net.meta.classes.name=opts.classNames;
net.meta.classes.description=opts.classDescriptions;

net.meta.augmentation.jitterLocation=true;
net.meta.augmentation.jitterFlip=true;
net.meta.augmentation.jitterBrightness=double(0.1*opts.colorDeviation);
net.meta.augmentation.jitterAspect=[2/3,3/2];

if ~opts.batchNormalization
	lr=logspace(-2, -4, 60);
else
	lr=logspace(-2, -4, 20);
end

net.meta.trainOpts.learningRate=lr;
net.meta.trainOpts.numEpochs=numel(lr);
net.meta.trainOpts.batchSize=bs;
net.meta.trainOpts.weightDecay=0.0005; 

net=vl_simplenn_tidy(net);


% --------------------------------------------------------------------
function net = add_block_b(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 0], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;


% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end


% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;  
end


% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization 
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end


% --------------------------------------------------------------------
function net =vgg16(net,opts)
% --------------------------------------------------------------------

net.layers={};
opts.batchNormalization=true;

net=add_block(net,opts,'1',3,3,3,64,1,1); % 32 32 64 
net=add_block(net,opts,'2',3,3,64,64,1,1); % 32 32 64

net.layers{end+1}=struct('type','pool','name','pool1',...
						'method','max',...
						'pool',[3,3],...
						'stride',2,...
						'pad',[0,1,0,1]); % 16 16 64

net=add_block(net,opts,'3',3,3,64,128,1,1); % 16 16 128
net=add_block(net,opts,'4',3,3,128,128,1,1); % 16 16 128

opts.batchNormalization=false;

net.layers{end+1}=struct('type','pool','name','pool2',...
						'method','max',...
						'pool',[3,3],...
						'stride',2,...
						'pad',[0,1,0,1]); % 8 8 128

net=add_block(net,opts,'5',3,3,128,256,1,1); % 8 8 256
net=add_block(net,opts,'6',3,3,256,256,1,1); % 8 8 256
net=add_block(net,opts,'7',3,3,256,256,1,1); % 8 8 256

net.layers{end+1}=struct('type','pool','name','pool3',...
						'method','max',...
						'pool',[3,3],...
						'stride',2,...
						'pad',[0,1,0,1]); % 4 4 256

net=add_block(net,opts,'8',3,3,256,512,1,1); % 4 4 512
net=add_block(net,opts,'9',3,3,512,512,1,1); % 4 4 512
net=add_block(net,opts,'10',3,3,512,512,1,1); % 4 4 512

net=add_block(net,opts,'11',4,4,512,1024,1,0); % 1 1 1024
net=add_dropout(net,opts,'11');

net=add_block(net,opts,'12',1,1,1024,1024,1,0);
net=add_dropout(net,opts,'12');

net=add_block(net,opts,'13',1,1,1024,10,1,0); % 1 1 10

net.layers(end) = [];
if opts.batchNormalization, net.layers(end) = []; end


% --------------------------------------------------------------------
function net = vgg_v(net,opts)
% --------------------------------------------------------------------

net.layers={};

% block contains the batch normalization layer
opts.batchNormalization=true;

net=add_block(net,opts,'1',5,5,3,32,1,2); % 32 32 32 

net=add_block(net,opts,'2',3,3,32,32,1,1); % 32 32 32
net.layers{end+1}=struct('type','pool','name','pool1',...
            'method','max',...
            'pool',[3,3],...
            'stride',2,...
            'pad',[0,1,0,1]); % 16 16 32
    
% block don't contain the batch normalization layer
opts.batchNormalization=false;

net=add_block(net,opts,'3',3,3,32,64,1,1); % 16 16 64
net.layers{end+1}=struct('type','pool','name','pool2',...
            'method','max',...
            'pool',[3,3],...
            'stride',2,...
            'pad',[0,1,0,1]); % 8 8 64

net=add_block(net,opts,'4',3,3,64,64,1,1); % 8 8 64
net.layers{end+1}=struct('type','pool','name','pool3',...
            'method','max',...
            'pool',[2,2],...
            'stride',2,...
            'pad',[0,0,0,0]); % 4 4 64

net=add_block_b(net,opts,'5',4,4,64,512,1,0); % 1 1 512
net=add_dropout(net,opts,'5');

net=add_block_b(net,opts,'6',1,1,512,10,1,0); % 1 1 10

net.layers(end) = [] ;

