function [net,info] = train_cifar(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.modelType = 'inception-v4-5' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir=fullfile(vl_rootnn,'data',['cifar-' opts.modelType]);
[opts,varargin]=vl_argparse(opts,varargin);

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(vl_rootnn,'data','cifar-lenet','imdb_whiten.mat'); %在这里设置dataset路径

opts.train=struct();
opts=vl_argparse(opts,varargin);

if ~isfield(opts.train,'gpus'), opts.train.gpus=[1]; end;

if exist(opts.imdbPath,'file')
  imdb=load(opts.imdbPath);
end

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
if isempty(opts.network)

  switch opts.modelType
    case 'inception-v4-5'
      net = inception_cifar_init() ;
      opts.networkType = 'dagnn' ;
      
    case 'res-inception-v2-2'
      net = res_cifar_init() ;
      opts.networkType = 'dagnn' ;
      
    case 'resnet'
      net = resnet_init() ;
      opts.networkType = 'dagnn' ;
  end
else
  net = opts.network ;
  opts.network = [] ;
end

net.meta.classes.name=imdb.meta.classes(:)';

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
switch opts.networkType
  case 'simplenn'
    trainFn=@cnn_train;
  case 'dagnn'
    trainFn=@cnn_train_dag;
end

[net,info]=trainFn(net,imdb,getBatch(opts),...
					'expDir',opts.expDir,...
					net.meta.trainOpts,...
					opts.train,...
					'val',find(imdb.images.set==3));


% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end


% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ; 
labels = imdb.images.labels(1,batch) ; 
if rand > 0.5, images=fliplr(images) ; end % images 

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ; 
labels = imdb.images.labels(1,batch) ; 
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ; 