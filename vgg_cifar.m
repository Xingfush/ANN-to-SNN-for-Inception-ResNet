function [net,info] = vgg_cifar(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.modelType = 'vgg16' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir=fullfile(vl_rootnn,'data',['cifar-' opts.modelType]);
[opts,varargin]=vl_argparse(opts,varargin);

opts.imdbPath = fullfile(vl_rootnn,'data','cifar-lenet','imdb.mat');

opts.train=struct();
opts=vl_argparse(opts,varargin);

if ~isfield(opts.train,'gpus'), opts.train.gpus=[1]; end;

net= vgg_cifar_init();

if exist(opts.imdbPath,'file')
	imdb=load(opts.imdbPath);
end

net.meta.classes.name=imdb.meta.classes(:)';

[net,info]=cnn_train(net,imdb,getBatch(opts),...
					'expDir',opts.expDir,...
					net.meta.trainOpts,...
					opts.train,...
					'val',find(imdb.images.set==3));

net=cnn_imagenet_deploy(net);
modelPath=fullfile(opts.expDir,'net-deployed.mat');

save(modelPath, '-struct', 'net') ;


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