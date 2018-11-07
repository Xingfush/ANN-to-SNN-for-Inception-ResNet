function net = resnet_init(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ; % 首先建立一个空的 net 结构

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;

function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args.pad = (ksize - 1) / 2 ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end % downsample 只用来控制 stride, pad 最好还是自己调整，在 cifar10
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end % [str1 str2] 就是连接字符串，pars 就是每层权值的名字
  net.addLayer([name  '_conv'], ... % 后面三个分别是 输入，输出，权值的名字，just name 字符串
               dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                          'stride', stride, ....
                          'pad', args.pad, ...
                          'hasBias', args.bias, ...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
               lastAdded.var, ...
               [name '_conv'], ...
               pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_conv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ; % 变量的名字（字符串）指定了前后的连接
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ;
  end
end



% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1',3,16,...
     'relu',true,...
     'bias',true);


% -------------------------------------------------------------------------
% Add intermediate sections % 构建残差模块，每个 section 有不同数目的 blocks
% -------------------------------------------------------------------------
for s=2:4

  switch s
    case 2, sectionLen=3;
    case 3, sectionLen=3;
    case 4, sectionLen=3;
  end

  for l=1:sectionLen
    depth=2^(s+2);
    sectionInput=lastAdded;
    name=sprintf('conv%d_%d',s,l);

    % Optional adapter layer
    if l==1 && s>=3 
      Conv([name '_adapt_conv'],1,depth,'downsample',true,'relu',false); % bias = 0, pad = 0
    end
    sumInput=lastAdded;

    lastAdded=sectionInput;
    if l==1 && s>=3
      Conv([name 'a'],3,depth,'downsample',true,'pad',[0 1 0 1]);
    else
      Conv([name 'a'],3,depth);
    end

    Conv([name 'b'],3,depth,'relu',false);

    net.addLayer([name '_sum'],...
                 dagnn.Sum(),...
                 {sumInput.var, lastAdded.var},...
                 [name '_sum']);
    net.addLayer([name '_relu'],...
                  dagnn.ReLU(),...
                  [name '_sum'],...
                  name);
    lastAdded.var=name;
  end
end

net.addLayer('prediction_avg' , ...
             dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
             lastAdded.var, ...
             'prediction_avg') ;

net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 64 10]), ...
             'prediction_avg', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;

net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;

    
% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [32 32 3] ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
net.meta.augmentation.jitterScale  = [0.4, 1.1] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

lr = logspace(-1, -3, 30) ;
% lr = [0.1 * ones(1,30), 0.01*ones(1,30), 0.001*ones(1,30)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 4 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
net.initParams() ; 

end   