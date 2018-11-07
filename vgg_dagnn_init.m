function net = vgg_dagnn_init(varargin)

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

net.meta.inputSize = [32 32 3 1] ;
net.meta.normalization.imageSize = net.meta.inputSize(1:3) ;

stack = {} ;

  function dup()
    stack{end+1} = stack{end} ;
  end

  function swap()
    stack([end-1 end]) = stack([end end-1]) ;
  end

  % 默认 feature map size 不变，可以通过输入改变 stride, pad
  function Conv(name, ksize, out, varargin)
    copts.stride = [1 1] ;
    copts.pad = (ksize-1)/2 ;
    copts = vl_argparse(copts, varargin) ;
    if isempty(stack)
      inputVar = 'input' ;
      in = 3 ;
    else
      prev = stack{end} ;
      stack(end) = [] ; % 对 cell 单元级别操作，使用 ()
      i = net.getLayerIndex(prev) ;
      inputVar = net.layers(i).outputs{1} ;
      sizes = net.getVarSizes({'input', net.meta.inputSize}) ;
      j = net.getVarIndex(inputVar) ;
      in = sizes{j}(3) ;
    end
    if numel(ksize) == 1, ksize = [ksize ksize] ; end
    net.addLayer(name , ...
      dagnn.Conv('size', [ksize in out], ...
      'stride', copts.stride, ....
      'pad', copts.pad, ...
      'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
      inputVar, ...
      [name '_conv'], ...
      {[name '_f'], [name '_b']}) ;
    net.addLayer([name '_bn'], ...
      dagnn.BatchNorm('numChannels', out), ...
      [name '_conv'], ...
      [name '_bn'], ...
      {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
    net.addLayer([name '_relu'] , ...
      dagnn.ReLU(), ...
      [name '_bn'], ...
      name) ;
    stack{end+1} = [name '_relu'] ;
  end

  % 默认 feature map size 不变，可以通过改变 stride, pad 进行调整
  function Pool(name, ksize, varargin)
    copts.stride = [1 1] ;
    copts.pad = (ksize-1)/2 ;
    copts.method = 'max' ;
    copts = vl_argparse(copts, varargin) ;

    prev = stack{end} ;
    stack(end) = [] ;
    i = net.getLayerIndex(prev) ;
    inputVar = net.layers(i).outputs{1} ;

    if numel(ksize) == 1, ksize = [ksize ksize] ; end
    net.addLayer(name , ...
      dagnn.Pooling('poolSize', ksize, ...
      'method', copts.method, ...
      'stride', copts.stride, ....
      'pad', copts.pad), ...
      inputVar, ...
      [name '_pool']) ;
    stack{end+1} = name ;
  end

  function Concat(name, num)
    inputVars = {} ;
    for layer = stack(end-num+1:end)
      prev = char(layer) ;
      i = net.getLayerIndex(prev) ;
      inputVars{end+1} = net.layers(i).outputs{1} ;
    end
    stack(end-num+1:end) = [] ;
    net.addLayer(name , ...
      dagnn.Concat(), ...
      inputVars, ...
      name) ;
    stack{end+1} = name ;
  end

  function Pred(name, out, varargin)
    prev = stack{end} ;
    stack(end) = [] ;
    i = net.getLayerIndex(prev) ;
    inputVar = net.layers(i).outputs{1} ;
    sizes = net.getVarSizes({'input', net.meta.inputSize}) ;
    j = net.getVarIndex(inputVar) ;
    in = sizes{j}(3) ;

    net.addLayer([name '_dropout'] , ...
      dagnn.DropOut('rate', 0.3), ...
      inputVar, ...
      [name '_dropout']) ;

    net.addLayer(name, ...
      dagnn.Conv('size', [1 1 in out]), ...
      [name '_dropout'], ...
      name, ...
      {[name '_f'], [name '_b']}) ;

    net.addLayer([name '_loss'], ...
      dagnn.Loss('loss', 'softmaxlog'), ...
      {name, 'label'}, ...
      'objective') ;

    net.addLayer([name '_top1error'], ...
      dagnn.Loss('loss', 'classerror'), ...
      {name, 'label'}, ...
      [name '_top1error']) ;

    net.addLayer([name '_top5error'], ...
      dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
      {name, 'label'}, ...
      [name '_top5error']) ;
  end

Conv('conv1_1',3,64);
Conv('conv1_2',3,64);
% Conv('conv1_3',3,64); % third version
Pool('pool_1',3,'method','max','stride',2,'pad',[0 1 0 1]);

Conv('conv2_1',3,128);
Conv('conv2_2',3,128);
% Conv('conv2_3',3,128); % second version
% Conv('conv2_4',3,128); % third version
Pool('pool_2',3,'method','max','stride',2,'pad',[0 1 0 1]);

Conv('conv3_1',3,256);
Conv('conv3_2',3,256); 
% Conv('conv3_3',3,256); % second version
% Conv('conv3_4',3,256); % third version
Pool('pool_3',3,'method','max','stride',2,'pad',[0 1 0 1]);

Conv('conv4_1',4,512,'stride',1,'pad',[0 0 0 0]);

Pred('prediction',10);


% Meta parameters
net.meta.normalization.fullImageSize = 32 ;
net.meta.normalization.averageImage = [] ;
net.meta.augmentation.rgbSqrtCovariance = zeros(3,'single') ;
net.meta.augmentation.jitter = true ;
net.meta.augmentation.jitterLight = 0.1 ;
net.meta.augmentation.jitterBrightness = 0.4 ;
net.meta.augmentation.jitterSaturation = 0.4 ;
net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

lr=logspace(-1,-3,50); % 60 -> 100
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 512 ; % 256 -> 128
net.meta.trainOpts.numSubBatches = 3 ;
net.meta.trainOpts.weightDecay = 0.002 ;

% Init parameters randomly
net.initParams() ;


end