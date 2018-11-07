function net = inception_init(varargin)

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
      dagnn.DropOut('rate', 0.2), ...
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

% Stem
Conv('conv1',3,32); % 32 32 32
Conv('conv2',3,64); % 32 32 64
pfx=sprintf('stem_1');
dup();
Conv([pfx '_a1'],3,64,'stride',2,'pad',[0 1 0 1]); % 16 16 64
swap(); 
Pool([pfx '_b1'],3,'method','max','stride',2,'pad',[0 1 0 1]); % 16 16 64
Concat(pfx,2); % 16 16 128

% Inception fig. 5 x 2 
for t=1:4
  pfx=sprintf('inception5_%d',t);
  dup();
  Conv([pfx '_a1'],1,48);
  swap(); dup();
  Conv([pfx '_b1'],1,24);
  Conv([pfx '_b2'],[1 5],32);
  Conv([pfx '_b3'],[5 1],32);
  swap(); dup();
  Conv([pfx '_c1'],1,24);
  Conv([pfx '_c2'],[1 5],28);
  Conv([pfx '_c3'],[5 1],28);
  Conv([pfx '_c4'],[1 5],32);
  Conv([pfx '_c5'],[5 1],32);
  swap(); 
  Pool([pfx '_d1'],3,'method','avg');
  Conv([pfx '_d2'],1,16);
  Concat(pfx,4); % 16 16 128
end

% Inception fig. 5 down
pfx=sprintf('inception5_5');
dup();
Conv([pfx '_a1'],1,64);
Conv([pfx '_a2'],3,64,'stride',2,'pad',[0 1 0 1]);
swap(); dup();
Conv([pfx '_b1'],1,48);
Conv([pfx '_b2'],[1 5],48);
Conv([pfx '_b3'],[5 1],78);
Conv([pfx '_b4'],3,78,'stride',2,'pad',[0 1 0 1]);
swap();
Pool([pfx '_c1'],3,'method','max','stride',2,'pad',[0 1 0 1]);
Concat(pfx,3); % 8 8 270



% Inception fig. 6 x 1
pfx=sprintf('inception6_1');
dup(); % stack{'concat','concat'}
Conv([pfx '_a1'],1,45); % stack{'concat','a1'}
swap(); dup(); % stack{'a1','concat','concat'}
Conv([pfx '_b1'],1,90); % stack{'a1','concat','b1'}
dup(); % stack{'a1','concat','b1','b1'}
Conv([pfx '_b1_1'],[1 3],45); % stack{'a1','concat','b1','b1_1'}
swap(); % stack{'a1','concat','b1_1','b1'}
Conv([pfx '_b1_2'],[3 1],45); % stack{'a1','concat','b1_1','b1_2'}
stack{end+1}=stack{end-2}; % 待验证
stack(end-3)=[]; % stack{'a1','b1_1','b1_2','concat','concat'}
dup();
Conv([pfx '_c1'],1,90);
Conv([pfx '_c2'],[1 3],90);
Conv([pfx '_c3'],[3 1],90); % stack{'a1','b1_1','b1_2','concat','c3'}
dup();
Conv([pfx '_c3_1'],[1 3],45); % stack{'a1','b1_1','b1_2','concat','c3','c3_1'}
swap();
Conv([pfx '_c3_2'],[3 1],45); % stack{'a1','b1_1','b1_2','concat','c3_1','c3_2'}
stack{end+1}=stack{end-2};
stack(end-3)=[]; % stack{'a1','b1_1','b1_2','c3_1','c3_2','concat'}
Pool([pfx '_d1'],3,'method','avg');
Conv([pfx '_d2'],1,45);
Concat(pfx,6); % 8 8 270


% Prediction
% Average pooling and loss
Pool('Pool2',8,'method','avg','pad',0); % 1 1 270
Pred('prediction',10); % 1 1 10

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

lr=logspace(-1,-3,80); % 60 -> 100
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 256 ; % 256 -> 128
net.meta.trainOpts.numSubBatches = 3 ;
net.meta.trainOpts.weightDecay = 0.0025 ;

% Init parameters randomly
net.initParams() ;


end