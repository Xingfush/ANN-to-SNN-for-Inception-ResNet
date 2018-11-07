function net = incep_res_init(varargin)

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
  % 加上对于 relu 的控制，'relu',true or false
  function Conv(name, ksize, out, varargin)
    copts.stride = [1 1] ;
    copts.pad = (ksize-1)/2 ;
    copts.relu = true ;
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
    if copts.relu
      net.addLayer([name '_relu'] , ...
        dagnn.ReLU(), ...
        [name '_bn'], ...
        name) ;
      stack{end+1} = [name '_relu'] ;
    else
      stack{end+1} = [name '_bn'] ;
    end
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

  function Sum(name)
    inputVars = {};
    for layer = stack(end-2+1:end)
      prev = char(layer) ;
      i = net.getLayerIndex(prev) ;
      inputVars{end+1} = net.layers(i).outputs{1} ;
    end
    stack(end-2+1:end) = [] ;
    net.addLayer(name , ...
      dagnn.Sum(), ...
      inputVars,...
      name) ;
    net.addLayer([name '_relu'] , ...
      dagnn.ReLU(), ...
      name, ...
      [name '_relu']) ;
    stack{end+1} = [name '_relu'] ;
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
      'objective') ; %  损失函数的输出必须是 'objective'，与 cnn_train_dagnn 里的设置相符

    net.addLayer([name '_top1error'], ...
      dagnn.Loss('loss', 'classerror'), ...
      {name, 'label'}, ...
      [name '_top1error']) ;

    net.addLayer([name '_top5error'], ...
      dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
      {name, 'label'}, ...
      [name '_top5error']) ;
  end


% stack 只保留最后一层的 layer name, 不断地删除加入进行更新

% Stem
Conv('conv1',3,32); % 32 32 32
Conv('conv2',3,64); % 32 32 64
pfx=sprintf('stem_1');
dup();
Conv([pfx '_a1'],3,64,'stride',2,'pad',[0 1 0 1]); % 16 16 64
swap(); 
Pool([pfx '_b1'],3,'method','max','stride',2,'pad',[0 1 0 1]); % 16 16 64
Concat(pfx,2); % 16 16 128

% res-inception 17 x 17
for t=1:2
  pfx=sprintf('resnet7_%d',t);
  dup(); dup(); % stack: stem_1, stem_1, stem_1
  Conv([pfx '_a1'],1,24); % stack: stem_1, stem_1, a1
  swap(); % stack: stem_1, a1, stem_1
  Conv([pfx '_b1'],1,16); % stack: stem_1, a1, b1
  Conv([pfx '_b2'],[1 5],24); % stack: stem_1, a1, b2
  Conv([pfx '_b3'],[5 1],24); % stack: stem_1, a1, b3
  Concat([pfx '_cat'],2); % stack: stem_1, concat
  Conv(pfx,1,128,'relu',false); % stack: stem_1, pfx
  Sum([pfx '_sum']);
end

% res-inception 17 x 17 downinto 8 x 8
pfx=sprintf('resnet7_5');
dup();
Conv([pfx '_a1'],1,24);
Conv([pfx '_a2'],3,32,'stride',2,'pad',[0 1 0 1]);
swap(); dup();
Conv([pfx '_b1'],1,24);
Conv([pfx '_b2'],3,48,'stride',2,'pad',[0 1 0 1]);
swap(); dup();
Conv([pfx '_c1'],1,32);
Conv([pfx '_c2'],3,40);
Conv([pfx '_c3'],3,48,'stride',2,'pad',[0 1 0 1]);
swap();
Pool([pfx '_d1'],3,'method','max','stride',2,'pad',[0 1 0 1]);
Concat(pfx,4);

% res-inception 8 x 8
for t=1:1
    pfx=sprintf('resnet8_%d',t);
    dup(); dup();
    Conv([pfx '_a1'],1,24);
    swap();
    Conv([pfx '_b1'],1,24);
    Conv([pfx '_b2'],[1 3],32);
    Conv([pfx '_b3'],[3 1],32);
    Concat([pfx 'cat'],2);
    Conv(pfx,1,256,'relu',false);
    Sum([pfx '_sum']);
end

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

lr=logspace(-1,-3,80);
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 3 ;
net.meta.trainOpts.weightDecay = 0.003 ;

% Init parameters randomly
net.initParams() ;


end