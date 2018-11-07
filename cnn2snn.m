function net = cnn2snn(net,imdb,opts)

% 图片输入尺寸
data=imdb.images.data(:,:,:,opts.batch);
ans_idx=imdb.images.labels(1,opts.batch);

[input_h,input_w,~,num_examples]=size(data);

% 计算每层网络的形状，[height, width, channel]
% pool 层的 pad 是数组[top down left right]，conv 的 pad 是一个 value
shape=zeros(numel(net.layers),3);

if strcmp(net.layers{1}.type,'conv')
	h=(input_h-size(net.layers{1}.weights{1},1)+2*net.layers{1}.pad)/net.layers{1}.stride+1;
	w=(input_w-size(net.layers{1}.weights{1},2)+2*net.layers{1}.pad)/net.layers{1}.stride+1;
	shape(1,:)=[h,w,size(net.layers{1}.weights{1},4)];
end

for l=2:numel(net.layers)
	if strcmp(net.layers{l}.type,'conv')
		h=(shape(l-1,1)-size(net.layers{l}.weights{1},1)+2*net.layers{l}.pad)/net.layers{l}.stride+1;
		w=(shape(l-1,2)-size(net.layers{l}.weights{1},1)+2*net.layers{l}.pad)/net.layers{l}.stride+1;
		shape(l,:)=[h,w,size(net.layers{l}.weights{1},4)];
	elseif strcmp(net.layers{l}.type,'relu')
		shape(l,:)=shape(l-1,:);
	elseif strcmp(net.layers{l}.type,'pool')
		h=(shape(l-1,1)-net.layers{l}.pool(1)+net.layers{l}.pad(1)+net.layers{l}.pad(2))/...
					net.layers{l}.stride+1;
		w=(shape(l-1,2)-net.layers{l}.pool(2)+net.layers{l}.pad(3)+net.layers{l}.pad(4))/...
					net.layers{l}.stride+1;
		shape(l,:)=[h,w,shape(l-1,3)];
	elseif strcmp(net.layers{l}.type,'softmax') || strcmp(net.layers{l}.type,'softmaxloss')
		shape(l,:)=shape(l-1,:);
	end
end

% 卷积层的 mem，sum_spikes 必须被初始化，spikes 不需要初始化，暂时采用所有属性初始化
% 不设 batch size，整体 samples 一次运行
for l=1:numel(net.layers)
	correctly_sized_zeros=zeros([shape(l,:) num_examples]);
	net.layers{l}.mem=correctly_sized_zeros;
	net.layers{l}.sum_spikes=correctly_sized_zeros;
end

% 统计每层神经元的发放状况：max firing rate decrease along layers
for l=1:numel(net.layers)
	if strcmp(net.layers{l}.type,'relu') || strcmp(net.layers{l}.type,'softmax')
		net.stats{l}.max_rate=[];
    end
end

% top1err
net.performance=[];

% 开始 simulation
for t=0:opts.dt:opts.duration

	% 首先采用泊松神经元序列进行测试
% 	rescale_fac=1/(opts.dt*opts.max_rate);
% 	spike_snapshot=rand(size(imdb.images.data))*rescale_fac;
% 	inp_image=single(spike_snapshot<=imdb.images.data); % 其实就是 spikes，作为输入层

	% analog current input，作为第一层卷积层的输入
	z=vl_nnconv(data,net.layers{1}.weights{1},net.layers{1}.weights{2},...
						'stride',net.layers{1}.stride,'pad',net.layers{1}.pad);
                        
	net.layers{1}.mem=net.layers{1}.mem+z;
    
	for l=2:numel(net.layers)-1
		if strcmp(net.layers{l}.type,'conv')
			z=vl_nnconv(net.layers{l-1}.spikes, net.layers{l}.weights{1}, net.layers{l}.weights{2},...
							'stride',net.layers{l}.stride, 'pad',net.layers{l}.pad);
			net.layers{l}.mem=net.layers{l}.mem+z;
		elseif strcmp(net.layers{l}.type,'relu')
			net.layers{l}.spikes=single(net.layers{l-1}.mem>=opts.threshold);
			net.layers{l-1}.mem(net.layers{l-1}.mem>=opts.threshold)=net.layers{l-1}.mem(net.layers{l-1}.mem>=opts.threshold)-1;
			net.layers{l}.sum_spikes=net.layers{l}.sum_spikes+net.layers{l}.spikes;
		elseif strcmp(net.layers{l}.type,'pool') 
			if  strcmp(net.layers{l}.method,'max')
				net.layers{l}.spikes=max_gate(net.layers{l-1}.spikes,net.layers{l-1}.sum_spikes,...
								net.layers{l}.pool,net.layers{l}.pad,net.layers{l}.stride);
            elseif strcmp(net.layers{l}.method,'avg')
				net.layers{l}.spikes=vl_nnpool(net.layers{l-1}.spikes,net.layers{l}.pool,'method','avg',...
								'stride',net.layers{l}.stride,'pad',net.layers{l}.pad);
            end
        end
    end
    
    % 将 softmax 直接简化为 relu 计算 spikes
% 	net.layers{end}.spikes=single(net.layers{end-1}.mem>=opts.threshold);
% 	net.layers{end-1}.mem(net.layers{end-1}.mem>=opts.threshold)=net.layers{end-1}.mem(net.layers{end-1}.mem>=opts.threshold)-1;
% 	net.layers{end}.sum_spikes=net.layers{end}.sum_spikes+net.layers{end}.spikes;
    
    % 对最后一层电动势直接取 softmax 进行分类
    net.layers{end}.mem=vl_nnsoftmax(net.layers{end-1}.mem);
   
    % plotting accuracy and stats
	if(mod(round(t/opts.dt),round(opts.report_every/opts.dt))==...
				0 && (t/opts.dt>0))

        [~,guess_idx]=max(squeeze(net.layers{end}.mem));

        % 观察神经元发放变化
%       fprintf('the prediction result is:\n');
%       disp(guess_idx(1:10));
%       fprintf('the true label is:\n');
%       disp(ans_idx(1:10));
		acc=sum(guess_idx==ans_idx)/num_examples*100;
		fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n',t,acc);
		net.performance(end+1)=acc;
        
        % 此处发放率的计算和时间无关
    	for l=1:numel(net.layers)
			if strcmp(net.layers{l}.type,'relu') || strcmp(net.layers{l}.type,'softmax')
    			net.stats{l}.max_rate(end+1)=max(max(max(max(net.layers{l}.sum_spikes))))/(t*1000);
            end
        end
        
        % Start plotting
		switchFigure(1) ; clf ;
        
		% accuracy
		subplot(1,2,1); 
	    plot(net.performance,'o-');
	    xlabel('epoch');
	    title('spiking accuracy');
	    grid on;

	    values=zeros(0,round(t/opts.report_every));
	    leg={};
	    % max_rate
	    for i=1:numel(net.layers)
	    	if strcmp(net.layers{i}.type,'relu') || strcmp(net.layers{i}.type,'softmax')
                values(end+1,:)=net.stats{i}.max_rate;
                leg{end+1}=sprintf('layer %d',i);
            end
        end
	    subplot(1,2,2);
	    plot(values','o-');
	    xlabel('epoch');	
	    title('max firing rate');
	    legend(leg{:});
	    grid on;
    end
    drawnow;
end

end

			
% -------------------------------------------------------------------------
function Y=max_gate(spikes,sum_spikes,filter,pad,stride)
% -------------------------------------------------------------------------
% max_gate 使用 sum_spikes 来衡量神经元的发放活跃程度，对某个时间的 spike 实现 max_pooling。
% Parameter format
% 	filter	: [3 3] 
%	pad		: [0 1 0 1]
%	stride	: 2

[h,w,c,n] = size(spikes);
h_o=(h+pad(1)+pad(2)-filter(1))/stride+1;
w_o=(w+pad(3)+pad(4)-filter(2))/stride+1;
Y=zeros(h_o,w_o,c,n);

spikes_t=zeros(h+pad(1)+pad(2),w+pad(3)+pad(4),c,n);
sum_spikes_t=zeros(h+pad(1)+pad(2),w+pad(3)+pad(4),c,n);

spikes_t(pad(1)+1:pad(1)+h,pad(3)+1:pad(3)+w,:,:)=spikes;
sum_spikes_t(pad(1)+1:pad(1)+h,pad(3)+1:pad(3)+w,:,:)=sum_spikes;

for j=1:h_o
    for i=1:w_o
		line1=reshape(sum_spikes_t(1+(j-1)*stride:(j-1)*stride+filter(1),...
                        1+(i-1)*stride:(i-1)*stride+filter(2),:,:),[],c*n);
        line2=reshape(spikes_t(1+(j-1)*stride:(j-1)*stride+filter(1),...
                        1+(i-1)*stride:(i-1)*stride+filter(2),:,:),[],c*n);
        [~,I]=max(line1);
        temp=zeros(1,c*n);
        for k=1:c*n
        	temp(k)=line2(I(k),k);
        end
        temp=reshape(temp,1,1,c,n);
        Y(j,i,:,:)=temp;
   end
end

Y=single(Y); % 全部使用 single 格式运算

end


% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------

if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

end
