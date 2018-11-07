function [performance,stats] = dagnn2snn(snn,imdb,opts)

% Parameters
% snn		:	SNN struct.
% imdb		:	images.data, images.labels, meta.
% opts		:	dt, duration, report_every, threshold, batch

% Compute the sizes of layers activations (cell array).

data=imdb.images.data(:,:,:,opts.batch);
ans_idx=imdb.images.labels(1,opts.batch);
num_examples=size(data,4);

% -------------------------------------------------------------------------
%                                       Add the SNN elements into snn model
% -------------------------------------------------------------------------
for l=1:numel(snn)
	if strcmp(snn{l}.type,'conv')
		snn{l}.mem=zeros([snn{l}.shape(1:3) num_examples]);
	elseif strcmp(snn{l}.type,'relu')
		snn{l}.sum_spikes=zeros([snn{l}.shape(1:3) num_examples]);
	end
end

% 统计每层神经元的发放状况：max firing rate decrease along layers
for l=1:numel(snn)
	if strcmp(snn{l}.type,'relu')
		stats{l}.max_rate=[];
%         stats{l}.rate={}; % 此处为特征可视化设计
    end
end


% top1err
performance=[];

% -------------------------------------------------------------------------
%                                                    SNN forward simulation
% -------------------------------------------------------------------------
for t=0:opts.dt:opts.duration

	% 第一层特殊处理，将图像作为恒定电流输入
	z=vl_nnconv(data,snn{1}.weights{1},snn{1}.weights{2},...
					'stride',snn{1}.stride,'pad',snn{1}.pad);             
	snn{1}.mem=snn{1}.mem+z;

	for l=2:numel(snn)
		if strcmp(snn{l}.type,'conv')
			z=vl_nnconv(snn{snn{l}.prev}.spikes,snn{l}.weights{1},snn{l}.weights{2},...
							'stride',snn{l}.stride,'pad',snn{l}.pad);
			snn{l}.mem=snn{l}.mem+z;

		elseif strcmp(snn{l}.type,'relu')
			% relu 的上一层一定是 conv
			snn{l}.spikes=single(snn{l-1}.mem>=opts.threshold);
			snn{l-1}.mem(snn{l-1}.mem>=opts.threshold)=snn{l-1}.mem(snn{l-1}.mem>=opts.threshold)-1;
			snn{l}.sum_spikes=snn{l}.sum_spikes+snn{l}.spikes;

		% concat, max pooling 层同时具有 spikes, sum_spikes 元素
		% max pooling 的 scale 同时对 spikea, sum_spikes 作用 
		elseif strcmp(snn{l}.type,'pool')
			if strcmp(snn{l}.method,'max')
				snn{l}.spikes=max_gate(snn{snn{l}.prev}.spikes,snn{snn{l}.prev}.sum_spikes,...
											snn{l}.filter,snn{l}.pad,snn{l}.stride);
				snn{l}.sum_spikes=vl_nnpool(snn{snn{l}.prev}.sum_spikes,snn{l}.filter,'method','max',...
											'stride',snn{l}.stride,'pad',snn{l}.pad);
			elseif strcmp(snn{l}.method,'avg')
				snn{l}.spikes=vl_nnpool(snn{snn{l}.prev}.spikes,snn{l}.filter,'method','avg',...
										'stride',snn{l}.stride,'pad',snn{l}.pad);
				snn{l}.sum_spikes=vl_nnpool(snn{snn{l}.prev}.sum_spikes,snn{l}.filter,'method','avg',...
										'stride',snn{l}.stride,'pad',snn{l}.pad);
			end
			snn{l}.spikes=snn{l}.spikes*snn{l}.scale;
			snn{l}.sum_spikes=snn{l}.sum_spikes*snn{l}.scale;

		elseif strcmp(snn{l}.type,'concat')
			input_spikes={};
			input_sum={};
			% Based on fact that concat must follow l-1 relu.
			for i=snn{l}.prev
				input_spikes{end+1}=snn{i}.spikes;
				input_sum{end+1}=snn{i}.sum_spikes;
			end
			snn{l}.spikes=vl_nnconcat(input_spikes,3);
			snn{l}.sum_spikes=vl_nnconcat(input_sum,3);

		elseif strcmp(snn{l}.type,'softmax')
			snn{l}.mem=vl_nnsoftmax(snn{l-1}.mem);
		end
	end


	% plotting accuracy and stats
	if(mod(round(t/opts.dt),round(opts.report_every/opts.dt))==...
				0 && (t/opts.dt>0))

        [~,guess_idx]=max(squeeze(snn{end}.mem));

        % 观察神经元发放变化
%       fprintf('the prediction result is:\n');
%       disp(guess_idx(1:10));
%       fprintf('the true label is:\n');
%       disp(ans_idx(1:10));
		acc=sum(guess_idx==ans_idx)/num_examples*100;
		fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n',t,acc);
		performance(end+1)=acc;
        
        % 此处发放率的计算和时间无关
        for l=1:numel(snn)
        	if strcmp(snn{l}.type,'relu')
    			stats{l}.max_rate(end+1)=max(max(max(max(snn{l}.sum_spikes))))/(t*1000);
            end
        end    
        
%         for l=1:4
% 			if strcmp(snn{l}.type,'relu')
%                 stats{l}.rate{end+1}=snn{l}.sum_spikes/(t*1000);
%             end
%         end
        
        % Start plotting
		switchFigure(1) ; clf ;
        
		% accuracy
		subplot(1,2,1); 
	    plot(performance,'o-');
	    xlabel('epoch');
	    title('spiking accuracy');
	    grid on;

	    values=zeros(0,round(t/opts.report_every));
	    leg={};
	    % max_rate
	    for l=1:numel(snn)
	    	if strcmp(snn{l}.type,'relu')
                values(end+1,:)=stats{l}.max_rate;
                leg{end+1}=sprintf('layer %d',l);
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

stride=stride(1);

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