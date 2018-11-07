function snn = parse_dagnn(net,link,percentile)

% Part 1:  Normalize the SNN weights.
% Part 2:  Parse the DagNN model.

% -------------------------------------------------------------------------
%                                     Compute the maximum of layers in link
% -------------------------------------------------------------------------
max_activations=zeros(1,numel(net.layers));

for l=nonzeros(unique(link))' % 把 link 中的 0 去掉
	if percentile==100
		max_activations(l)=max(max(max(max(net.vars(net.layers(l).outputIndexes).value))));
	else
		temp=reshape(max(max(max(max(net.vars(net.layers(l).outputIndexes).value)))),1,[]);
		max_activations(l)=prctile(temp,percentile);
	end
	if max_activations(l)~=0
		fprintf('The max value of activation in layer %d is:%.4f\n',l, max_activations(l));
	end
end


% -------------------------------------------------------------------------
%                                                 Normalize the dagNN model
% -------------------------------------------------------------------------
% 第一层特殊处理
scale_factor=max_activations(link(1,2));

net.params(net.layers(1).paramIndexes(1)).value=net.params(net.layers(1).paramIndexes(1)).value/scale_factor;
net.params(net.layers(1).paramIndexes(2)).value=net.params(net.layers(1).paramIndexes(2)).value/scale_factor;

for l=2:numel(net.layers)
	if isa(net.layers(l).block,'dagnn.Conv')

		scale_factor=max_activations(link(l,2));
		previous_factor=max_activations(link(l,1));

		current_factor=scale_factor/previous_factor;
		
		net.params(net.layers(l).paramIndexes(1)).value=net.params(net.layers(l).paramIndexes(1)).value/current_factor;
		net.params(net.layers(l).paramIndexes(2)).value=net.params(net.layers(l).paramIndexes(2)).value/scale_factor;
	end
end

% index=[52 66 88];
% factor=1.1;
% for l=index
% 	if isa(net.layers(l).block,'dagnn.Concat')
% 		inputs=net.layers(l).inputIndexes-1;
% 		for i=inputs
% 			ll=i;
% 			while true
% 				j=net.layers(ll).inputIndexes-1;
% 				if isa(net.layers(j).block,'dagnn.Conv')
% 					net.params(net.layers(j).paramIndexes(1)).value=...
% 						net.params(net.layers(j).paramIndexes(1)).value*factor;
% 					net.params(net.layers(j).paramIndexes(2)).value=...
% 						net.params(net.layers(j).paramIndexes(2)).value*factor;	
% 					break;
%                 elseif isa(net.layers(i).block,'dagnn.Pooling') && strcmp(net.layers(i).block.method,'max')
%                     break;
%                 end
% 				ll=j;
% 			end
%         end
% 	end
% end

index=[11 13 16 18];
factor=1.2;
for l=index
    if isa(net.layers(l).block,'dagnn.Conv')
        net.params(net.layers(l).paramIndexes(1)).value=...
            net.params(net.layers(l).paramIndexes(1)).value*factor;
        net.params(net.layers(l).paramIndexes(2)).value=...
            net.params(net.layers(l).paramIndexes(2)).value*factor;
    end
end

% -------------------------------------------------------------------------
%                                          Parse Dagnn model into snn model
% -------------------------------------------------------------------------
% SNN model should contains: pre-connections(层连接关系), weights, mem, sum_spikes.

% SNN is a cell array.
snn=cell(1,numel(net.layers));

% Compute the sizes of layers activations (cell array).
% 若是使用其他数据集，必须在这里替换输入图片大小
sizes=net.getVarSizes({'input',[32 32 3]}); 

for l=1:numel(net.layers)
	if isa(net.layers(l).block,'dagnn.Conv')
		% Get the params from dagnn.params (struct array).
		params={};
		for i=net.layers(l).paramIndexes
		 	params{end+1}=net.params(i).value; 
		end

		% 注意 {params} 而不是 params，因为 struct 会自动解析 {} 里的内容为键值对，必须加 {} 保护
		snn{l}=struct('type','conv','name',net.layers(l).name,...
							'weights',{params},...
							'stride',net.layers(l).block.stride,...
							'pad',net.layers(l).block.pad,...
							'filter',net.layers(l).block.size);

		% outputIndex = layer index + 1，排序规则 
		snn{l}.prev=net.layers(l).inputIndexes-1;  
		snn{l}.shape=sizes{net.layers(l).outputIndexes};
		% snn{l}.mem=zeros([snn{l}.shape num_examples]);

	elseif isa(net.layers(l).block,'dagnn.ReLU')
		snn{l}=struct('type','relu','name',net.layers(l).name);
		snn{l}.prev=net.layers(l).inputIndexes-1;
		snn{l}.shape=sizes{net.layers(l).outputIndexes};

	elseif isa(net.layers(l).block,'dagnn.Pooling')
		snn{l}=struct('type','pool','name',net.layers(l).name,...
							'method',net.layers(l).block.method,...
							'stride',net.layers(l).block.stride,...
							'pad',net.layers(l).block.pad,...
							'filter',net.layers(l).block.poolSize);
		snn{l}.prev=net.layers(l).inputIndexes-1;
		snn{l}.shape=sizes{net.layers(l).outputIndexes};
		snn{l}.scale=1;

	elseif isa(net.layers(l).block,'dagnn.Concat')
		snn{l}=struct('type','concat','name',net.layers(l).name,...
							'dim',net.layers(l).block.dim);
		snn{l}.prev=net.layers(l).inputIndexes-1;
		snn{l}.shape=sizes{net.layers(l).outputIndexes};

		% 基于假设：只有 max_pooling 会出现在单独的 reduction block 支路上
		% 在这个地方进行计算 scale 之路的值，在 simulation 时加进去，修改 max_gate
		for i=(net.layers(l).inputIndexes-1)
			if isa(net.layers(i).block,'dagnn.Pooling') && strcmp(net.layers(i).block.method,'max')
				snn{i}.scale=max_activations(link(i,1))/max_activations(link(i,2));  
			end
		end

	elseif isa(net.layers(l).block,'dagnn.SoftMax')
		snn{l}=struct('type','softmax','name',net.layers(l).name);
		snn{l}.prev=net.layers(l).inputIndexes-1;
		snn{l}.shape=sizes{net.layers(l).outputIndexes};

	end
end
