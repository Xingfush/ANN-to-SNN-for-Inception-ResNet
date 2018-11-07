figure();clf;
plot(rate');
xlabel('layer depth');
grid on

% figure(); clf;
% plot(rate_20);
% hold on
% plot(rate_32);
% hold on
% plot(rate_44);
% hold on
% plot(rate_56)
% grid on

% figure();clf;
% subplot(2,1,1);
% plot(resnet(:,1:150)');
% subplot(2,1,2);
% plot(inception(:,1:150)');


% performance=[];
% for i=1:150
%     if mod(i,8)~=0
%         performance(end+1)=acc(8,i);
%     end
% end

% rate0=[];
% rate1=[];
% rate2=[];
% rate3=[];
% rate4=[];
% rate5=[];
% for i=1:numel(stats)
%     if isa(stats{i},'struct')
%         rate0(end+1)=stats{i}.max_rate(1);
%         rate1(end+1)=stats{i}.max_rate(10);
%         rate2(end+1)=stats{i}.max_rate(50);
%         rate3(end+1)=stats{i}.max_rate(100);
%         rate4(end+1)=stats{i}.max_rate(200);
%         rate5(end+1)=stats{i}.max_rate(300);
%     end
% end
% rate=[rate0;rate1;rate2;rate3;rate4;rate5];
% rate=[];
% for i=1:numel(stats)
%     if isa(stats{i},'struct')
%         rate(end+1)=stats{i}.max_rate(300);
%     end
% end
% expDir=fullfile(vl_rootnn,'data','visualization-1');
% mkdir(expDir);

% for i=1:numel(net.vars)
% 	varDir=fullfile(expDir,net.vars(i).name);
% 	mkdir(varDir);
% 	for s=1:size(net.vars(i).value,4)
% 		samDir=fullfile(varDir,sprintf('sample_%d',s));
% 		mkdir(samDir);
% 		for c=1:size(net.vars(i).value,3)
% 			if c<5
% 				x=net.vars(i).value(:,:,c,s);
% 				imwrite(x*256,jet,fullfile(samDir,sprintf('filter_%d.jpg',c)));
% 			end
% 			% 保存图片进入文件夹
% 			% 图片命名问题
% 		end
% 	end
% end

% for l=[2,4]
% 	varDir=fullfile(expDir,sprintf('relu_%d',l));
% 	mkdir(varDir);
% 	for t=1:100
% 		timeDir=fullfile(varDir,sprintf('time_%d',t));
% 		mkdir(timeDir);
% 		for s=1:size(stats{l}.rate{t},4)
% 			samDir=fullfile(timeDir,sprintf('sample_%d',s));
% 			mkdir(samDir);
% 			for c=1:size(stats{l}.rate{t},3)
% 				if c<5
% 					x=stats{l}.rate{t}(:,:,c,s);
% 					imwrite(x*255,jet,fullfile(samDir,sprintf('filter_%d.jpg',c)));
% 				end
% 			end
% 		end
% 	end
% end

