clear all;
close all;
load dtcwpt_filters;
load dtcwpt_filters_long;

%check PR
max_level=4;

xx=rand(1,512);

y1 = DTWPT(xx,first_1,h,f,max_level);
y2 = DTWPT(xx,first_2,g,f,max_level);


x1 = IDTWPT(y1,first_1(:,end:-1:1),h(:,end:-1:1),f(:,end:-1:1));
x2 = IDTWPT(y2,first_2(:,end:-1:1),g(:,end:-1:1),f(:,end:-1:1));

max(abs(xx-x2))
%% View the DTFT of a wavelet
xx=zeros(1,1024);

y1 = DTWPT(xx,first_1,h,f,max_level);
y2 = DTWPT(xx,first_2,g,f,max_level);

sb=7;
y1{sb}(30)=1;
y2{sb}(30)=1;

x1 = IDTWPT(y1,first_1(:,end:-1:1),h(:,end:-1:1),f(:,end:-1:1));
x2 = IDTWPT(y2,first_2(:,end:-1:1),g(:,end:-1:1),f(:,end:-1:1));

len_fft=1024;
b=fft(x1-i*x2,len_fft);

plot((-len_fft/2+1:len_fft/2)/len_fft,abs(fftshift(b)));

set(gca,'Xtick',[-0.5:0.25:0.5])
set(gca,'FontName','Symbol');set(gca,'XTickLabel',{'-p','-p/2','0','p/2','p'});
set(gca,'Ytick',[]);