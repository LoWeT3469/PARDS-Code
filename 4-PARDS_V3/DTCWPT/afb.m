function [y] = afb(x,h0,h1)
%x : input
%h0, h1 : analysis filters
%y : output -> [lowpass_channel ; highpass_channel] 

fil=h0;
temp=conv(fil,x);
temp(1:length(temp)-length(x))=temp(1:length(temp)-length(x))+temp(length(x)+1:end);
take=temp(1:length(x));
y0=take(1:2:length(x));

fil=h1;
temp=conv(fil,x);
temp(1:length(temp)-length(x))=temp(1:length(temp)-length(x))+temp(length(x)+1:end);
take=temp(1:length(x));
y1=take(1:2:length(x));

y=[y0;y1];