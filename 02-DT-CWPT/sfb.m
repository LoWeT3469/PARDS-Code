function x = sfb(y,g0,g1)

% y : output from 'afb.m'
% g0,g1 : synthesis filters
% x : reconstructed input

x0 = zeros(1,length(y)*2);
x1 = zeros(1,length(y)*2);

x0(1:2:end) = y(1,1:end);
x1(1:2:end) = y(2,1:end);

x = x0;
fil = g0;
temp = conv(fil,x);
temp(1:length(temp)-length(x)) = temp(1:length(temp)-length(x))+temp(length(x)+1:end);
take = temp(1:length(x));
shift = length(fil);
take0 = take(mod((0:length(x)-1)+shift-1,length(x))+1);

x = x1;
fil = g1;
temp = conv(fil,x);
temp(1:length(temp)-length(x)) = temp(1:length(temp)-length(x))+temp(length(x)+1:end);
take = temp(1:length(x));
shift = length(fil);
take = take(mod((0:length(x)-1)+shift-1,length(x))+1);

x = take+take0;
