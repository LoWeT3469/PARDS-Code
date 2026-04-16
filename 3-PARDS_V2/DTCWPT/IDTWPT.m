function [x] = IDTWPT(y,h_first,h,f)
%full packet inverse transform
%y : the cell array arranged as in DTWPT
%h_first : first stage synthesis filters([h0_first;h1_first])
%h : dual-tree synthesis filters([h0;h1])
%f : the 'same' synthesis filters([f0;f1])

max_level = log2(size(y,2));

xx = y(1,:);

for n = max_level:-1:3,
    for k=1:2^(n-1),
        if mod(k,2^(n-2))==1,
            fil0=h(1,:);
            fil1=h(2,:);
        else
            fil0=f(1,:);
            fil1=f(2,:);
        end
        x2{k} = sfb([xx{2*k-1};xx{2*k}],fil0,fil1);
    end
    xx=x2;
end

%second stage
fil0=h(1,:);
fil1=h(2,:);
x2{1}=sfb([xx{1};xx{2}],fil0,fil1);
x2{2}=sfb([xx{3};xx{4}],fil0,fil1);

%first stage
fil0=h_first(1,:);
fil1=h_first(2,:);
x=sfb([x2{1};x2{2}],fil0,fil1);