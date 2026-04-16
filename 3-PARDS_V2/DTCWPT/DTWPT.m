function [y] = DTWPT(x,h_first,h,f,max_level)
%x : input
%h_first : first stage filters([h0_first;h1_first])
%h : dual-tree filters([h0;h1])
%f : the 'same' filters([f0;f1])
%max_level : maximum level 
%y : output cell array containing all of the branches)-- the last row
%gives the full packet

%first stage
fil0=h_first(1,:);
fil1=h_first(2,:);
yy=afb(x,fil0,fil1);
y{1,1}=yy(1,:);
y{1,2}=yy(2,:);

%second stage
fil0=h(1,:);
fil1=h(2,:);
yy=afb(y{1,1},fil0,fil1);
y{2,1}=yy(1,:);
y{2,2}=yy(2,:);

yy=afb(y{1,2},fil0,fil1);
y{2,3}=yy(1,:);
y{2,4}=yy(2,:);

for n=3:max_level,
    for k=1:2^(n-1),
        if mod(k,2^(n-2))==1,
            fil0=h(1,:);
            fil1=h(2,:);
        else
            fil0=f(1,:);
            fil1=f(2,:);
        end
        yy=afb(y{n-1,k},fil0,fil1);
        y{n,2*k-1}=yy(1,:);
        y{n,2*k}=yy(2,:);
    end
end

y = y(max_level,:);