clear all;
close all;
load dtcwpt_filters.mat;
load dtcwpt_filters_long.mat;

max_level=4;

% Read CSV file into a table
flow = readtable('flow.csv');
flow_HT = readtable('flow_HT.csv');

% Convert table to array of doubles
flow = table2array(flow);
flow_HT = table2array(flow_HT);

% Transpose the doubles
flow = flow';
flow_HT = flow_HT';

y1 = DTWPT(flow,first_1,h,f,max_level);
y2 = DTWPT(flow_HT,first_2,g,f,max_level);

x1 = IDTWPT(y1,first_1(:,end:-1:1),h(:,end:-1:1),f(:,end:-1:1));
x2 = IDTWPT(y2,first_2(:,end:-1:1),g(:,end:-1:1),f(:,end:-1:1));

max(abs(flow-x2))