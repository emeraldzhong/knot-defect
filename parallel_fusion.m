clc
clear
close all

load('E:\wood surface defect\wooddefectdata\train_inception_v3.mat');
% single image classification
classNames = net.Layers(end).Classes;
J = imread('E:\wood surface defect\data\378.jpg');
[label1,scores1] = classify(net,J);
h = figure()
imshow(J)
if (label=='dead knot')
  text(80,80,['dead knot,',num2str(100*scores1(classNames == label1),3),'%'],'Color','red','FontSize',14);
else
  text(80,80,['live knot,',num2str(100*scores1(classNames == label1),3),'%'],'Color','red','FontSize',14); 
end
% print('FillPageFigure','-dpdf','-fillpage')
 print(h,'Myreult','-dpng');

% single image localization and area
I= im2double(rgb2gray(J));
loca_improved_kmeans(I);