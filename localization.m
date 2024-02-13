clc
close all
clear
path= 'E:\wood surface defect\KNOTDATA\';
digitData = imageDatastore(path,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(digitData);
count = labelCount{1,2};

for num = 1:count
    img = readimage(digitData,num);
    rgb  = im2double(img);
%     figure()
%     imshow(img)
    figure()
%     subplot(141)
    imshow(img)
% imwrite(rgb,['E:\wood surface defect\paper_data\promgram\',num2str(3*(num-1)+1),'.jpg'])
    I =rgb2gray(rgb);
%     figure()
%     imhist(I)
     loca_improved_kmeans(I)
end


