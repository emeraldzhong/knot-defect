%inception v3 
clc
clear
close all
%__________________________________________________________________________
digitDatasetPath= 'E:\wood surface defect\KNOTDATA\';
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(266,36);
for i = 1:36
    subplot(6,6,i);
    imshow(digitData.Files{perm(i)});
end
labelCount = countEachLabel(digitData)

img = readimage(digitData,1);
size(img)
[trainDigitData,valDigitData,testDigitdata] = splitEachLabel(digitData,0.6,0.2,'randomize');
lgraph = layerGraph();
%__________________________________________________________________________
tempLayers = [
    imageInputLayer([400 400 3],"Name","input_1","Normalization","rescale-symmetric")
    convolution2dLayer([3 3],32,"Name","conv2d_1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2d_2","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_2","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","conv2d_3","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_3","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Stride",[2 2])
    convolution2dLayer([1 1],80,"Name","conv2d_4","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_4","Epsilon",0.001)
    reluLayer("Name","activation_4_relu")
    convolution2dLayer([3 3],192,"Name","conv2d_5","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","batch_normalization_5","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_2","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_9","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_9","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_10","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_10","Epsilon",0.001)
    reluLayer("Name","activation_10_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_11","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_11","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_7","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_7","Epsilon",0.001)
    reluLayer("Name","activation_7_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_8","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_8","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_1","Padding","same")
    convolution2dLayer([1 1],32,"Name","conv2d_12","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_12","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_6","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_6","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed0");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_16","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_16","Epsilon",0.001)
    reluLayer("Name","activation_16_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_17","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_17","Epsilon",0.001)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_18","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_18","Epsilon",0.001)
    reluLayer("Name","activation_18_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_14","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_14","Epsilon",0.001)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_15","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_15","Epsilon",0.001)
    reluLayer("Name","activation_15_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_2","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv2d_19","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_19","Epsilon",0.001)
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_13","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_13","Epsilon",0.001)
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_23","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_23","Epsilon",0.001)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_24","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_24","Epsilon",0.001)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_25","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_25","Epsilon",0.001)
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_21","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_21","Epsilon",0.001)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_22","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_22","Epsilon",0.001)
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_3","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv2d_26","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_26","Epsilon",0.001)
    reluLayer("Name","activation_26_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_20","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_20","Epsilon",0.001)
    reluLayer("Name","activation_20_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_28","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_28","Epsilon",0.001)
    reluLayer("Name","activation_28_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_29","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_29","Epsilon",0.001)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_30","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_30","Epsilon",0.001)
    reluLayer("Name","activation_30_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],384,"Name","conv2d_27","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_27","Epsilon",0.001)
    reluLayer("Name","activation_27_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_3","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(3,"Name","mixed3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_35","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_35","Epsilon",0.001)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([7 1],128,"Name","conv2d_36","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_36","Epsilon",0.001)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 7],128,"Name","conv2d_37","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_37","Epsilon",0.001)
    reluLayer("Name","activation_37_relu")
    convolution2dLayer([7 1],128,"Name","conv2d_38","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_38","Epsilon",0.001)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_39","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_39","Epsilon",0.001)
    reluLayer("Name","activation_39_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_32","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_32","Epsilon",0.001)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([1 7],128,"Name","conv2d_33","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_33","Epsilon",0.001)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_34","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_34","Epsilon",0.001)
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_4","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_40","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_40","Epsilon",0.001)
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_31","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_31","Epsilon",0.001)
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_45","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_45","Epsilon",0.001)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_46","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_46","Epsilon",0.001)
    reluLayer("Name","activation_46_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_47","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_47","Epsilon",0.001)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_48","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_48","Epsilon",0.001)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_49","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_49","Epsilon",0.001)
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_42","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_42","Epsilon",0.001)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_43","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_43","Epsilon",0.001)
    reluLayer("Name","activation_43_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_44","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_44","Epsilon",0.001)
    reluLayer("Name","activation_44_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_5","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_50","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_50","Epsilon",0.001)
    reluLayer("Name","activation_50_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_41","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_41","Epsilon",0.001)
    reluLayer("Name","activation_41_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_55","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_55","Epsilon",0.001)
    reluLayer("Name","activation_55_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_56","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_56","Epsilon",0.001)
    reluLayer("Name","activation_56_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_57","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_57","Epsilon",0.001)
    reluLayer("Name","activation_57_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_58","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_58","Epsilon",0.001)
    reluLayer("Name","activation_58_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_59","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_59","Epsilon",0.001)
    reluLayer("Name","activation_59_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_52","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_52","Epsilon",0.001)
    reluLayer("Name","activation_52_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_53","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_53","Epsilon",0.001)
    reluLayer("Name","activation_53_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_54","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_54","Epsilon",0.001)
    reluLayer("Name","activation_54_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_6","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_60","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_60","Epsilon",0.001)
    reluLayer("Name","activation_60_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_51","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_51","Epsilon",0.001)
    reluLayer("Name","activation_51_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_65","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_65","Epsilon",0.001)
    reluLayer("Name","activation_65_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_66","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_66","Epsilon",0.001)
    reluLayer("Name","activation_66_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_67","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_67","Epsilon",0.001)
    reluLayer("Name","activation_67_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_68","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_68","Epsilon",0.001)
    reluLayer("Name","activation_68_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_69","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_69","Epsilon",0.001)
    reluLayer("Name","activation_69_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_62","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_62","Epsilon",0.001)
    reluLayer("Name","activation_62_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_63","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_63","Epsilon",0.001)
    reluLayer("Name","activation_63_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_64","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_64","Epsilon",0.001)
    reluLayer("Name","activation_64_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_7","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_70","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_70","Epsilon",0.001)
    reluLayer("Name","activation_70_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_61","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_61","Epsilon",0.001)
    reluLayer("Name","activation_61_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_73","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_73","Epsilon",0.001)
    reluLayer("Name","activation_73_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_74","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_74","Epsilon",0.001)
    reluLayer("Name","activation_74_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_75","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_75","Epsilon",0.001)
    reluLayer("Name","activation_75_relu")
    convolution2dLayer([3 3],192,"Name","conv2d_76","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_76","Epsilon",0.001)
    reluLayer("Name","activation_76_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_71","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_71","Epsilon",0.001)
    reluLayer("Name","activation_71_relu")
    convolution2dLayer([3 3],320,"Name","conv2d_72","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","batch_normalization_72","Epsilon",0.001)
    reluLayer("Name","activation_72_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_4","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(3,"Name","mixed8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],448,"Name","conv2d_81","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_81","Epsilon",0.001)
    reluLayer("Name","activation_81_relu")
    convolution2dLayer([3 3],384,"Name","conv2d_82","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_82","Epsilon",0.001)
    reluLayer("Name","activation_82_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","conv2d_78","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_78","Epsilon",0.001)
    reluLayer("Name","activation_78_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_79","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_79","Epsilon",0.001)
    reluLayer("Name","activation_79_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_80","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_80","Epsilon",0.001)
    reluLayer("Name","activation_80_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_83","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_83","Epsilon",0.001)
    reluLayer("Name","activation_83_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_84","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_84","Epsilon",0.001)
    reluLayer("Name","activation_84_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_8","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_85","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_85","Epsilon",0.001)
    reluLayer("Name","activation_85_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],320,"Name","conv2d_77","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_77","Epsilon",0.001)
    reluLayer("Name","activation_77_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","mixed9_0");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","concatenate_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],448,"Name","conv2d_90","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_90","Epsilon",0.001)
    reluLayer("Name","activation_90_relu")
    convolution2dLayer([3 3],384,"Name","conv2d_91","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_91","Epsilon",0.001)
    reluLayer("Name","activation_91_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","conv2d_87","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_87","Epsilon",0.001)
    reluLayer("Name","activation_87_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_88","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_88","Epsilon",0.001)
    reluLayer("Name","activation_88_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_89","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_89","Epsilon",0.001)
    reluLayer("Name","activation_89_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_92","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_92","Epsilon",0.001)
    reluLayer("Name","activation_92_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_93","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_93","Epsilon",0.001)
    reluLayer("Name","activation_93_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_9","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_94","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_94","Epsilon",0.001)
    reluLayer("Name","activation_94_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],320,"Name","conv2d_86","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","batch_normalization_86","Epsilon",0.001)
    reluLayer("Name","activation_86_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","mixed9_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","concatenate_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","mixed10")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(2,"Name","predictions")
    softmaxLayer("Name","predictions_softmax")
    classificationLayer("Name","ClassificationLayer_predictions")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers;

lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_9");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_7");
lgraph = connectLayers(lgraph,"max_pooling2d_2","average_pooling2d_1");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_6");
lgraph = connectLayers(lgraph,"activation_6_relu","mixed0/in1");
lgraph = connectLayers(lgraph,"activation_8_relu","mixed0/in2");
lgraph = connectLayers(lgraph,"activation_11_relu","mixed0/in3");
lgraph = connectLayers(lgraph,"activation_12_relu","mixed0/in4");
lgraph = connectLayers(lgraph,"mixed0","conv2d_16");
lgraph = connectLayers(lgraph,"mixed0","conv2d_14");
lgraph = connectLayers(lgraph,"mixed0","average_pooling2d_2");
lgraph = connectLayers(lgraph,"mixed0","conv2d_13");
lgraph = connectLayers(lgraph,"activation_13_relu","mixed1/in1");
lgraph = connectLayers(lgraph,"activation_15_relu","mixed1/in2");
lgraph = connectLayers(lgraph,"activation_18_relu","mixed1/in3");
lgraph = connectLayers(lgraph,"activation_19_relu","mixed1/in4");
lgraph = connectLayers(lgraph,"mixed1","conv2d_23");
lgraph = connectLayers(lgraph,"mixed1","conv2d_21");
lgraph = connectLayers(lgraph,"mixed1","average_pooling2d_3");
lgraph = connectLayers(lgraph,"mixed1","conv2d_20");
lgraph = connectLayers(lgraph,"activation_20_relu","mixed2/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","mixed2/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","mixed2/in3");
lgraph = connectLayers(lgraph,"activation_26_relu","mixed2/in4");
lgraph = connectLayers(lgraph,"mixed2","conv2d_28");
lgraph = connectLayers(lgraph,"mixed2","conv2d_27");
lgraph = connectLayers(lgraph,"mixed2","max_pooling2d_3");
lgraph = connectLayers(lgraph,"activation_27_relu","mixed3/in1");
lgraph = connectLayers(lgraph,"activation_30_relu","mixed3/in2");
lgraph = connectLayers(lgraph,"max_pooling2d_3","mixed3/in3");
lgraph = connectLayers(lgraph,"mixed3","conv2d_35");
lgraph = connectLayers(lgraph,"mixed3","conv2d_32");
lgraph = connectLayers(lgraph,"mixed3","average_pooling2d_4");
lgraph = connectLayers(lgraph,"mixed3","conv2d_31");
lgraph = connectLayers(lgraph,"activation_31_relu","mixed4/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","mixed4/in2");
lgraph = connectLayers(lgraph,"activation_39_relu","mixed4/in3");
lgraph = connectLayers(lgraph,"activation_40_relu","mixed4/in4");
lgraph = connectLayers(lgraph,"mixed4","conv2d_45");
lgraph = connectLayers(lgraph,"mixed4","conv2d_42");
lgraph = connectLayers(lgraph,"mixed4","average_pooling2d_5");
lgraph = connectLayers(lgraph,"mixed4","conv2d_41");
lgraph = connectLayers(lgraph,"activation_41_relu","mixed5/in1");
lgraph = connectLayers(lgraph,"activation_44_relu","mixed5/in2");
lgraph = connectLayers(lgraph,"activation_49_relu","mixed5/in3");
lgraph = connectLayers(lgraph,"activation_50_relu","mixed5/in4");
lgraph = connectLayers(lgraph,"mixed5","conv2d_55");
lgraph = connectLayers(lgraph,"mixed5","conv2d_52");
lgraph = connectLayers(lgraph,"mixed5","average_pooling2d_6");
lgraph = connectLayers(lgraph,"mixed5","conv2d_51");
lgraph = connectLayers(lgraph,"activation_51_relu","mixed6/in1");
lgraph = connectLayers(lgraph,"activation_54_relu","mixed6/in2");
lgraph = connectLayers(lgraph,"activation_59_relu","mixed6/in3");
lgraph = connectLayers(lgraph,"activation_60_relu","mixed6/in4");
lgraph = connectLayers(lgraph,"mixed6","conv2d_65");
lgraph = connectLayers(lgraph,"mixed6","conv2d_62");
lgraph = connectLayers(lgraph,"mixed6","average_pooling2d_7");
lgraph = connectLayers(lgraph,"mixed6","conv2d_61");
lgraph = connectLayers(lgraph,"activation_61_relu","mixed7/in1");
lgraph = connectLayers(lgraph,"activation_64_relu","mixed7/in2");
lgraph = connectLayers(lgraph,"activation_69_relu","mixed7/in3");
lgraph = connectLayers(lgraph,"activation_70_relu","mixed7/in4");
lgraph = connectLayers(lgraph,"mixed7","conv2d_73");
lgraph = connectLayers(lgraph,"mixed7","conv2d_71");
lgraph = connectLayers(lgraph,"mixed7","max_pooling2d_4");
lgraph = connectLayers(lgraph,"activation_72_relu","mixed8/in1");
lgraph = connectLayers(lgraph,"activation_76_relu","mixed8/in2");
lgraph = connectLayers(lgraph,"max_pooling2d_4","mixed8/in3");
lgraph = connectLayers(lgraph,"mixed8","conv2d_81");
lgraph = connectLayers(lgraph,"mixed8","conv2d_78");
lgraph = connectLayers(lgraph,"mixed8","average_pooling2d_8");
lgraph = connectLayers(lgraph,"mixed8","conv2d_77");
lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_79");
lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_80");
lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_83");
lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_84");
lgraph = connectLayers(lgraph,"activation_79_relu","mixed9_0/in1");
lgraph = connectLayers(lgraph,"activation_80_relu","mixed9_0/in2");
lgraph = connectLayers(lgraph,"activation_83_relu","concatenate_1/in1");
lgraph = connectLayers(lgraph,"activation_84_relu","concatenate_1/in2");
lgraph = connectLayers(lgraph,"activation_77_relu","mixed9/in1");
lgraph = connectLayers(lgraph,"mixed9_0","mixed9/in2");
lgraph = connectLayers(lgraph,"concatenate_1","mixed9/in3");
lgraph = connectLayers(lgraph,"activation_85_relu","mixed9/in4");
lgraph = connectLayers(lgraph,"mixed9","conv2d_90");
lgraph = connectLayers(lgraph,"mixed9","conv2d_87");
lgraph = connectLayers(lgraph,"mixed9","average_pooling2d_9");
lgraph = connectLayers(lgraph,"mixed9","conv2d_86");
lgraph = connectLayers(lgraph,"activation_87_relu","conv2d_88");
lgraph = connectLayers(lgraph,"activation_87_relu","conv2d_89");
lgraph = connectLayers(lgraph,"activation_91_relu","conv2d_92");
lgraph = connectLayers(lgraph,"activation_91_relu","conv2d_93");
lgraph = connectLayers(lgraph,"activation_88_relu","mixed9_1/in1");
lgraph = connectLayers(lgraph,"activation_89_relu","mixed9_1/in2");
lgraph = connectLayers(lgraph,"activation_92_relu","concatenate_2/in1");
lgraph = connectLayers(lgraph,"activation_93_relu","concatenate_2/in2");
lgraph = connectLayers(lgraph,"activation_86_relu","mixed10/in1");
lgraph = connectLayers(lgraph,"mixed9_1","mixed10/in2");
lgraph = connectLayers(lgraph,"concatenate_2","mixed10/in3");
lgraph = connectLayers(lgraph,"activation_94_relu","mixed10/in4");

%__________________________________________________________________________
options = trainingOptions('sgdm',...
    'MiniBatchSize',64, ...
    'MaxEpochs',45, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Shuffle','every-epoch',...
    'Verbose',true,...
    'Plots','training-progress');

net = trainNetwork(trainDigitData,lgraph,options);
save("train_inception_v3.mat",net);
%__________________________________________________________________________
[YPred,predictedLabels] = classify(net,testDigitdata);
idx = randperm(numel(testDigitdata.Files),16);
figure()
for k = 1:16
    subplot(4,4,k)
    I = readimage(testDigitdata, idx(k));
    imshow(I)
    label = YPred(idx(k));
    if (label=='dead knot')
        title('dead knot');
    else 
        title('live knot');
    end
end
testLabels = testDigitdata.Labels;

accuracy = sum(YPred == testLabels)/numel(testLabels)
figure()
confusionchart(testLabels,YPred)