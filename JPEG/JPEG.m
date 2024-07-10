clear
url = 'https://www.cs.toronto.edu/%7Ekriz/cifar-10-matlab.tar.gz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder,'cifar-10-matlab.tar.gz');
dataFolder = fullfile(downloadFolder,'cifar-10-batches-mat');
if ~exist(dataFolder,'dir')
    fprintf("Downloading CIFAR-10 dataset (175 MB)... ");
    websave(filename,url);
    untar(filename,downloadFolder);
    fprintf("Done.\n")
end

[XTrain,YTrain,org_img,YValidation] = loadCIFARData(downloadFolder);
org_img = org_img(:,:,:,1:100);

% JPEG圧縮の品質設定（0〜100の範囲、高いほど高品質）
jpeg_quality = 20;

save_folder = 'CIFAR10_JPEG';

% 画像をJPEG形式で圧縮して保存
for i = 1:size(org_img, 4)
  %img = permute(reshape(org_img(i,:), [32, 32, 3]), [2, 1, 3]); % 画像の形式を整える
  img = org_img(:,:,:,i);
  img_filename = sprintf('%s/cifar_image_%d.jpg', save_folder, i);
  imwrite(img, img_filename, 'jpg', 'Quality', jpeg_quality);
end

% 圧縮後のファイルサイズを取得
file_info = dir(fullfile(save_folder, '*.jpg'));
for i = 1:length(file_info)
  fprintf('Image %d の圧縮後のサイズ: %d バイト\n', i, file_info(i).bytes);
end


function [XTrain,YTrain,XTest,YTest] = loadCIFARData(location)

location = fullfile(location,'cifar-10-batches-mat');

[XTrain1,YTrain1] = loadBatchAsFourDimensionalArray(location,'data_batch_1.mat');
[XTrain2,YTrain2] = loadBatchAsFourDimensionalArray(location,'data_batch_2.mat');
[XTrain3,YTrain3] = loadBatchAsFourDimensionalArray(location,'data_batch_3.mat');
[XTrain4,YTrain4] = loadBatchAsFourDimensionalArray(location,'data_batch_4.mat');
[XTrain5,YTrain5] = loadBatchAsFourDimensionalArray(location,'data_batch_5.mat');
XTrain = cat(4,XTrain1,XTrain2,XTrain3,XTrain4,XTrain5);
YTrain = [YTrain1;YTrain2;YTrain3;YTrain4;YTrain5];

[XTest,YTest] = loadBatchAsFourDimensionalArray(location,'test_batch.mat');
end

function [XBatch,YBatch] = loadBatchAsFourDimensionalArray(location,batchFileName)
s = load(fullfile(location,batchFileName));
XBatch = s.data';
XBatch = reshape(XBatch,32,32,3,[]);
XBatch = permute(XBatch,[2 1 3 4]);
YBatch = convertLabelsToCategorical(location,s.labels);
end

function categoricalLabels = convertLabelsToCategorical(location,integerLabels)
s = load(fullfile(location,'batches.meta.mat'));
categoricalLabels = categorical(integerLabels,0:9,s.label_names);
end