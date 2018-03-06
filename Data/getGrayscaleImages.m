%% Converts the dataset to grayscale
nnd = imageDatastore('./Data/images','IncludeSubfolders',true,'LabelSource','foldernames');

images = [];
for i = 1:size(nnd.Files)
    cell = nnd.Files(i)
    i = imread(cell{1});
    
    if(size(i,3) == 3)
        g = rgb2gray(i);
        path = replace([cell{1}], 'images','gsImages');
        imwrite(g, fullfile(path), 'jpg');
    else
        imwrite(i, cell{1}, 'jpg');
    end
    
   
end


