function paths = getPedImgs()

%function to loop over that ped images

path = 'Data\pedestrian\';
file = fullfile(path,'image*.jpg');
pedImgs = dir(file);

paths = [];
%loop images in file
for k=1:numel(pedImgs)
    
    I=fullfile(path,pedImgs(k).name);
    paths = [paths; I];
   
end

end

