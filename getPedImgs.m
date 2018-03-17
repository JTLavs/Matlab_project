function [I] = getPedImgs()

%function to loop over that ped images

path = 'Data\pedestrian\';
file = fullfile(path,'image*.jpg');
d = dir(file);

for k=1:numel(d)
    
    I=fullfile(path,d(k).name);
    imshow(I);
end

end

