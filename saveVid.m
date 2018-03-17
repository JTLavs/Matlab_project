function [vidObj] = saveVid()

vidObj = VideoWriter('peds.avi');
%set frame rate - normally 30 but 7.5 to slow vid
vidObj.FrameRate = 7.5;

%open vid object
open(vidObj);

path = 'Data\pedestrian\';
file = fullfile(path,'image*.jpg');
pedImgs = dir(file);
pedImgs = {pedImgs.name};

%loop through images writing to the video 
for k = 1:length(pedImgs)
   img = imread(fullfile(path,pedImgs{k}));
   writeVideo(vidObj,img);
end

%close video object
close(vidObj);

end

