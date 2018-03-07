function fImage = medFilter(image, N)

%change image to gray scale image
imageGray = rgb2gray(image);

%use NxN 2d med filter on image
fImage = medfilt2(imageGray, [N,N]);

end

