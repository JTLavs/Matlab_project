boatnois = imread('C:\Users\leann\OneDrive\Documents\MATLAB\Practical2\boatnois.jpg');


maskB = 1/9*ones(3);

boat1 = uint8(conv2(maskB, boatnois));
boat2 = uint8(filter2(maskB, boatnois));
s
maskC = 1/9*ones(5);

boat3 = uint8(conv2(maskC, boatnois));

subplot(2,2,1), imshow(boatnois);
subplot(2,2,2), imshow(boat1);
subplot(2,2,3), imshow(boat2);
subplot(2,2,4), imshow(boat3);

imshow(noiseReduction(boatnois, 3));