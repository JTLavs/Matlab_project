boatnois = imread('C:\Users\leann\OneDrive\Documents\MATLAB\Practical2\boatnois.jpg');

B1 = [-1 0 1;-1 0 1;-1 0 1];
B2 = [-1 -1 -1;0 0 0;1 1 1];

boat = edgeExtraction(boatnois,B1,B2);
imagesc(boat);