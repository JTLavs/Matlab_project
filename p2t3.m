vehicle = imread('C:\Users\leann\OneDrive\Documents\MATLAB\Practical2\vehicle.jpg');

vehicle1 = 1.75*vehicle + 25;
vehicle2 = noiseReduction(vehicle1, 3);

Ibinary = vehicle <= 35;

subplot(2,3,1), imshow(vehicle);
subplot(2,3,2), imshow(vehicle2);
subplot(2,3,3), imshow(Ibinary);
subplot(2,3,4), histogram(vehicle, 'BinLimits',[0,256], 'BinWidth', 1), title('Graph1');
subplot(2,3,5), histogram(vehicle2, 'BinLimits',[0,256], 'BinWidth', 1), title('Graph2');;
subplot(2,3,6);

