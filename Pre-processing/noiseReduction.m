function Iout = noiseReduction(I, N)

mask = (1/9)*(ones(N));
Iout = uint8(conv2(mask, I));

end

