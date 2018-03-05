function [Edges,Ihor, Iver] = edgeExtraction(Iin,B1,B2)

IinDouble=double(Iin);
B1Double=double(B1);
B2Double=double(B2);

Ihor = conv2(B1Double, IinDouble);
Iver = conv2(B2Double, IinDouble);
Edges = sqrt((Iver.^2) + (Ihor.^2));


end

