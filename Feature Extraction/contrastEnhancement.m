function [Iout] = contrastEnhancement(Iin,m,c)
    for i=1:size(Iin,3)
        Iout(:,:,i) = (m*In(:,:,i)) + c; 
    end
end

