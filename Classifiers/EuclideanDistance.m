function dEuc = EuclideanDistance( sample1, sample2 )
    total = 0;
    for t=1:size(sample2,2)
        total = total +  (sample1(t) - sample2(t))^2;
    end
    dEuc = sqrt(double(total));
end

