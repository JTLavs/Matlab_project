function [hogEdEx, hogHor, hogVer, hogIm] = getHogFeatures(images, maskA, maskB)

    hogIm = [];
    hogEdEx = [];
    hogHor = [];
    hogVer = [];
    for i=1:size(images, 1)
        Im = reshape(images(i,:),160,96);
        %ImBrightness =  brightEnchance(Im,50);
        [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);

        hogIm = [hogIm; hog_feature_vector(Im)];
        hogEdEx= [hogEdEx; hog_feature_vector(ImEdEx)];
        hogHor = [hogHor; hog_feature_vector(ImIhor)];
        hogVer = [hogVer; hog_feature_vector(ImIver)];
    end
end