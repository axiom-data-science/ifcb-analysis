function [  ] = img_proc_plot( img, img_pc, img_edge, img_dark, img_blob )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

figure(1), clf
subplot(5,1,1), imshow(img); title('original')
    perimeter = bwboundaries(img_blob, 'noholes');
    subplot(5,1,1), hold on
    for count = 1:length(perimeter),
        plot(perimeter{count}(:,2), perimeter{count}(:,1), 'r')
    end;
subplot(5,1,2), imshow(img_pc); title('phase cong')
subplot(5,1,3), imshow(img_edge); title('edges')
subplot(5,1,4), imshow(img_dark); title('dark areas')
subplot(5,1,5), imshow(img_blob); title('blob')
end
