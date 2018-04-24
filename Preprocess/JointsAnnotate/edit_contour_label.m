% Modify existing results
function be_quit = edit_contour_label(ImgPath, ImgName, ImgExt, MatExt, DesSuffix, chan, use_cell_score,finalImgExt,resizescale)
if ~exist('resizescale','var')
   resizescale = 1;
end
close all;
rgb = imread(fullfile(ImgPath, [ImgName '.' ImgExt]));
rgb = imresize(rgb,resizescale);



Img_Info_Path = fullfile(ImgPath , [ImgName '_' MatExt '.mat']);

if exist(Img_Info_Path, 'file') 
    load(Img_Info_Path);
else
    %return;
    Contours = [];
    Labels = [];
end

%------------resize the contour for better annotation----------
    for i = 1:length(Contours)
      
        Contours{i} = Contours{i}*resizescale;
        %pause
    end
%----------------------------------------------------------------

if chan > 0
    rgb = rgb(:,:,chan);
end

figure, imshow(rgb),title(ImgName)
%%
if ~exist('Labels','var')
Labels = ones(1, length(Contours));
end
rectsize = 5;
[CellContours1 Labels ActiveFlag be_quit] = plot_contour_label(rgb, Contours, Labels, rectsize);

Contours = CellContours1(ActiveFlag > 0);
Labels = Labels(ActiveFlag > 0);

% figure('Visible', 'off') 
figure,
imshow(rgb), hold on

for i = 1:length(Contours)
    cc = Contours{i};
    sX = cc(1,:);
    sY = cc(2,:);
    switch Labels(i)
        case 0
            plot(sX, sY, 'g', 'LineWidth', 1.5)
        case 1
            plot(sX, sY, 'r', 'LineWidth', 1.5)
    end
    Contours{i} = Contours{i}/resizescale;
    %pause
end
hold off


save(fullfile(ImgPath , [ImgName '_' DesSuffix '.mat']), 'Contours', 'Labels');

dialog.outputPath = ImgPath;
dialog.outputFilename = [ImgName '_' DesSuffix];


N = 4;
set(gcf, 'Units', 'pixels', 'Position', [32, 32, size(rgb, 2) / N, size(rgb, 1) / N]);
set(gca, 'Units', 'normalized', 'Position', [0,0,1,1]);
set(gca, 'Units', 'points');
set(gcf, 'Units', 'points', 'PaperUnits', 'points', 'PaperPositionMode', 'auto');
