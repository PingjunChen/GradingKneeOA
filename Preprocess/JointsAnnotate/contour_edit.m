function contour_edit(ImgPath, ImgExt, MatExt, ResultExt)

chan = 0;
use_cell_score = 0;
files = dir([ImgPath '\*.' ImgExt]);

for i = 1:length(files)
    file_name = files(i).name;
    ImgName = file_name(1:length(file_name)-4);
    be_quit = edit_contour_label(ImgPath, ImgName, ImgExt, MatExt, ResultExt, chan, use_cell_score);
    if be_quit == 1
        break;
    end
end