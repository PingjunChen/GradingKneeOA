function [snakeContour Labels ActiveFlag be_quit] = plot_contour_label(rgb, snakeContour0,...
    Labels0, RectSize, cell_score)
    if nargin < 5
        cell_score = [];
    end
    snakeContour = [];
    Labels = [];
    ActiveFlag = [];
    be_quit = [];
    textLabel = {};
    previousevent = '';
    previouscontour = -1;
    Allcolormaps = colormap;
    colormaps = [Allcolormaps(1:6:end,:);  Allcolormaps(1:6:end,:);Allcolormaps(2:6:end,:);...
        Allcolormaps(3:6:end,:); Allcolormaps(4:6:end,:);Allcolormaps(5:6:end,:);Allcolormaps(6:6:end,:)];
    
    be_quit = 0;
    gMFlag = 0;
    gMotion = 0;
    hRect = cell(3,1);%to hold rectangles
    gMCellIndex = 0;
    gCurve1 = [];
    gCurve2 = [];
    MarkerRegion = [];
    RectSize = 15;
    
    count = 1;
    clickCount = 0;
    clickPos = [];
    plot_line_width = 1.6;
    
    num_pts = 2;
    split_index = 2;
    step = 0.1;
    
    fg = figure('Visible', 'on');
    
    set(fg, 'KeyPressFcn', @CorrectContour1, 'WindowButtonDownFcn',@MDragBegin,...
        'WindowButtonMotionFcn',@MDragMotion, 'WindowButtonUpFcn',@MDragDone);
  
    imshow(rgb),hold all
    % hp = impixelinfo;
    Labels = [];
    for iS = 1:length(snakeContour0)
            if ~isempty(cell_score)
                if cell_score(iS) < -0.5
                    continue;
                end
            end
            curSnake = snakeContour0{iS};
            sX = curSnake(1,:);
            sY = curSnake(2,:);    
            % if length(sX) < 10
            %     continue;
            % end
            sX(sX <= 1) = 1;
            sX(sX >= size(rgb, 2)) = size(rgb, 2);

            sY(sY <= 1) = 1;
            sY(sY >= size(rgb, 1)) = size(rgb, 1);

            fdX = sX(:);
            fdY = sY(:);
            if ~isempty(Labels0)
                %hp = plot(fdX, fdY, 'color',getcolormap(Labels0(iS)), 'LineWidth', plot_line_width);
                hp = plot(fdX, fdY, 'color','g', 'LineWidth', plot_line_width);
                Labels = [Labels Labels0(iS)];
            else
                Labels0(iS) = -1;
                %hp = plot(fdX, fdY, 'color',getcolormap(Labels0(iS)), 'LineWidth', plot_line_width);
                hp = plot(fdX, fdY, 'color','g', 'LineWidth', plot_line_width);
                Labels = [Labels  -1];
            end
            thistext =  text(mean(fdX),mean(fdY),num2str(Labels0(iS)),'HorizontalAlignment','right');
            textLabel{count} = thistext;
            
            hContour{count} = hp;
            snakeContour{count} = [fdX';fdY'];
            ActiveFlag(count) = 1;
            count = count + 1;
    end

    
    waitfor(fg);
    
    function MDragBegin(src, event)
        
        [x y] = gpos(gca);
        if gMFlag == 1
            if x >= MarkerRegion(1) && x <= MarkerRegion(1) + MarkerRegion(3) ...
                    && y >= MarkerRegion(2) && y <= MarkerRegion(2) + MarkerRegion(4)
                gMotion = 1;
            else
                clickCount = 0;
                clickPos = []; 
                gMFlag = 0;
                gMotion = 0;
                gMCellIndex = 0;
                
                delete(hRect{1});
                delete(hRect{2});
                delete(hRect{3});
            end
        else
            nnCells = [];
            clickErr = 20;
            for iS2 = 1:(count - 1)
                if ActiveFlag(iS2) == 0
                    continue;
                 end
                 curSnake = snakeContour{iS2};
                 sX = curSnake(1,:);
                 sY = curSnake(2,:);
                  
                 if x > min(sX)-clickErr && x < max(sX)+clickErr && y > min(sY)-clickErr && y < max(sY)+clickErr
                    tmp = (sX(:) - x*ones(length(sX), 1)).^2  + (sY(:) - y*ones(length(sX), 1)).^2;
                    nnCells = [nnCells; iS2 min(tmp)];
                 end
            end
            
            if ~isempty(nnCells)
                 [tmp indice] = sort(nnCells(:, 2));
                 nnCells = nnCells(indice, :);
                 nnCells = [nnCells; 0 Inf];
                clickCount = clickCount + 1;
                clickPos = [clickPos; x y nnCells(1, 1) nnCells(2, 1) nnCells(1, 2) nnCells(2, 2)];
                
               
                if clickCount == 2
                    % check if clickCount are valid
                    gMCellIndex = 0;
                    
                    allCandCell = [];
                    for i = 1:2
                        for j = 1:2
                            if clickPos(1, i+2) == clickPos(2, j+2)
                                if clickPos(1, i+2) > 0
                                    
                                    allCandCell = [allCandCell; clickPos(1, i+2) clickPos(1, i+4) + clickPos(2, j+4)];
                                end
                                
                            end
                        end
                    end
                    if ~isempty(allCandCell)
                        [tmp indice] = sort(allCandCell(:,2));
                        allCandCell = allCandCell(indice, :);
                        gMCellIndex = allCandCell(1, 1);
                    end
                    if gMCellIndex > 0 % valid
                        
                            desX = snakeContour{gMCellIndex}(1, :);
                            desY = snakeContour{gMCellIndex}(2, :);
                            desX = desX';
                            desY = desY';
                            x1 = clickPos(1, 1);
                            y1 = clickPos(1, 2);
                            x2 = clickPos(2, 1);
                            y2 = clickPos(2, 2);

                            N = length(desX);
                            Dis1 = (desX - x1*ones(N, 1)).^2  + (desY - y1*ones(N, 1)).^2;
                            index1 = find(Dis1 == min(Dis1));
                            index1 = index1(1);

                            Dis2 = (desX - x2*ones(N, 1)).^2  + (desY - y2*ones(N, 1)).^2;
                            index2 = find(Dis2 == min(Dis2));
                            index2 = index2(1);

                            if index2 > index1
                                pSet1 = index1:index2;
                                pSet2 = [index2:N 1:index1];
                            else
                                pSet1 = [index1:N 1:index2];
                                pSet2 = index2:index1;

                            end

                            if length(pSet1) > length(pSet2)
                                midIndex = pSet2(1) + 0.5*length(pSet2);
                                gCurve1 = [desX(pSet1)'; desY(pSet1)'];
                                gCurve2 = [x2 x1;y2 y1];
                            else
                                midIndex = pSet1(1) + 0.5*length(pSet1);
                                gCurve1 = [desX(pSet2)'; desY(pSet2)'];
                                gCurve2 = [x1 x2; y1 y2];
                            end

                            midIndex = round(midIndex);

                            if midIndex > N
                                midIndex = midIndex - N;
                            end

                            hRect{1} = rectangle('Position', [desX(index1)-0.5*RectSize desY(index1)-0.5*RectSize RectSize RectSize], 'FaceColor', 'r');
                            hRect{2} = rectangle('Position', [desX(index2)-0.5*RectSize desY(index2)-0.5*RectSize RectSize RectSize], 'FaceColor', 'r');
                            MarkerRegion = [desX(midIndex)-0.5*RectSize desY(midIndex)-0.5*RectSize RectSize RectSize];
                            hRect{3} = rectangle('Position', MarkerRegion, 'FaceColor', 'g');
                            gMFlag = 1;
                    end
                    clickCount = 0;
                    clickPos = []; 
                end
                
            end
           
        end
    end
    
    function MDragMotion(src, event)
        if gMFlag == 1 && gMotion == 1
                [x y] = gpos(gca);
        
                delete(hRect{3});
                MarkerRegion = [x-0.5*RectSize y-0.5*RectSize RectSize RectSize];
                hRect{3} = rectangle('Position', MarkerRegion, 'FaceColor', 'g');

                x1 = gCurve2(1,1);
                x2 = gCurve2(1,2);
                y1 = gCurve2(2,1);
                y2 = gCurve2(2,2);
            

                X = [x1 x x2];
                Y = [y1 y y2];
                t = 1:3;
                ts = 1:0.1:3;
                xx = spline(t,X,ts);
                yy = spline(t,Y,ts);

                [xx yy] = Interp_snake(xx, yy, 1.5);
                xx = xx(:);
                yy = yy(:);
            
                newCurve = [gCurve1 [xx';yy']];

                delete(hContour{gMCellIndex});
                 hp = plot(newCurve(1,:), newCurve(2,:), 'color',getcolormap(Labels(gMCellIndex)), 'LineWidth',plot_line_width);
                hContour{gMCellIndex} = hp;
                
           % end
        end
    end

    function MDragDone(src, event)
        if gMFlag == 1 && gMotion == 1
            [x y] = gpos(gca);
            delete(hRect{3});
            MarkerRegion = [x y RectSize RectSize];
            hRect{3} = rectangle('Position', MarkerRegion, 'FaceColor', 'r');
            
            x1 = gCurve2(1,1);
            x2 = gCurve2(1,2);
            y1 = gCurve2(2,1);
            y2 = gCurve2(2,2);
            
            X = [x1 x x2];
            Y = [y1 y y2];
            t = 1:3;
            ts = 1:0.1:3;
            xx = spline(t,X,ts);
            yy = spline(t,Y,ts);

            [xx yy] = Interp_snake(xx, yy, 1.5);
            xx = xx(:);
            yy = yy(:);
            
            newCurve = [gCurve1 [xx';yy']];
            
            sX = newCurve(1,:);
            sY = newCurve(2,:);
            [fdX fdY] = fourier_descriptor(sX, sY, 10);
            
            delete(hContour{gMCellIndex});
            delete(textLabel{gMCellIndex});
            
            hp = plot(fdX, fdY, 'color', getcolormap(Labels(gMCellIndex)),'LineWidth',plot_line_width);
            thistext =  text(mean(fdX),mean(fdY),getlegend(0),'HorizontalAlignment','right');
            textLabel{gMCellIndex} = thistext;
            hContour{gMCellIndex} = hp;
            snakeContour{gMCellIndex} = [fdX';fdY'];                  
            
            gMFlag = 0;
            gMotion = 0;
            gMCellIndex = 0;
            delete(hRect{1});
            delete(hRect{2});
            delete(hRect{3});
        end
    end

    function CorrectContour1(src, event)
        
        switch event.Character
            case {'d', 'D'} %deletion
                %previousevent = event.Character;
                if gMFlag == 1
                    delete(hRect{1});
                    delete(hRect{2});
                    delete(hRect{3});
                end
                clickCount = 0;
                clickPos = []; 
                gMFlag = 0;
                gMotion = 0;
                gMCellIndex = 0;
            
                [x y] = ginput(1);
                x = round(x);
                y = round(y);
                for iS2 = 1:(count - 1)
                    if ActiveFlag(iS2) == 0
                        continue;
                    end
                    curSnake = snakeContour{iS2};
                    sX = curSnake(1,:);
                    sY = curSnake(2,:);

                    if (x > min(sX) & x < max(sX)) & (y > min(sY) & y < max(sY))
                        tmp = roipoly(rgb, sX, sY);

                        if tmp(y,x) == 1 & ActiveFlag(iS2) == 1
                            delete(hContour{iS2});
                            ActiveFlag(iS2) = 0;
                            %textLabel
                            delete(textLabel{iS2});
                            break;
                        end
                    end
                end
                previousevent = event.Character;
            case {'p','P'}
                  %previousevent = event.Character;
                  if gMFlag == 1
                    delete(hRect{1});
                    delete(hRect{2});
                    delete(hRect{3});
                 end
                 clickCount = 0;
                 clickPos = []; 
                 gMFlag = 0;
                 gMotion = 0;
                 gMCellIndex = 0;
                 
                  [X, Y] = ginput(1);
                  
                  [fdX, fdY]=circlepoints(X,Y,8);
                  hp = plot(fdX, fdY, 'color', 'g', 'LineWidth', plot_line_width);
                  
                  %thistext = text(mean(fdX),mean(fdY),getlegend(0),'HorizontalAlignment','right');
                  %textLabel{count} = thistext;
                  
                  hContour{count} = hp;
                  snakeContour{count} = [fdX';fdY'];
                  Labels = [Labels 0];
                  ActiveFlag(count) = 1;
                  count = count + 1;
                  previousevent = event.Character;
                                      
            case {'a', 'A'} %addition
                %previousevent = event.Character;
                 if gMFlag == 1
                    delete(hRect{1});
                    delete(hRect{2});
                    delete(hRect{3});
                 end
                 clickCount = 0;
                 clickPos = []; 
                 gMFlag = 0;
                 gMotion = 0;
                 gMCellIndex = 0;
                 
                 [X, Y] = ginput(num_pts);
                 % assume in clock-wise orders
%                  tmpX1 = [X(1), X(2)];
%                  tmpY1 = [Y(1), Y(1)];
%                  tmpX2 = [X(2), X(2)];
%                  tmpY2 = [Y(1), Y(2)];
%                  tmpX3 = [X(2), X(1)];
%                  tmpY3 = [Y(2), Y(2)];
%                  tmpX4 = [X(1), X(1)];
%                  tmpY4 = [Y(2), Y(1)];
%                  
%                  t = 1:split_index;
%                  ts = 1:step:split_index;
%                     
%                  xx1 = spline(t,tmpX1,ts);
%                  yy1 = spline(t,tmpY1,ts);
%                  xx1 = xx1(:)';
%                  yy1 = yy1(:)';
%                  
%                  xx2 = spline(t,tmpX2,ts);
%                  yy2 = spline(t,tmpY2,ts);
%                  xx2 = xx2(:)';
%                  yy2 = yy2(:)';
%                     
%                  xx3 = spline(t,tmpX3,ts);
%                  yy3 = spline(t,tmpY3,ts);
%                  xx3 = xx3(:)';
%                  yy3 = yy3(:)';
%                  
%                  xx4 = spline(t,tmpX4,ts);
%                  yy4 = spline(t,tmpY4,ts);
%                  xx4 = xx4(:)';
%                  yy4 = yy4(:)'; 
%                  
%                  
%                  xx = [xx1 xx2(2:end) xx3(2:end) xx4(2:end)];
%                  yy = [yy1 yy2(2:end) yy3(2:end) yy4(2:end)];
%                  
%                  [xx, yy] = Interp_snake(xx, yy, 1.5); 


                 xx = [X(1) X(2) X(2) X(1) X(1)];
                 yy = [Y(1) Y(1) Y(2) Y(2) Y(1)];
                 fdX = xx(:)';
                 fdY = yy(:)';
                 snakeContour{count} = [xx(:)'; yy(:)'];                 
                 
                 

                 Labels = [Labels 0];
                 hp = plot(fdX, fdY, 'color',getcolormap(0), 'LineWidth', plot_line_width);
                 thistext  = text(mean(fdX),mean(fdY),getlegend(0),'HorizontalAlignment','right');
                 hContour{count} = hp;
                 textLabel{count} = thistext;
                 ActiveFlag(count) = 1;
                 count = count + 1;
                 previousevent = event.Character;                    

            case {'1', '2','3','4','5','6','7','8','9','0'} %s for select
                
                [x y] = ginput(1);
                x = round(x);
                y = round(y);
                for iS2 = 1:length(snakeContour)
                    if ActiveFlag(iS2) == 0
                        continue;
                    end
                    curSnake = snakeContour{iS2};
                    sX = curSnake(1,:);
                    sY = curSnake(2,:);
                    
                    if x > min(sX) && x < max(sX) && y > min(sY) && y < max(sY)
                        tmp = roipoly(rgb(:,:,1), sX, sY);                        
                        if tmp(y,x) == 1
                            idlabel = str2num(event.Character);
                            %disp(['current label: ' ,  num2str(idlabel)]);
                            
                            delete(hContour{iS2});
                            delete(textLabel{iS2});
                            
                            Labels(iS2) = idlabel;
                            hContour{iS2} = plot(sX, sY, 'color',getcolormap(Labels(iS2)), 'LineWidth', plot_line_width);
                            % thistext = text(mean(sX),mean(sY),getlegend(Labels(iS2)),'HorizontalAlignment','right');
                            thistext = text(mean(sX),mean(sY),num2str(Labels(iS2)),'HorizontalAlignment','right');
                            textLabel{iS2} = thistext;
                            previouscontour = iS2;
                            break;                            
                      end
                    end    
                end
                previousevent = event.Character;
          case {'c','C'}
                [x y] = ginput(1);
                x = round(x);
                y = round(y);
                for iS2 = 1:length(snakeContour)
                    if ActiveFlag(iS2) == 0
                        continue;
                    end
                    curSnake = snakeContour{iS2};
                    sX = curSnake(1,:);
                    sY = curSnake(2,:);

                    if x > min(sX) && x < max(sX) && y > min(sY) && y < max(sY)
                        tmp = roipoly(rgb(:,:,1), sX, sY);
                        if tmp(y,x) == 1
                                delete(hContour{iS2});
                                Labels(iS2) = 0;
                                hContour{iS2} = plot(sX, sY, 'color',getcolormap(Labels(iS2)), 'LineWidth', plot_line_width);
                                thistext = text(mean(sX),mean(sY),getlegend((iS2)),'HorizontalAlignment','right');
                                textLabel{iS2} = thistext;
                            break;
                        end 
                    end
                    
                end
                previousevent = event.Character; 
          case {'e', 'E'}
                beOK = 1;
                if beOK == 1
                    close all;
                    return;
                else
                    title('Some Fiber Types Are Missing!');
                end
                
          case {'q', 'Q'}
                beOK = 1;
                if beOK == 1
                    be_quit = 1;
                    close all;
                    return;
                else
                    title('Some Fiber Types Are Missing!');
                end
        end
        
    end
    
    function colm = getcolormap(Label)
              
                 if Label == 0 
                          colm =  [0.64,0.21,0.12];
                 elseif Label == -1
                         colm = [0,0,0];
                 elseif Label > size(colormaps,1)
                        colm = [1,1,1];
                 else
                         colm = colormaps(Label,:);
                 end
    end

    function lagends = getlegend(Label)
     classes = {'asd','swf','sd','gwad','fws','sds'};
     
     if Label <= 0 || Label > length(classes)
         lagends = num2str(Label);
     else
         lagends = [num2str(Label),'-' ,classes{Label}];
     end
    end
end