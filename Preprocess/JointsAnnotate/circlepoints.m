function [x, y ]=circlepoints(xCenter,yCenter,radius)

theta = [0: pi/6 :2*pi];
x =reshape(floor( xCenter + radius* cos(theta) ), [] ,1);
y = reshape(floor(yCenter + radius * sin(theta)), [],1 );

end