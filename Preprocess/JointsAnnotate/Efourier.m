function [result] = Efourier(outline, n)
    rFSDs = fEfourier(outline, n, 0, 0);
    result = rEfourier( rFSDs, n,length(outline));
end