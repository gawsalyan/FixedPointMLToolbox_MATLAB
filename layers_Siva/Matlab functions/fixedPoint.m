function fxpInt16 = fixedPoint(dblIn,FracBits)
% Fixed Point Implementation 
% 16 bit word size
% 13 fraction bit
% signed


fxpInt16 = (typecast(swapbytes(int16((dblIn)*2^FracBits)),'uint8'));
%fxpInt16 = swapbytes(typecast(swapbytes(int16((dblIn)*2^FracBits)),'uint8'));
%fxpInt16(1) = fxpInt16(1) + (floor((dblIn))*2^(5));


% if dblIn < 0
%     dblIn = 2^2 + dblIn;
%     fxpInt16 = swapbytes(typecast(swapbytes(int16(mod(abs(dblIn),1)*2^FracBits)),'uint8'));
%     fxpInt16(1) = fxpInt16(1) + (floor(abs(dblIn))*2^(5)) + 2^7;
% else
%     fxpInt16 = swapbytes(typecast(swapbytes(int16(mod(abs(dblIn),1)*2^FracBits)),'uint8'));
%     fxpInt16(1) = fxpInt16(1) + (floor(abs(dblIn))*2^(5));
% end
%for sign bit
%if dblIn < 0
%  fxpInt16(1) = fxpInt16(1) + 2^7;  
%end

%fxpInt16 = round(dblIn * 2^FracBits);

end