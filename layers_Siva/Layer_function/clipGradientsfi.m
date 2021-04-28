 function clipGrad = clipGradientsfi(grads, factor)
    k = keys(grads);
    val = values(grads); 
    for i = 1:length(grads)
         val{i} = clip(val{i},-factor,factor);
    end
    if isempty(k)
        clipGrad = containers.Map('empty',0);
    else
        clipGrad = containers.Map(k,val);
    end
 end