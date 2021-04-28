function lstmLayer = lstmLayerInit(n_a, n_f, n_y, weightFactor)

lstmLayer = containers.Map;
lstmLayer('n_a') = n_a;
lstmLayer('n_f') = n_f;
lstmLayer('n_y') = n_y;
lstmLayer('Wf') = rand(n_a,n_a+n_f) * weightFactor;  
lstmLayer('bf') = rand(n_a,n_f) * weightFactor; 
lstmLayer('Wi') = rand(n_a,n_a+n_f) * weightFactor;  
lstmLayer('bi') = rand(n_a,n_f) * weightFactor; 
lstmLayer('Wc') = rand(n_a,n_a+n_f) * weightFactor;  
lstmLayer('bc') = rand(n_a,n_f) * weightFactor; 
lstmLayer('Wo') = rand(n_a,n_a+n_f) * weightFactor;  
lstmLayer('bo') = rand(n_a,n_f) * weightFactor; 
lstmLayer('Wy') = rand(n_y,n_a) * weightFactor;  
lstmLayer('by') = rand(n_y,n_f) * weightFactor; 

end