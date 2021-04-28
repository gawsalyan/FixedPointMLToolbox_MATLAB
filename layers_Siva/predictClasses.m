 function outputClass = predictClasses(netSiva, X)
           A{1} = netSiva.Layers{1}.predict(netSiva.Layers{1}, X);
           for i = 2:netSiva.no_ofLayer
                A{i} = netSiva.Layers{i}.predict(netSiva.Layers{i},A{i-1});
           end
           outputClass = A{end};
 end