 function outputClass = predictMulitinetClassesfi(netSiva, X1, X2)
           A1 = netSiva.Nets{1}.predict(netSiva.Nets{1},X1);
           A2 = netSiva.Nets{2}.predict(netSiva.Nets{2},X2);
           A = netSiva.Nets{3}.predict(netSiva.Nets{3},...
               [A1{netSiva.Nets{1}.no_ofLayer};A2{netSiva.Nets{2}.no_ofLayer}]);
           outputClass = A{end};
 end